"""SC integration layer — make any HF causal LM run all matmuls through SC.

This is the speculative-decoding repo's self-contained port of
``scmp_llm_llama/model/sc_common.py``. Both the draft and the target model
are passed through :func:`make_sc_model`, after which:

  * every ``nn.Linear`` inside the decoder layers is replaced with
    :class:`SCLinear`, whose forward dispatches to ``scmp_kernels.sc_matmul``;
  * the model family's ``eager_attention_forward`` (Q·Kᵀ and softmax·V) is
    monkey-patched with :func:`sc_eager_attention_forward`.

So *all* GEMMs that dominate the model — QKV/O projections, the MLP, the LM
head's siblings, and both attention score matmuls — are simulated in
stochastic computing. (The LM head itself is left in fp by default; flip
``replace_lm_head=True`` to include it.)

SC knobs live on ``model.config`` (see :data:`SC_CONFIG_DEFAULTS`) and can be
toggled per-model at runtime, which is convenient for sweeps: e.g. run the
draft in fp16 and only the target in SC, or vice-versa.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Iterable, Optional

import torch
from torch import nn

# Repo-wide kernel defaults: bit-reversed Owen scramble mask + scrambling on the
# rescale path (e.g. when halve / short stoc_len rescales onto a coarser grid).
# scmp_kernels reads these from the environment at launch; setdefault so an
# explicit env var still wins. Per-row granularity for *everything* (attention
# included) is set via SC_CONFIG_DEFAULTS below. NOTE: these defaults are
# scoped to this repo — they intentionally do NOT change scmp_kernels' own
# defaults (which scmp_llm's benchmarks rely on).
os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_RESCALE", "1")

try:
    from scmp_kernels import sc_matmul as _sc_matmul
    _HAS_SC = True
except ImportError:  # pragma: no cover - exercised only without the kernels
    _sc_matmul = None
    _HAS_SC = False


# Per-row everywhere: both the attention score/context matmuls and the
# nn.Linear path use per_row granularity (scmp_llm_llama defaults attention to
# per_head; here we make everything per_row per the repo convention).
SC_CONFIG_DEFAULTS = {
    "use_sc_attn": True,
    "use_sc_linear": True,
    "sc_prec": 8,
    "sc_stoc_len": 256,
    "sc_mode": "bipolar",
    "sc_granularity": "per_row",         # attention score / context matmuls
    "sc_linear_granularity": "per_row",  # nn.Linear path
    "sc_linear_chunk_d": 128,
    "sc_halve_bipolar": True,            # uSystolic cycle-halving (2^(sc_prec-1)
                                         # cycles); the rescale path it triggers
                                         # is scrambled by default (SC_SCRAMBLE_
                                         # RESCALE=1), keeping quality (x1.056).
}


def apply_sc_config_defaults(config) -> None:
    """Set any missing SC knob on ``config`` to its default."""
    for k, v in SC_CONFIG_DEFAULTS.items():
        if not hasattr(config, k):
            setattr(config, k, v)


# ---------------------------------------------------------------------------
# SCLinear
# ---------------------------------------------------------------------------

class SCLinear(nn.Linear):
    """``nn.Linear`` whose forward routes ``x @ Wᵀ`` through ``sc_matmul``.

    Weight/bias semantics and ``state_dict`` keys match the parent exactly so
    HF checkpoints load unchanged. Falls back to ``F.linear`` when SC is
    unavailable or ``config.use_sc_linear`` is False.
    """

    def __init__(self, in_features, out_features, bias=True, *, sc_config=None, **kw):
        super().__init__(in_features, out_features, bias=bias, **kw)
        self._sc_config = sc_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self._sc_config
        use_sc = _HAS_SC and bool(getattr(config, "use_sc_linear", True))
        if not use_sc:
            return nn.functional.linear(x, self.weight, self.bias)

        sc_gran = getattr(config, "sc_linear_granularity", "per_row")
        sc_mode = getattr(config, "sc_mode", "bipolar")
        sc_prec = int(getattr(config, "sc_prec", 8))
        sc_stoc_len = int(getattr(config, "sc_stoc_len", 256))
        sc_chunk_d = int(getattr(config, "sc_linear_chunk_d", 128))
        sc_halve = bool(getattr(config, "sc_halve_bipolar", False))
        # Under halve, hand stoc_len=None so the kernel sets the cycle count and
        # rng grid to 2^(sc_prec-1); passing 256 would defeat the halving.
        eff_stoc_len = None if sc_halve else sc_stoc_len

        orig_dtype = x.dtype
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).to(torch.float32).contiguous()
        w_fp32 = self.weight.to(torch.float32).contiguous()
        out_flat = _sc_matmul(
            x_flat, w_fp32,
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=eff_stoc_len, chunk_d=sc_chunk_d,
            halve_bipolar_stoc_len=sc_halve,
        )
        out = out_flat.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linears_with_sc(
    root: nn.Module,
    *,
    config,
    layer_filter,
    skip_names: Iterable[str] = (),
) -> int:
    """Replace every ``nn.Linear`` under a ``layer_filter``-marked subtree.

    Reuses the original weight/bias tensors (no extra memory). Returns the
    number of replacements.
    """
    skip = set(skip_names)
    n_replaced = 0

    def _swap_within(parent: nn.Module) -> None:
        nonlocal n_replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, SCLinear):
                if name in skip:
                    continue
                new_lin = SCLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, sc_config=config,
                )
                new_lin.weight = child.weight
                if child.bias is not None:
                    new_lin.bias = child.bias
                # Preserve accelerate's dispatch/offload hook. Under
                # device_map="auto" + CPU offload the original Linear carries an
                # AlignDevicesHook whose weights_map holds the real CPU weights
                # while the module param is a `meta` placeholder. Dropping the
                # hook leaves the SCLinear with a meta weight that is never
                # materialized at forward -> "Tensor on device meta" crash.
                hf_hook = getattr(child, "_hf_hook", None)
                if hf_hook is not None:
                    from accelerate.hooks import (
                        add_hook_to_module,
                        remove_hook_from_module,
                    )
                    remove_hook_from_module(child)
                    add_hook_to_module(new_lin, hf_hook)
                else:
                    new_lin.to(child.weight.device, dtype=child.weight.dtype)
                setattr(parent, name, new_lin)
                n_replaced += 1
            else:
                _swap_within(child)

    def _walk(parent: nn.Module) -> None:
        for child in parent.children():
            if layer_filter(child):
                _swap_within(child)
            else:
                _walk(child)

    _walk(root)
    return n_replaced


# ---------------------------------------------------------------------------
# SC attention forward — drop-in for HF's eager_attention_forward
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expansion: (B, H_kv, N, D) -> (B, H_kv*n_rep, N, D)."""
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv * n_rep, slen, head_dim)


def _sc_attention_matmul_ab_t(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    granularity: str,
    mode: str,
    sc_prec: int,
    stoc_len: Optional[int],
    halve: bool = False,
) -> torch.Tensor:
    """4D-aware wrapper computing ``a @ b.T`` via sc_matmul.

    a: (B, H, N, K), b: (B, H, M, K) -> (B, H, N, M).
    """
    orig_dtype = a.dtype
    B, H, N, K = a.shape
    M = b.shape[-2]
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()
    out3 = _sc_matmul(
        a3, b3,
        granularity=granularity, mode=mode,
        sc_prec=sc_prec, stoc_len=stoc_len,
        halve_bipolar_stoc_len=halve,
    )
    return out3.reshape(B, H, N, M).to(orig_dtype)


def sc_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in for HF's ``eager_attention_forward``; routes both score matmuls
    through SC when ``module.config.use_sc_attn`` is set. Ignores the
    ``sliding_window`` kwarg some families pass through."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    config = getattr(module, "config", None)
    use_sc = _HAS_SC and bool(getattr(config, "use_sc_attn", True))
    sc_gran = getattr(config, "sc_granularity", "per_head")
    sc_mode = getattr(config, "sc_mode", "bipolar")
    sc_prec = int(getattr(config, "sc_prec", 8))
    sc_stoc_len = int(getattr(config, "sc_stoc_len", 256))
    sc_halve = bool(getattr(config, "sc_halve_bipolar", False))
    eff_stoc_len = None if sc_halve else sc_stoc_len

    if use_sc:
        attn_weights = _sc_attention_matmul_ab_t(
            query, key_states,
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=eff_stoc_len, halve=sc_halve,
        ) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)

    if use_sc:
        attn_output = _sc_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=eff_stoc_len, halve=sc_halve,
        )
    else:
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# make_sc_model — generic loader+patcher for any HF causal LM
# ---------------------------------------------------------------------------

def _find_decoder_layers(model: nn.Module):
    """Return the ``ModuleList`` of decoder layers for a HF causal LM.

    Tries the conventional ``model.model.layers`` first, then falls back to
    the first ``ModuleList`` whose elements look like decoder layers.
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0 and \
                m[0].__class__.__name__.lower().endswith("decoderlayer"):
            return m
    raise RuntimeError(
        "could not locate decoder layers; pass a custom layer_filter to "
        "make_sc_model")


def make_sc_model(
    model_path: str,
    *,
    torch_dtype: torch.dtype = torch.float16,
    device_map: Any = "auto",
    sc_overrides: Optional[dict] = None,
    replace_lm_head: bool = False,
    **from_pretrained_kwargs,
):
    """Load an HF causal LM with full SC integration and return it.

    Forces ``attn_implementation="eager"`` (HF's default ``sdpa`` silently
    bypasses the patched attention). All matmuls inside the decoder layers,
    plus both attention score matmuls, are routed through ``sc_matmul``.

    Args:
        model_path: HF model id or local path.
        sc_overrides: dict of SC knobs to set on ``model.config`` after defaults.
        replace_lm_head: also swap the final ``lm_head`` Linear for SC.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    apply_sc_config_defaults(model.config)
    if sc_overrides:
        for k, v in sc_overrides.items():
            setattr(model.config, k, v)

    # Patch the family's module-global eager_attention_forward in place.
    mod = importlib.import_module(type(model).__module__)
    if hasattr(mod, "eager_attention_forward"):
        mod.eager_attention_forward = sc_eager_attention_forward

    layers = _find_decoder_layers(model)
    layer_ids = {id(l) for l in layers}
    n = replace_linears_with_sc(
        model, config=model.config,
        layer_filter=lambda m: id(m) in layer_ids,
    )
    if replace_lm_head and isinstance(getattr(model, "lm_head", None), nn.Linear) \
            and not isinstance(model.lm_head, SCLinear):
        head = model.lm_head
        sc_head = SCLinear(head.in_features, head.out_features,
                           bias=head.bias is not None, sc_config=model.config)
        sc_head.weight = head.weight
        if head.bias is not None:
            sc_head.bias = head.bias
        # Preserve accelerate's offload hook (see replace_linears_with_sc).
        hf_hook = getattr(head, "_hf_hook", None)
        if hf_hook is not None:
            from accelerate.hooks import add_hook_to_module, remove_hook_from_module
            remove_hook_from_module(head)
            add_hook_to_module(sc_head, hf_hook)
        else:
            sc_head.to(head.weight.device, dtype=head.weight.dtype)
        model.lm_head = sc_head
        n += 1

    model._sc_linear_replacements = n
    return model
