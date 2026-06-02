"""Load an SC-enabled draft + target model pair for speculative decoding.

Default pair follows Hugging Face's assisted-generation recommendation:

    draft  = meta-llama/Llama-3.2-1B-Instruct
    target = meta-llama/Llama-3.1-8B-Instruct

Same family / shared tokenizer, so draft proposals are directly verifiable by
the target. Both models are passed through :func:`sc_model.make_sc_model`, so
*every* matmul in both is simulated in stochastic computing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .sc_model import make_sc_model

DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class SpecModels:
    target: Any
    draft: Any
    tokenizer: Any


def load_spec_models(
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
    *,
    dtype: torch.dtype = torch.float16,
    device_map: Any = "auto",
    target_sc_overrides: Optional[dict] = None,
    draft_sc_overrides: Optional[dict] = None,
    target_max_memory: Optional[dict] = None,
    draft_max_memory: Optional[dict] = None,
) -> SpecModels:
    """Load both models SC-enabled plus a shared tokenizer.

    ``*_sc_overrides`` let you configure the two models independently — e.g.
    run the draft in fp16 (``{"use_sc_attn": False, "use_sc_linear": False}``)
    while keeping the target in SC, to isolate where SC noise costs you
    acceptance rate.

    ``*_max_memory`` (optional) are passed straight to ``from_pretrained`` as
    ``max_memory``. On a single GPU this lets you cap how much of each model
    sits on-device (offloading the rest to CPU) so there is headroom left for
    the SC kernels' scratch buffers — otherwise both models pack the card and
    the SC path OOMs. Left ``None``, behavior is unchanged (plain auto-dispatch).
    """
    from transformers import AutoTokenizer

    tgt_kw = {"max_memory": target_max_memory} if target_max_memory else {}
    drf_kw = {"max_memory": draft_max_memory} if draft_max_memory else {}

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    target = make_sc_model(target_model, torch_dtype=dtype, device_map=device_map,
                           sc_overrides=target_sc_overrides, **tgt_kw)
    draft = make_sc_model(draft_model, torch_dtype=dtype, device_map=device_map,
                          sc_overrides=draft_sc_overrides, **drf_kw)
    target.eval()
    draft.eval()
    return SpecModels(target=target, draft=draft, tokenizer=tokenizer)
