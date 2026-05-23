"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small *draft* model proposes ``gamma`` tokens autoregressively; the large
*target* model verifies all of them in a single forward pass. Each draft
token ``x`` drawn from draft distribution ``q`` is accepted with probability
``min(1, p(x)/q(x))`` where ``p`` is the target distribution; on the first
rejection we resample from the normalized residual ``(p - q)₊``. When every
draft token is accepted we additionally emit a free *bonus* token from the
target's next-position distribution. This procedure is provably equivalent to
sampling from the target model alone — see the references — so SC noise in the
two models only changes the *speed* (acceptance rate), not the target
distribution being matched.

The implementation is deliberately KV-cache-free: every iteration re-runs the
full forward on the running sequence. That is O(T) wasteful, but the dominant
cost here is the SC matmul *simulation*, and the cache-free path keeps the
accept/reject logic obviously correct and family-agnostic. The point of this
repo is measuring how SC precision affects acceptance / output quality, not
wall-clock throughput.

``generate`` is greedy when ``do_sample=False`` (temperature ignored): greedy
speculative decoding is exact iff the target's argmax matches the draft token,
and the emitted sequence is bit-identical to plain target greedy decoding — a
property the test-suite checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SpecDecodeStats:
    """Per-run telemetry. ``acceptance_rate`` is the headline SC-quality metric."""
    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    new_tokens: int = 0
    accepted_per_step: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted / self.steps if self.steps else 0.0

    def __str__(self) -> str:
        return (f"steps={self.steps} proposed={self.proposed} "
                f"accepted={self.accepted} "
                f"accept_rate={self.acceptance_rate:.3f} "
                f"mean_accepted/step={self.mean_accepted_per_step:.2f} "
                f"new_tokens={self.new_tokens}")


def _next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits for the next position only: (1, vocab)."""
    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out
    return logits[:, -1, :]


def _all_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits at every position: (1, seq, vocab)."""
    out = model(input_ids)
    return out.logits if hasattr(out, "logits") else out


def _probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
           do_sample: bool) -> torch.Tensor:
    """Convert logits (..., V) to a probability distribution.

    For ``do_sample=False`` returns a one-hot at the argmax so the sampling and
    greedy code paths share one shape. Truncation (top_k/top_p) is applied
    identically to the draft and target, which preserves the speculative
    sampling guarantee w.r.t. the *truncated* target distribution.
    """
    if not do_sample:
        idx = logits.argmax(dim=-1, keepdim=True)
        oh = torch.zeros_like(logits)
        oh.scatter_(-1, idx, 1.0)
        return oh

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cdf > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(scatter_remove, float("-inf"))
    return F.softmax(logits, dim=-1)


def _sample(probs: torch.Tensor, do_sample: bool,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Draw one token: probs (1, V) -> (1, 1)."""
    if not do_sample:
        return probs.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs[0], num_samples=1, generator=generator).view(1, 1)


def _walk_accept(
    p_logits: List[torch.Tensor],
    q_dists: List[torch.Tensor],
    draft_tokens: List[torch.Tensor],
    bonus_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rand_fn,
    generator: Optional[torch.Generator],
) -> tuple[int, torch.Tensor]:
    """Speculative accept/reject over ``gamma`` proposals.

    ``p_logits[i]`` is the target's logits for the i-th drafted position (i.e.
    the distribution *before* seeing ``draft_tokens[i]``); ``q_dists[i]`` is the
    draft distribution it was sampled from; ``bonus_logits`` is the target's
    distribution after the last proposal (used when all are accepted). Shared by
    the cache-free :func:`generate` and the cached :func:`generate_cached` so the
    acceptance math can never drift between them.

    Returns ``(accepted, next_tok)`` where ``next_tok`` is the correction token
    (first rejection) or the bonus token (all accepted).
    """
    gamma = len(draft_tokens)
    accepted = 0
    next_tok: Optional[torch.Tensor] = None
    for i in range(gamma):
        p = _probs(p_logits[i], temperature, top_k, top_p, do_sample)
        x = draft_tokens[i]
        if do_sample:
            p_x = p[0, x[0, 0]]
            q_x = q_dists[i][0, x[0, 0]]
            ratio = torch.clamp(p_x / (q_x + 1e-12), max=1.0)
            if rand_fn() < ratio:
                accepted += 1
                continue
            resid = torch.clamp(p - q_dists[i], min=0.0)
            s = resid.sum(dim=-1, keepdim=True)
            # Degenerate residual (p≈q on the rejected token): fall back to p.
            resid = torch.where(s > 0, resid / s, p)
            next_tok = torch.multinomial(resid[0], 1, generator=generator).view(1, 1)
            break
        else:
            tgt_tok = p.argmax(dim=-1, keepdim=True)
            if int(tgt_tok) == int(x):
                accepted += 1
                continue
            next_tok = tgt_tok
            break

    if accepted == gamma:
        p_bonus = _probs(bonus_logits, temperature, top_k, top_p, do_sample)
        next_tok = _sample(p_bonus, do_sample, generator)
    return accepted, next_tok


@torch.no_grad()
def generate(
    target,
    draft,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    gamma: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, SpecDecodeStats]:
    """Speculative-decode ``max_new_tokens`` continuation tokens.

    Args:
        target, draft: callables/Modules returning ``.logits`` of shape
            (1, seq, vocab) given ``input_ids`` (1, seq).
        input_ids: prompt token ids, shape (1, T).
        gamma: draft tokens proposed per verification step.
        do_sample: probabilistic speculative sampling if True, else greedy.
        generator: optional ``torch.Generator`` for reproducible acceptance
            coin-flips and resampling.

    Returns:
        (sequence, stats) — ``sequence`` is the prompt with appended tokens.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, T)")
    device = input_ids.device
    cur = input_ids
    stats = SpecDecodeStats()

    def _rand() -> torch.Tensor:
        return torch.rand((), device=device, generator=generator)

    while stats.new_tokens < max_new_tokens:
        T = cur.shape[1]

        # 1) draft proposes gamma tokens autoregressively, recording q(·).
        draft_seq = cur
        q_dists: List[torch.Tensor] = []
        draft_tokens: List[torch.Tensor] = []
        for _ in range(gamma):
            q = _probs(_next_token_logits(draft, draft_seq),
                       temperature, top_k, top_p, do_sample)
            tok = _sample(q, do_sample, generator)
            q_dists.append(q)
            draft_tokens.append(tok)
            draft_seq = torch.cat([draft_seq, tok], dim=1)

        # 2) target verifies all gamma proposals in one forward.
        tgt_logits = _all_logits(target, draft_seq)  # (1, T+gamma, V)

        # 3) walk proposals, accept/reject. p_logits[i] is the target's logits
        #    for the i-th drafted position: index T-1+i predicts the token at
        #    position T+i (= draft_tokens[i]); index T-1+gamma is the bonus.
        p_logits = [tgt_logits[:, T - 1 + i, :] for i in range(gamma)]
        accepted, next_tok = _walk_accept(
            p_logits, q_dists, draft_tokens, tgt_logits[:, T - 1 + gamma, :],
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            rand_fn=_rand, generator=generator,
        )

        # 4) commit accepted draft tokens.
        if accepted > 0:
            cur = torch.cat([cur, *draft_tokens[:accepted]], dim=1)

        # 5) emit the extra token (_walk_accept returned the bonus or correction).
        cur = torch.cat([cur, next_tok], dim=1)

        stats.steps += 1
        stats.proposed += gamma
        stats.accepted += accepted
        stats.accepted_per_step.append(accepted)
        stats.new_tokens += accepted + 1

        if eos_token_id is not None and int(next_tok) == int(eos_token_id):
            break

    return cur, stats


@torch.no_grad()
def generate_cached(
    target,
    draft,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    gamma: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    cache_factory: Optional[Callable] = None,
) -> tuple[torch.Tensor, SpecDecodeStats]:
    """KV-cached speculative decoding — O(T) instead of :func:`generate`'s O(T²).

    Requires HuggingFace-style models that accept ``past_key_values`` /
    ``use_cache`` / ``cache_position`` and return ``.past_key_values`` (e.g. the
    SC-patched models from :func:`make_sc_model`). Each step the draft proposes
    ``gamma`` tokens incrementally and the target verifies them in a single
    cached forward; on rejection the rejected draft tokens are dropped from both
    caches with ``DynamicCache.crop`` and only the accepted prefix is kept — the
    same rollback HF and lucidrains use.

    KV storage is orthogonal to SC: the caches hold the (quantized) K/V while
    only the matmuls run on SC, so caching does not change the result relative
    to :func:`generate` — just the cost. The accept/reject math is shared with
    :func:`generate` via :func:`_walk_accept`, so the two cannot diverge.

    Same arguments and return value as :func:`generate`. ``cache_factory`` is a
    zero-arg callable producing a fresh cache (defaults to ``DynamicCache``);
    override it only for testing without transformers.
    """
    if cache_factory is None:
        try:
            from transformers import DynamicCache
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "generate_cached needs transformers (DynamicCache); pass "
                "cache_factory=, or use generate() for non-HF models") from e
        cache_factory = DynamicCache

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, T)")
    device = input_ids.device
    stats = SpecDecodeStats()

    def _rand() -> torch.Tensor:
        return torch.rand((), device=device, generator=generator)

    def _fwd(model, cache, tokens):
        """Run ``tokens`` through ``model`` extending ``cache``; return (logits, cache)."""
        past = cache.get_seq_length()
        pos = torch.arange(past, past + tokens.shape[1], device=device)
        out = model(input_ids=tokens, past_key_values=cache,
                    use_cache=True, cache_position=pos)
        return out.logits, out.past_key_values

    target_cache = cache_factory()
    draft_cache = cache_factory()

    # Prefill: ingest the whole prompt once into both caches.
    cur = input_ids
    t_logits, target_cache = _fwd(target, target_cache, cur)
    d_logits, draft_cache = _fwd(draft, draft_cache, cur)
    target_last = t_logits[:, -1, :]   # target dist for the next proposal (p(x_0))
    draft_last = d_logits[:, -1, :]    # draft dist for the next proposal (q(x_0))

    while stats.new_tokens < max_new_tokens:
        C = cur.shape[1]   # committed length at the start of this step

        # 1) draft proposes gamma tokens; ingest x_0..x_{gamma-2} into its cache
        #    (x_{gamma-1} is sampled but not ingested — the target may reject it).
        q_dists: List[torch.Tensor] = []
        draft_tokens: List[torch.Tensor] = []
        last = draft_last
        for i in range(gamma):
            q = _probs(last, temperature, top_k, top_p, do_sample)
            x = _sample(q, do_sample, generator)
            q_dists.append(q)
            draft_tokens.append(x)
            if i < gamma - 1:
                d_logits, draft_cache = _fwd(draft, draft_cache, x)
                last = d_logits[:, -1, :]
        # draft_cache now covers C + (gamma-1) tokens.

        # 2) target verifies all gamma proposals in one cached forward.
        block = torch.cat(draft_tokens, dim=1)            # (1, gamma)
        t_logits, target_cache = _fwd(target, target_cache, block)
        # target_cache now covers C + gamma tokens. p(x_0) is last step's
        # target_last; p(x_i>=1) is the logits produced after ingesting x_{i-1};
        # the bonus distribution is the final position.
        p_logits = [target_last] + [t_logits[:, j, :] for j in range(gamma - 1)]
        bonus_logits = t_logits[:, gamma - 1, :]

        # 3) accept/reject (shared with generate()).
        accepted, next_tok = _walk_accept(
            p_logits, q_dists, draft_tokens, bonus_logits,
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            rand_fn=_rand, generator=generator,
        )

        # 4) commit accepted drafts + the extra (correction or bonus) token.
        if accepted > 0:
            cur = torch.cat([cur, *draft_tokens[:accepted]], dim=1)
        cur = torch.cat([cur, next_tok], dim=1)   # length now C + accepted + 1

        # 5) roll caches back to the accepted prefix, then re-ingest the
        #    committed tail so both caches cover all of `cur` and we hold fresh
        #    last-logits. accepted drafts occupy positions C..C+accepted-1.
        keep = C + accepted
        if target_cache.get_seq_length() > keep:
            target_cache.crop(keep)
        if draft_cache.get_seq_length() > keep:
            draft_cache.crop(keep)
        tail_t = cur[:, target_cache.get_seq_length():]
        t_logits, target_cache = _fwd(target, target_cache, tail_t)
        target_last = t_logits[:, -1, :]
        tail_d = cur[:, draft_cache.get_seq_length():]
        d_logits, draft_cache = _fwd(draft, draft_cache, tail_d)
        draft_last = d_logits[:, -1, :]

        stats.steps += 1
        stats.proposed += gamma
        stats.accepted += accepted
        stats.accepted_per_step.append(accepted)
        stats.new_tokens += accepted + 1

        if eos_token_id is not None and int(next_tok) == int(eos_token_id):
            break

    return cur, stats
