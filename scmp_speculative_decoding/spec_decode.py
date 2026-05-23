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


def _sample(probs: torch.Tensor, do_sample: bool) -> torch.Tensor:
    """Draw one token: probs (1, V) -> (1, 1)."""
    if not do_sample:
        return probs.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs[0], num_samples=1).view(1, 1)


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
            tok = _sample(q, do_sample)
            q_dists.append(q)
            draft_tokens.append(tok)
            draft_seq = torch.cat([draft_seq, tok], dim=1)

        # 2) target verifies all gamma proposals in one forward.
        tgt_logits = _all_logits(target, draft_seq)  # (1, T+gamma, V)

        # 3) walk proposals, accept/reject.
        accepted = 0
        next_tok: Optional[torch.Tensor] = None
        for i in range(gamma):
            p = _probs(tgt_logits[:, T - 1 + i, :],
                       temperature, top_k, top_p, do_sample)
            x = draft_tokens[i]
            if do_sample:
                p_x = p[0, x[0, 0]]
                q_x = q_dists[i][0, x[0, 0]]
                ratio = torch.clamp(p_x / (q_x + 1e-12), max=1.0)
                if _rand() < ratio:
                    accepted += 1
                    continue
                resid = torch.clamp(p - q_dists[i], min=0.0)
                s = resid.sum(dim=-1, keepdim=True)
                # Degenerate residual (p≈q on the rejected token): fall back to p.
                resid = torch.where(s > 0, resid / s, p)
                next_tok = torch.multinomial(
                    resid[0], 1, generator=generator).view(1, 1)
                break
            else:
                tgt_tok = p.argmax(dim=-1, keepdim=True)  # (1,1)
                if int(tgt_tok) == int(x):
                    accepted += 1
                    continue
                next_tok = tgt_tok
                break

        # 4) commit accepted draft tokens.
        if accepted > 0:
            cur = torch.cat([cur, *draft_tokens[:accepted]], dim=1)

        # 5) emit one more token: bonus (all accepted) or correction (rejection).
        if accepted == gamma:
            p_bonus = _probs(tgt_logits[:, T - 1 + gamma, :],
                             temperature, top_k, top_p, do_sample)
            next_tok = _sample(p_bonus, do_sample)
        cur = torch.cat([cur, next_tok], dim=1)

        stats.steps += 1
        stats.proposed += gamma
        stats.accepted += accepted
        stats.accepted_per_step.append(accepted)
        stats.new_tokens += accepted + 1

        if eos_token_id is not None and int(next_tok) == int(eos_token_id):
            break

    return cur, stats
