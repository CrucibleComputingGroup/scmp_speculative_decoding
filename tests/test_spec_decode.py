"""Correctness tests for the speculative-decoding loop.

Pure CPU, no SC kernels and no HuggingFace: tiny stub "models" stand in for the
draft/target so we can check the accept/reject math directly.

Key guarantees:
  * greedy speculative decoding == plain target greedy decoding (bit-exact),
    for *any* draft — the draft only affects speed, never the output;
  * sampling speculative decoding reproduces the target's next-token
    distribution (statistically), again for any draft.

Run with ``pytest tests/`` or directly: ``python tests/test_spec_decode.py``.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scmp_speculative_decoding.spec_decode import generate, generate_cached  # noqa: E402


class StubModel:
    """Causal LM stub: logits at position t depend only on token t.

    ``logits[:, t, :] = W[input_ids[t]]`` — a valid causal map (each position
    sees only tokens up to itself), enough to define next-token distributions.
    """

    def __init__(self, W: torch.Tensor):
        self.W = W  # (vocab, vocab)

    def __call__(self, input_ids: torch.Tensor):
        logits = self.W[input_ids[0]].unsqueeze(0)  # (1, seq, vocab)
        return type("O", (), {"logits": logits})()


class IntCache:
    """Minimal stand-in for transformers' DynamicCache (length + crop only).

    The stub model's logits depend only on the current token, so the cache
    contents don't affect outputs — but tracking length and honoring ``crop``
    exercises generate_cached's rollback / re-ingest bookkeeping exactly as
    DynamicCache would.
    """

    def __init__(self):
        self.n = 0

    def get_seq_length(self) -> int:
        return self.n

    def crop(self, max_length: int) -> None:
        self.n = min(self.n, max_length)

    def advance(self, k: int) -> None:
        self.n += k


class CachedStubModel(StubModel):
    """StubModel speaking the cached forward protocol generate_cached expects."""

    def __call__(self, input_ids, past_key_values=None, use_cache=False,
                 cache_position=None):
        logits = self.W[input_ids[0]].unsqueeze(0)
        if past_key_values is not None:
            past_key_values.advance(input_ids.shape[1])
        return type("O", (), {"logits": logits,
                              "past_key_values": past_key_values})()


def _greedy_reference(model, ids: torch.Tensor, n: int) -> torch.Tensor:
    cur = ids
    for _ in range(n):
        nxt = model(cur).logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    return cur


def test_greedy_matches_plain_target():
    torch.manual_seed(0)
    V = 17
    target = StubModel(torch.randn(V, V))
    draft = StubModel(torch.randn(V, V))  # deliberately a different model
    prompt = torch.tensor([[3, 1, 4]])

    for gamma in (1, 2, 4, 8):
        out, stats = generate(target, draft, prompt,
                              max_new_tokens=20, gamma=gamma, do_sample=False)
        # Plain target greedy decode to the same total length.
        ref = _greedy_reference(target, prompt, out.shape[1] - prompt.shape[1])
        assert torch.equal(out, ref), f"gamma={gamma}: {out} != {ref}"
        assert stats.proposed == stats.steps * gamma
        assert 0 <= stats.accepted <= stats.proposed
    # Identical draft==target should accept everything in greedy mode.
    out, stats = generate(target, target, prompt,
                          max_new_tokens=20, gamma=4, do_sample=False)
    assert stats.acceptance_rate == 1.0


def test_sampling_matches_target_distribution():
    torch.manual_seed(0)
    V = 8
    Wt = torch.randn(V, V)
    target = StubModel(Wt)
    draft = StubModel(torch.randn(V, V))
    prompt = torch.tensor([[2]])
    # Reference: target next-token distribution from the last prompt token.
    ref_p = torch.softmax(Wt[2], dim=-1)

    counts = torch.zeros(V)
    N = 4000
    for i in range(N):
        gen = torch.Generator().manual_seed(1000 + i)
        out, _ = generate(target, draft, prompt, max_new_tokens=1,
                          gamma=4, do_sample=True, temperature=1.0, generator=gen)
        counts[int(out[0, prompt.shape[1]])] += 1
    emp = counts / counts.sum()
    # Total-variation distance should be small for N=4000 draws.
    tv = 0.5 * (emp - ref_p).abs().sum().item()
    assert tv < 0.05, f"TV distance {tv:.3f} too large; emp={emp}, ref={ref_p}"


def test_cached_matches_cachefree_greedy():
    """The KV-cached path must produce byte-identical output to the cache-free
    reference (greedy) — this pins all the crop / re-ingest index bookkeeping."""
    torch.manual_seed(0)
    V = 17
    Wt = torch.randn(V, V)
    Wd = torch.randn(V, V)
    prompt = torch.tensor([[3, 1, 4]])
    for gamma in (1, 2, 4, 8):
        out_free, sf = generate(StubModel(Wt), StubModel(Wd), prompt,
                                max_new_tokens=25, gamma=gamma, do_sample=False)
        out_cached, sc = generate_cached(CachedStubModel(Wt), CachedStubModel(Wd),
                                         prompt, max_new_tokens=25, gamma=gamma,
                                         do_sample=False, cache_factory=IntCache)
        assert torch.equal(out_free, out_cached), \
            f"gamma={gamma}: cached {out_cached} != free {out_free}"
        assert sf.accepted == sc.accepted and sf.new_tokens == sc.new_tokens
    # draft==target accepts every proposal in greedy mode.
    _, st = generate_cached(CachedStubModel(Wt), CachedStubModel(Wt), prompt,
                            max_new_tokens=25, gamma=4, do_sample=False,
                            cache_factory=IntCache)
    assert st.acceptance_rate == 1.0


def test_stats_accounting():
    torch.manual_seed(1)
    V = 10
    target = StubModel(torch.randn(V, V))
    draft = StubModel(torch.randn(V, V))
    prompt = torch.tensor([[0, 5]])
    out, stats = generate(target, draft, prompt,
                          max_new_tokens=15, gamma=3, do_sample=False)
    # Each step emits accepted + 1 tokens; new_tokens must equal what was added.
    assert stats.new_tokens == out.shape[1] - prompt.shape[1]
    assert stats.new_tokens >= 15
    assert len(stats.accepted_per_step) == stats.steps


if __name__ == "__main__":
    test_greedy_matches_plain_target()
    test_sampling_matches_target_distribution()
    test_cached_matches_cachefree_greedy()
    test_stats_accounting()
    print("all spec-decode tests passed")
