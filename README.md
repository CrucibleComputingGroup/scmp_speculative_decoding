# scmp_speculative_decoding

Speculative decoding where **every matmul in both models is simulated in
stochastic computing (SC)** via [`scmp_kernels`](https://github.com/CrucibleComputingGroup/scmp_kernels)`.sc_matmul`.

A small **draft** model proposes `gamma` tokens; the large **target** model
verifies them in one forward pass. Each model is SC-enabled by replacing every
`nn.Linear` in its decoder layers with an `SCLinear` and monkey-patching the
attention's `eager_attention_forward`, so the QKV/O projections, the MLP, and
both attention score matmuls (Q·Kᵀ and softmax·V) all run through the SC
kernel. The same wiring is used in
[`scmp_llm_llama`](https://github.com/CrucibleComputingGroup/scmp_llm) — this
repo reuses that pattern and adds the speculative-decoding loop on top.

The headline metric is the **acceptance rate**: SC quantization noise makes the
draft and target disagree more often, so as `stoc_len` / `sc_prec` drop, fewer
proposals are accepted and the speedup shrinks — while (by construction) the
target distribution being sampled is unchanged.

## Model pair

Defaults to Hugging Face's recommended assisted-generation pair (same family /
shared tokenizer):

| role   | model |
|--------|-------|
| draft  | `meta-llama/Llama-3.2-1B-Instruct` |
| target | `meta-llama/Llama-3.1-8B-Instruct` |

Override via `DRAFT_MODEL` / `TARGET_MODEL` env vars or `load_spec_models(...)`.

## Install

```bash
pip install -r requirements.txt
pip install -e ../scmp_kernels   # supplies sc_matmul (Triton kernels, needs GPU)
pip install -e .
```

`torch` + the Triton SC kernels require a CUDA GPU; on this cluster run under a
GPU Slurm allocation. The accept/reject logic itself is CPU-testable without
SC (see below).

## Use

```python
import torch
from scmp_speculative_decoding import load_spec_models, generate

m = load_spec_models()                       # both models SC-enabled
ids = m.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain stochastic computing."}],
    add_generation_prompt=True, return_tensors="pt").to(m.target.device)

out, stats = generate(m.target, m.draft, ids, max_new_tokens=128, gamma=4)
print(m.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
print(stats)        # acceptance rate, mean accepted/step, ...
```

SC knobs live on each `model.config` and can be set independently
(`use_sc_attn`, `use_sc_linear`, `sc_prec`, `sc_stoc_len`, `sc_mode`,
`sc_granularity`, `sc_halve_bipolar`); see `SC_CONFIG_DEFAULTS`. For example,
to isolate where SC costs acceptance, run the draft in fp16 and only the target
in SC:

```python
m = load_spec_models(draft_sc_overrides={"use_sc_attn": False,
                                         "use_sc_linear": False})
```

## Sweep script

`run_spec_decode.py` loads the pair once and sweeps `STOC_LENS`, printing the
acceptance rate and decoded text at each precision (fp16 baseline first):

```bash
STOC_LENS=256,128,64,32 NEW_TOKENS=128 GAMMA=4 python run_spec_decode.py
```

Env vars: `TARGET_MODEL`, `DRAFT_MODEL`, `PROMPT`, `NEW_TOKENS`, `GAMMA`,
`DO_SAMPLE`, `TEMPERATURE`, `SC_PREC`, `STOC_LENS`, `SC_ATTN_GRANULARITY`,
`SEED`.

## Algorithm & correctness

Standard speculative sampling (Leviathan et al. 2023; Chen et al. 2023): accept
draft token `x ~ q` with probability `min(1, p(x)/q(x))`; on first rejection
resample from the normalized residual `(p − q)₊`; if all `gamma` are accepted,
emit a free bonus token from the target. This is provably equivalent to
sampling from the target alone, so SC noise affects only speed, not the target
distribution. The loop is intentionally **KV-cache-free** (it re-runs the full
forward each step) — the SC matmul simulation dominates runtime anyway, and the
cache-free path keeps the accept/reject logic obviously correct.

## Tests

CPU-only, no SC kernels or HF download required (tiny stub models):

```bash
python tests/test_spec_decode.py     # or: pytest tests/
```

These pin the two guarantees: greedy speculative decoding is **bit-identical**
to plain target greedy decoding, and sampling reproduces the target's
next-token distribution — for any draft model.
