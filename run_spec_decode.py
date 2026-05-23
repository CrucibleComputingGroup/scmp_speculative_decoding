"""Run SC-simulated speculative decoding and report acceptance rate.

Loads the draft+target pair once, then sweeps ``STOC_LENS`` to show how SC
stream length affects draft/target agreement (acceptance rate) and output.

Env vars (mirrors scmp_llm_llama/check_gen.py where they overlap):
    TARGET_MODEL   — HF id (default: meta-llama/Llama-3.1-8B-Instruct)
    DRAFT_MODEL    — HF id (default: meta-llama/Llama-3.2-1B-Instruct)
    PROMPT         — prompt string
    NEW_TOKENS     — number of tokens to generate (default 128)
    GAMMA          — draft tokens per verification step (default 4)
    DO_SAMPLE      — "1" for sampling, "0" for greedy (default 0)
    TEMPERATURE    — sampling temperature (default 1.0)
    SC_PREC        — SC precision (default 8)
    STOC_LENS      — comma-separated sweep (default 256,128,64,32)
    SC_ATTN_GRANULARITY — per_head | per_row (default per_head)
    SEED           — RNG seed for reproducibility (default 0)
"""
import os
import time

import torch

from scmp_speculative_decoding import generate, load_spec_models

TARGET_MODEL = os.environ.get("TARGET_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
PROMPT = os.environ.get("PROMPT", "Explain stochastic computing in two sentences.")
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "128"))
GAMMA = int(os.environ.get("GAMMA", "4"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "0") == "1"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
SC_PREC = int(os.environ.get("SC_PREC", "8"))
STOC_LENS = [int(x) for x in os.environ.get("STOC_LENS", "256,128,64,32").split(",")]
SC_ATTN_GRANULARITY = os.environ.get("SC_ATTN_GRANULARITY", "per_head")
SEED = int(os.environ.get("SEED", "0"))


def _set_sc(model, *, enabled, stoc_len):
    model.config.use_sc_attn = enabled
    model.config.use_sc_linear = enabled
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = stoc_len
    model.config.sc_granularity = SC_ATTN_GRANULARITY


def main() -> None:
    m = load_spec_models(TARGET_MODEL, DRAFT_MODEL, dtype=torch.float16)
    msgs = [{"role": "user", "content": PROMPT}]
    try:
        ids = m.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = m.tokenizer(PROMPT, return_tensors="pt").input_ids
    ids = ids.to(m.target.device)
    eos = m.tokenizer.eos_token_id

    def _run(label: str) -> None:
        gen = torch.Generator(device=ids.device).manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out, stats = generate(
            m.target, m.draft, ids,
            max_new_tokens=NEW_TOKENS, gamma=GAMMA,
            do_sample=DO_SAMPLE, temperature=TEMPERATURE,
            eos_token_id=eos, generator=gen,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        text = m.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"=== [{label}] {dt:.1f}s ===")
        print(f"    {stats}")
        print(text)
        print()

    # fp16 baseline (both models SC off) — the acceptance-rate ceiling.
    _set_sc(m.target, enabled=False, stoc_len=256)
    _set_sc(m.draft, enabled=False, stoc_len=256)
    _run(f"FP16 baseline (target={TARGET_MODEL}, draft={DRAFT_MODEL})")

    # SC sweep — both models at each stoc_len.
    for sl in STOC_LENS:
        _set_sc(m.target, enabled=True, stoc_len=sl)
        _set_sc(m.draft, enabled=True, stoc_len=sl)
        _run(f"SC sc_prec={SC_PREC} stoc_len={sl} gran={SC_ATTN_GRANULARITY}")


if __name__ == "__main__":
    main()
