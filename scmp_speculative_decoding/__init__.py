"""SC-simulated speculative decoding.

Loads a draft + target model pair whose *every* matmul is simulated in
stochastic computing (via ``scmp_kernels.sc_matmul``), then runs speculative
decoding and reports the acceptance rate — the metric that captures how SC
precision degrades draft/target agreement.

    from scmp_speculative_decoding import load_spec_models, generate
    m = load_spec_models()
    ids = m.tokenizer("Explain stochastic computing.", return_tensors="pt").input_ids
    out, stats = generate(m.target, m.draft, ids.to(m.target.device),
                          max_new_tokens=128, gamma=4)
    print(m.tokenizer.decode(out[0]), stats)
"""
from .loader import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    SpecModels,
    load_spec_models,
)
from .sc_model import SC_CONFIG_DEFAULTS, SCLinear, make_sc_model
from .spec_decode import SpecDecodeStats, generate, generate_cached

__all__ = [
    "load_spec_models",
    "SpecModels",
    "DEFAULT_DRAFT_MODEL",
    "DEFAULT_TARGET_MODEL",
    "make_sc_model",
    "SCLinear",
    "SC_CONFIG_DEFAULTS",
    "generate",
    "generate_cached",
    "SpecDecodeStats",
]
