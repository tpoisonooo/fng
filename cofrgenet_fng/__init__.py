"""Unofficial PyTorch implementation of CoFrGeNet-FNG.

Non-Expanding Gated Continued Fraction feed-forward layers for language
models (arXiv:2611.26835), plus classic FFN baselines and fused Triton
forward/backward kernels for the continued-fraction ladders.
"""

from .layers import CoFrGeNetFNG, SwiGLUFFN, VanillaFFN, continuant_fraction
from .triton_ladder import (
    HAS_TRITON,
    CoFrGeNetFNGTriton,
    ladder_apply,
    ladder_backward,
    ladder_forward,
    triton_backend,
)

__version__ = "0.2.0"

__all__ = [
    "CoFrGeNetFNG",
    "CoFrGeNetFNGTriton",
    "SwiGLUFFN",
    "VanillaFFN",
    "continuant_fraction",
    "HAS_TRITON",
    "ladder_apply",
    "ladder_backward",
    "ladder_forward",
    "triton_backend",
    "__version__",
]
