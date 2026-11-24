"""Unofficial PyTorch implementation of CoFrGeNet-FNG.

Non-Expanding Gated Continued Fraction feed-forward layers for language
models (arXiv:2611.26835), plus classic FFN baselines and a fused Triton
kernel for the continued-fraction ladders.
"""

from .layers import CoFrGeNetFNG, SwiGLUFFN, VanillaFFN, continuant_fraction
from .triton_ladder import HAS_TRITON, ladder_forward, triton_backend

__version__ = "0.1.0"

__all__ = [
    "CoFrGeNetFNG",
    "SwiGLUFFN",
    "VanillaFFN",
    "continuant_fraction",
    "HAS_TRITON",
    "ladder_forward",
    "triton_backend",
    "__version__",
]
