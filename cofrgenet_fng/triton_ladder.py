"""Fused Triton kernel for the CoFrGeNet-FNG continued-fraction ladders.

The ladder part of the FNG block is a poor fit for eager PyTorch: a serial
loop of d tiny elementwise kernels plus a chain of sign/abs/log/exp ops.
Each (batch, seq) position is independent, so the whole thing fuses into a
single pointwise kernel: load d+1 coefficients, run the continuant
recursion in registers, one fused log/exp "division" (paper eq. 8), store
L outputs.

Backward is not implemented (forward-only kernel); the eager path in
cofrgenet_fng.layers stays the training reference.

Backend selection (at import time):
  * CUDA available     -> kernels launch on GPU
  * triton-cpu backend -> set TRITON_DEFAULT_BACKEND=cpu, or it is picked
                          automatically when no GPU backend exists.
                          Prebuilt wheels: github.com/triton-lang/triton-cpu/releases
  * neither            -> falls back to TRITON_INTERPRET=1 (slow, for
                          correctness checks only)
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on environment
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "ladder_forward", "triton_backend"]


def triton_backend() -> str:
    """'cuda', 'cpu', 'interpreter', or 'unavailable'."""
    if not HAS_TRITON:
        return "unavailable"
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "interpreter"
    return "cpu"


if HAS_TRITON:
    _INTERPRET = False
    if not torch.cuda.is_available():
        # Prefer the triton-cpu backend; fall back to the numpy interpreter.
        try:
            triton.runtime.driver.set_active_to_cpu()
        except (AttributeError, RuntimeError):
            os.environ["TRITON_INTERPRET"] = "1"
            _INTERPRET = True

    @triton.jit
    def _sign(x):
        # torch.sign semantics: 0 -> 0 (tl.where keeps it branch-free)
        return tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))

    @triton.jit
    def _ladder_kernel(
        c_ptr,  # (N, D1) fp32: per ladder [a0, a_1, ..., a_d]
        out_ptr,  # (N,) fp32
        eps,
        N,
        D1: tl.constexpr,  # d + 1
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < N
        base = c_ptr + rows * D1

        a0 = tl.load(base + 0, mask=mask, other=0.0)

        # continuant recursion K_k = a_{d-k+1} * K_{k-1} + K_{k-2}, from a_d up
        Km2 = tl.full((BLOCK,), 1.0, tl.float32)  # K_0
        Km1 = tl.load(base + (D1 - 1), mask=mask, other=0.0)  # K_1 = a_d
        for k in tl.static_range(2, D1):
            a = tl.load(base + (D1 - k), mask=mask, other=0.0)  # a_{d-k+1}
            Km2, Km1 = Km1, a * Km1 + Km2
        # Km1 = K_d(a_1..a_d), Km2 = K_{d-1}(a_2..a_d)

        # paper eq. 8: division rewritten as log/exp, denominator clamped
        num, den = Km2, Km1
        frac = (
            _sign(num)
            * _sign(den)
            * tl.exp(
                tl.log(tl.maximum(tl.abs(num), 1e-12))
                - tl.log(tl.maximum(tl.abs(den), eps))
            )
        )
        tl.store(out_ptr + rows, a0 + frac, mask=mask)

    def ladder_forward(c: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """c: (..., L, d+1) coefficients [a0, a_1..a_d] -> (..., L) outputs."""
        shape = c.shape
        c2 = c.reshape(-1, shape[-1]).contiguous().float()
        n = c2.shape[0]
        out = torch.empty(n, device=c2.device, dtype=torch.float32)
        block = 256
        _ladder_kernel[(triton.cdiv(n, block),)](
            c2, out, eps, n, D1=shape[-1], BLOCK=block
        )
        return out.reshape(shape[:-1])

else:

    def ladder_forward(c: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        raise ImportError(
            "triton is not installed. Install it (or the CPU backend wheel "
            "from github.com/triton-lang/triton-cpu/releases) to use "
            "ladder_forward; the eager equivalent is "
            "cofrgenet_fng.layers.continuant_fraction."
        )
