"""Fused Triton forward + backward kernels for CoFrGeNet-FNG ladders.

The ladder part of the FNG block is a poor fit for eager PyTorch: a serial
loop of d tiny elementwise kernels plus a chain of sign/abs/log/exp ops.
Each (batch, seq) position is independent, so both directions fuse into
single pointwise kernels.

Forward: out = a0 + frac, where frac = K_{d-1} / K_d evaluated by the
continuant recursion (1 division, paper eqs. 3-6) with the log/exp division
and clamped denominator (paper eq. 8):

    frac = num / den_c,   num = K_{d-1}, den = K_d,
    den_c = sgn(den) * max(|den|, eps)

Backward (analytic adjoint, also division-free):

    dnum = g * sgn(den) * exp(-log max(|den|, eps))
    dden = -g * frac * sgn(den) * exp(-log max(|den|, eps))   if |den| >= eps
           0                                                  otherwise

    then backprop through K_k = a_{d-k+1} K_{k-1} + K_{k-2} in reverse:
    for k = d .. 2:
        d a_{d-k+1} = A1 * km1_old
        (A2, A1)    = (A1, A1 * a + A2)
    d a_d = A1,  d a_0 = g

Since Triton jit has no register arrays, km1_old is recomputed per step
(O(d^2) scalar ops in registers; d <= 8, negligible next to exp/log).

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
import torch.nn.functional as F

from .layers import CoFrGeNetFNG

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on environment
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = [
    "HAS_TRITON",
    "triton_backend",
    "ladder_forward",
    "ladder_backward",
    "ladder_apply",
    "CoFrGeNetFNGTriton",
]


def triton_backend() -> str:
    """'cuda', 'cpu', 'interpreter', or 'unavailable'."""
    if not HAS_TRITON:
        return "unavailable"
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "interpreter"
    return "cpu"


def _no_triton(*args, **kwargs):
    raise ImportError(
        "triton is not installed. Install it (GPU: pip install triton; "
        "CPU: wheel from github.com/triton-lang/triton-cpu/releases), "
        "or use cofrgenet_fng.CoFrGeNetFNG instead of CoFrGeNetFNGTriton."
    )


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
    def _frac_logexp(num, den, eps):
        # frac = num / (sgn(den) * max(|den|, eps)), via log/exp (paper eq. 8)
        return (
            _sign(num)
            * _sign(den)
            * tl.exp(
                tl.log(tl.maximum(tl.abs(num), 1e-12))
                - tl.log(tl.maximum(tl.abs(den), eps))
            )
        )

    @triton.jit
    def _ladder_kernel(
        c_ptr,  # (N, D1) {fp32, bf16, fp16}: per ladder [a0, a_1, ..., a_d]
        out_ptr,  # (N,) same dtype as c_ptr
        eps,
        N,
        D1: tl.constexpr,  # d + 1
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < N
        base = c_ptr + rows * D1

        # mixed precision: load in native dtype, compute in fp32 (stable near
        # poles), cast back on store
        a0 = tl.load(base + 0, mask=mask, other=0.0).to(tl.float32)

        # continuant recursion K_k = a_{d-k+1} * K_{k-1} + K_{k-2}, from a_d up
        Km2 = tl.full((BLOCK,), 1.0, tl.float32)  # K_0
        Km1 = tl.load(base + (D1 - 1), mask=mask, other=0.0).to(tl.float32)  # K_1 = a_d
        for k in tl.static_range(2, D1):
            a = tl.load(base + (D1 - k), mask=mask, other=0.0).to(tl.float32)
            Km2, Km1 = Km1, a * Km1 + Km2
        # Km1 = K_d(a_1..a_d), Km2 = K_{d-1}(a_2..a_d)

        out = a0 + _frac_logexp(Km2, Km1, eps)
        tl.store(out_ptr + rows, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _ladder_bwd_kernel(
        c_ptr,  # (N, D1) {fp32, bf16, fp16}: forward coefficients
        g_ptr,  # (N,) same dtype: upstream grad w.r.t. ladder output
        dc_ptr,  # (N, D1) same dtype: output grads w.r.t. c
        eps,
        N,
        D1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < N
        base = c_ptr + rows * D1
        g = tl.load(g_ptr + rows, mask=mask, other=0.0).to(tl.float32)

        # 1) recompute forward to get num, den, frac (d <= 8: cheap recompute,
        #    saves storing O(d) activations); all math in fp32
        Km2 = tl.full((BLOCK,), 1.0, tl.float32)
        Km1 = tl.load(base + (D1 - 1), mask=mask, other=0.0).to(tl.float32)  # K_1 = a_d
        for k in tl.static_range(2, D1):
            a = tl.load(base + (D1 - k), mask=mask, other=0.0).to(tl.float32)
            Km2, Km1 = Km1, a * Km1 + Km2
        num, den = Km2, Km1
        frac = _frac_logexp(num, den, eps)  # frac = num / den_c

        # 2) adjoint of frac = num / den_c, without explicit divisions
        log_cap = tl.log(tl.maximum(tl.abs(den), eps))
        sgn_den = _sign(den)
        d_num = g * sgn_den * tl.exp(-log_cap)  # g / den_c
        d_den = tl.where(
            tl.abs(den) >= eps,
            -g * frac * sgn_den * tl.exp(-log_cap),  # -g * frac / den_c
            0.0,
        )

        tl.store(
            dc_ptr + rows * D1 + 0,
            g.to(dc_ptr.dtype.element_ty),
            mask=mask,
        )  # d a_0

        if D1 == 2:
            # d = 1: num = K_0 = 1 is constant, den = a_1
            tl.store(
                dc_ptr + rows * D1 + 1,
                d_den.to(dc_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            # 3) reverse adjoint of Km2, Km1 = Km1, a * Km1 + Km2
            A2, A1 = d_num, d_den
            for i in range(0, D1 - 2):
                rev_k = (D1 - 1) - i  # reverse step: d, d-1, ..., 2
                a = tl.load(base + (D1 - rev_k), mask=mask, other=0.0).to(tl.float32)

                # recompute state (v2, v1) before step rev_k (steps 2 .. rev_k-1)
                v2 = tl.full((BLOCK,), 1.0, tl.float32)
                v1 = tl.load(base + (D1 - 1), mask=mask, other=0.0).to(tl.float32)
                for jj in range(0, rev_k - 2):
                    aj = tl.load(base + (D1 - jj - 2), mask=mask, other=0.0).to(tl.float32)
                    v2, v1 = v1, aj * v1 + v2

                d_a = A1 * v1  # grad w.r.t. the coefficient consumed at step rev_k
                tl.store(
                    dc_ptr + rows * D1 + (D1 - rev_k),
                    d_a.to(dc_ptr.dtype.element_ty),
                    mask=mask,
                )
                A2, A1 = A1, A1 * a + A2
            tl.store(
                dc_ptr + rows * D1 + (D1 - 1),
                A1.to(dc_ptr.dtype.element_ty),
                mask=mask,
            )  # d a_d

    def _prep(c: torch.Tensor) -> torch.Tensor:
        if c.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError(
                f"ladder kernels support float32/float16/bfloat16, got {c.dtype}"
            )
        return c.reshape(-1, c.shape[-1]).contiguous()

    def ladder_forward(c: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """c: (..., L, d+1) coefficients [a0, a_1..a_d] -> (..., L) outputs."""
        c2 = _prep(c)
        out = torch.empty(c2.shape[0], device=c.device, dtype=c.dtype)
        block = 256
        _ladder_kernel[(triton.cdiv(c2.shape[0], block),)](
            c2, out, eps, c2.shape[0], D1=c.shape[-1], BLOCK=block
        )
        return out.reshape(c.shape[:-1])

    def ladder_backward(
        c: torch.Tensor, grad_out: torch.Tensor, eps: float = 0.1
    ) -> torch.Tensor:
        """Gradient w.r.t. c for ladder_forward; grad_out has shape (..., L)."""
        c2 = _prep(c)
        g2 = grad_out.to(c.dtype).reshape(-1).contiguous()
        dc = torch.empty_like(c2)
        block = 256
        _ladder_bwd_kernel[(triton.cdiv(c2.shape[0], block),)](
            c2, g2, dc, eps, c2.shape[0], D1=c.shape[-1], BLOCK=block
        )
        return dc.reshape(c.shape)

    class _LadderTritonFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, c, eps: float):
            ctx.save_for_backward(c)
            ctx.eps = float(eps)
            return ladder_forward(c, eps)

        @staticmethod
        def backward(ctx, grad_out):
            (c,) = ctx.saved_tensors
            return ladder_backward(c, grad_out, ctx.eps), None

    def ladder_apply(c: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """Differentiable continued-fraction ladder (fused fwd + bwd)."""
        return _LadderTritonFn.apply(c, eps)

    class CoFrGeNetFNGTriton(CoFrGeNetFNG):
        """CoFrGeNetFNG whose ladder runs the fused Triton fwd+bwd kernels.

        Same parameters / state_dict as CoFrGeNetFNG; input/output are fp32
        (the surrounding gate_up, coeffs and down layers stay in autograd)."""

        def forward(self, x):
            a, b = self.gate_up(x).chunk(2, dim=-1)
            z = a * F.silu(b)
            c = self.coeffs(z).unflatten(-1, (self.L, self.d + 1))
            return self.down(ladder_apply(c, self.eps))

else:
    ladder_forward = _no_triton
    ladder_backward = _no_triton
    ladder_apply = _no_triton

    class CoFrGeNetFNGTriton(CoFrGeNetFNG):
        def forward(self, x):
            _no_triton()
