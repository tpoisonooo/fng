"""
fng_triton.py

Benchmark of the fused Triton ladder kernel (cofrgenet_fng.triton_ladder)
against the eager PyTorch ladder loop:

  [1] correctness of the kernel vs. the eager continuant reference
  [2] ladder-only forward speed
  [3] whole FNG block forward: all-torch vs. Triton ladder inside

Backend: CUDA if available, else the triton-cpu backend (set
TRITON_DEFAULT_BACKEND=cpu or let it auto-select), else the slow numpy
interpreter (correctness only, timing skipped).

Run from the repo root:  python fng_triton.py
"""

import time

import torch
import torch.nn.functional as F

from cofrgenet_fng import (
    CoFrGeNetFNG,
    HAS_TRITON,
    continuant_fraction,
    ladder_forward,
    triton_backend,
)


def check_correctness():
    print("=" * 72)
    print("[1] Triton ladder vs eager continuant_fraction")
    print("=" * 72)
    torch.manual_seed(0)
    worst = 0.0
    for d in (1, 2, 5, 8):
        c = torch.randn(4096, 8, d + 1) * 2  # (N, L, d+1), fp32
        ref = c[..., 0] + continuant_fraction(c[..., 1:], eps=0.1)
        got = ladder_forward(c, eps=0.1)
        # near a pole |f~| is huge and both fp32 evaluations carry ~1e-4
        # noise, so judge by relative error with a depth-tolerant bound
        err = ((ref - got).abs() / ref.abs().clamp_min(1.0)).max().item()
        worst = max(worst, err)
        print(f"d={d}: max rel diff = {err:.2e}")
    assert worst < 1e-3, "triton kernel disagrees with eager reference"
    print("OK\n")


def _time(fn, iters=50):
    for _ in range(3):  # warmup
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_ladder():
    print("=" * 72)
    print("[2] Ladder-only speed: eager loop vs fused Triton kernel (fwd)")
    print("=" * 72)
    if triton_backend() == "interpreter":
        print("interpreter mode (TRITON_INTERPRET=1): correctness only, skip timing\n")
        return
    for B, S, L, d in ((16, 256, 8, 8), (64, 1024, 16, 8)):
        c = torch.randn(B, S, L, d + 1)
        n = B * S * L
        ms_eager = _time(lambda: c[..., 0] + continuant_fraction(c[..., 1:], eps=0.1))
        ms_fused = _time(lambda: ladder_forward(c, eps=0.1))
        print(
            f"(B={B}, S={S}, L={L}, d={d})  N={n:,} ladders: "
            f"eager {ms_eager:.3f} ms | triton {ms_fused:.3f} ms "
            f"({ms_eager / ms_fused:.2f}x)"
        )
    print()


def bench_module():
    print("=" * 72)
    print("[3] Whole FNG block fwd: all-torch vs Triton ladder inside")
    print("=" * 72)
    if triton_backend() == "interpreter":
        print("interpreter mode: skip timing\n")
        return
    p, L, d = 768, 8, 8
    m = CoFrGeNetFNG(p, L, d)
    x = torch.randn(16, 256, p)

    def triton_forward(x):
        a, b = m.gate_up(x).chunk(2, dim=-1)
        z = a * F.silu(b)
        c = m.coeffs(z).unflatten(-1, (m.L, m.d + 1))
        return m.down(ladder_forward(c, m.eps))

    ms_torch = _time(lambda: m(x))
    ms_triton = _time(lambda: triton_forward(x))
    print(f"{'all-torch':<14} {ms_torch:8.3f} ms")
    print(f"{'triton-ladder':<14} {ms_triton:8.3f} ms")
    print()


def main():
    import triton

    print(
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"backend: {triton_backend()}\n"
    )
    check_correctness()
    bench_ladder()
    bench_module()


if __name__ == "__main__":
    if not HAS_TRITON:
        raise SystemExit(
            "triton is not installed; see README for the triton-cpu wheel."
        )
    main()
