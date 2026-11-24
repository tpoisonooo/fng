"""
fng_vs_ffn.py

CoFrGeNet-FNG ("Non-Expanding Gated Continued Fraction" FFN replacement,
arXiv:2611.26835, IBM) vs. classic FFN baselines, in plain PyTorch.

  [1] correctness of the continuant / log-exp evaluation vs. the naive
      continued-fraction recursion
  [2] parameter count + fwd/bwd speed at transformer scale (p=768)
  [3] teacher-student fitting demo: can each module learn a frozen random
      teacher MLP R^p -> R^p?

Run from the repo root:  python fng_vs_ffn.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from cofrgenet_fng import CoFrGeNetFNG, SwiGLUFFN, VanillaFFN, continuant_fraction

# --------------------------------------------------------------------------
# Demo 1: correctness of the continuant / log-exp evaluation
# --------------------------------------------------------------------------


def naive_fraction(A: torch.Tensor) -> torch.Tensor:
    """Direct recursive evaluation of f~ (d divisions, for checking only)."""
    f = A[..., -1]
    for k in range(A.shape[-1] - 1, 0, -1):
        f = A[..., k - 1] + 1.0 / f
    return 1.0 / f


def check_ladder_math():
    print("=" * 72)
    print("[1] Sanity check: continuant + log/exp == naive continued fraction")
    print("=" * 72)
    torch.manual_seed(0)
    # float64: verifies the math exactly (fp32 near a pole is ill-conditioned)
    A = torch.randn(10000, 5, dtype=torch.float64) * 2
    ref = naive_fraction(A)
    fast = continuant_fraction(A, eps=1e-9)
    # near a pole the two parameterizations legitimately differ; skip those
    mask = ref.abs() < 50
    err = (ref[mask] - fast[mask]).abs().max().item()
    print(f"max |continuant_logexp - naive| over {mask.sum().item()} pts: {err:.2e}")
    assert err < 1e-8, "continuant evaluation mismatch"
    print("OK\n")


# --------------------------------------------------------------------------
# Demo 2: parameter count + fwd/bwd speed at transformer scale
# --------------------------------------------------------------------------


def count_params(m: nn.Module) -> int:
    return sum(t.numel() for t in m.parameters())


def bench(module: nn.Module, x: torch.Tensor, iters: int = 30) -> float:
    """Avg fwd+bwd time per call in ms."""
    device = x.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    # warmup
    for _ in range(3):
        module(x).pow(2).mean().backward()
        module.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        module(x).pow(2).mean().backward()
        module.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def compare_at_scale(device: torch.device):
    print("=" * 72)
    print("[2] Transformer scale: p=768, input (16, 256, 768)")
    print("=" * 72)
    p = 768
    x = torch.randn(16, 256, p, device=device)
    modules = {
        "VanillaFFN (alpha=4)": VanillaFFN(p, 4.0),
        "SwiGLUFFN (alpha=8/3)": SwiGLUFFN(p, 8 / 3),
        "CoFrGeNetFNG (L=5, d=5)": CoFrGeNetFNG(p, 5, 5),
        "CoFrGeNetFNG (L=8, d=8)": CoFrGeNetFNG(p, 8, 8),
    }
    base = None
    print(f"{'module':<28} {'params':>12} {'vs FFN':>8} {'fwd+bwd':>10}")
    for name, m in modules.items():
        m = m.to(device)
        n = count_params(m)
        base = base or n
        ms = bench(m, x)
        print(f"{name:<28} {n:>12,} {n / base:>7.2f}x {ms:>8.2f} ms")
    print()


# --------------------------------------------------------------------------
# Demo 3: can it learn? teacher-student fitting
# --------------------------------------------------------------------------


def fit_demo(device: torch.device, p: int = 64, steps: int = 600):
    print("=" * 72)
    print(f"[3] Fitting demo: learn a frozen random teacher MLP R^{p} -> R^{p}")
    print("=" * 72)
    torch.manual_seed(42)
    teacher = nn.Sequential(nn.Linear(p, 4 * p), nn.Tanh(), nn.Linear(4 * p, p))
    for q in teacher.parameters():
        q.requires_grad_(False)
    teacher = teacher.to(device)

    X = torch.randn(4096, p, device=device)
    with torch.no_grad():
        Y = teacher(X)

    models = {
        "VanillaFFN (alpha=4)": VanillaFFN(p, 4.0),
        "SwiGLUFFN (alpha=8/3)": SwiGLUFFN(p, 8 / 3),
        "CoFrGeNetFNG (L=8, d=4)": CoFrGeNetFNG(p, 8, 4),
        "CoFrGeNetFNG (L=16, d=8)": CoFrGeNetFNG(p, 16, 8),
    }
    for name, m in models.items():
        m = m.to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        n = count_params(m)
        losses = []
        for step in range(steps):
            idx = torch.randint(0, X.shape[0], (512,), device=device)
            loss = F.mse_loss(m(X[idx]), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0:
                losses.append(loss.item())
        curve = " ".join(f"{v:.4f}" for v in losses)
        print(f"{name:<28} params={n:>8,}  mse@100..600: {curve}")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device: {device}\n")
    check_ladder_math()
    compare_at_scale(device)
    fit_demo(device)


if __name__ == "__main__":
    main()
