import pytest
import torch

from cofrgenet_fng import (
    HAS_TRITON,
    CoFrGeNetFNG,
    CoFrGeNetFNGTriton,
    SwiGLUFFN,
    VanillaFFN,
    continuant_fraction,
    ladder_apply,
    ladder_forward,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def naive_fraction(A: torch.Tensor) -> torch.Tensor:
    """Direct recursive evaluation of f~ (d divisions, for checking only)."""
    f = A[..., -1]
    for k in range(A.shape[-1] - 1, 0, -1):
        f = A[..., k - 1] + 1.0 / f
    return 1.0 / f


@pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
def test_continuant_matches_naive(d):
    torch.manual_seed(0)
    # float64: verifies the math exactly (fp32 near a pole is ill-conditioned)
    A = torch.randn(10000, d, dtype=torch.float64) * 2
    ref = naive_fraction(A)
    fast = continuant_fraction(A, eps=1e-9)
    # near a pole the two parameterizations legitimately differ; skip those
    mask = ref.abs() < 50
    assert mask.sum() > 9000
    err = (ref[mask] - fast[mask]).abs().max().item()
    assert err < 1e-8


def test_depth1_is_clamped_reciprocal():
    a1 = torch.tensor([0.05, -0.05, 2.0, -2.0]).unsqueeze(-1)  # (4, d=1)
    out = continuant_fraction(a1, eps=0.1)  # (..., d) -> (...,)
    expected = torch.tensor([10.0, -10.0, 0.5, -0.5])
    assert out.shape == (4,)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("module", [VanillaFFN(32), SwiGLUFFN(32), CoFrGeNetFNG(32, 4, 5)])
def test_module_api(module):
    x = torch.randn(2, 3, 32)
    y = module(x)
    assert y.shape == x.shape


def test_fng_param_count():
    p, L, d = 768, 5, 5
    m = CoFrGeNetFNG(p, L, d)
    expected = (
        (2 * p * p + 2 * p)  # gate_up weight + bias
        + ((d + 1) * L * p + (d + 1) * L)  # coeffs weight + bias
        + (L * p + p)  # down weight + bias
    )
    assert sum(t.numel() for t in m.parameters()) == expected


@pytest.mark.parametrize("d", [2, 5, 8])
def test_deep_ladder_gradients_finite(d):
    """The log/exp division (paper eq. 8) is what makes d > 1 trainable
    without the incremental training schedule of the original CoFrGeNet."""
    torch.manual_seed(0)
    m = CoFrGeNetFNG(64, n_ladders=8, depth=d)
    x = torch.randn(32, 64)
    m(x).pow(2).mean().backward()
    for name, t in m.named_parameters():
        assert t.grad is not None, name
        assert torch.isfinite(t.grad).all(), name


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("d", [1, 2, 5, 8])
def test_triton_ladder_matches_eager(d):
    torch.manual_seed(0)
    c = torch.randn(4096, 8, d + 1, device=DEVICE) * 2
    ref = c[..., 0] + continuant_fraction(c[..., 1:], eps=0.1)
    got = ladder_forward(c, eps=0.1)
    # near a pole |f~| is huge and both fp32 evaluations carry ~1e-4 noise,
    # so judge by relative error with a depth-tolerant bound
    err = ((ref - got).abs() / ref.abs().clamp_min(1.0)).max().item()
    assert err < 1e-3


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("d", [1, 2, 5, 8])
@pytest.mark.parametrize("eps", [0.1, 2.0])  # 2.0 exercises the clamped branch
def test_triton_backward_matches_eager(d, eps):
    torch.manual_seed(0)
    c = (torch.randn(64, 8, d + 1, device=DEVICE) * 1.5).requires_grad_(True)
    c_ref = c.detach().clone().requires_grad_(True)
    g = torch.randn(64, 8, device=DEVICE)

    # eager reference (autograd through continuant recursion + log/exp)
    out_ref = c_ref[..., 0] + continuant_fraction(c_ref[..., 1:], eps=eps)
    out_ref.backward(g)

    # fused triton forward + backward
    out = ladder_apply(c, eps=eps)
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)
    out.backward(g)

    assert torch.isfinite(c.grad).all()
    err = ((c.grad - c_ref.grad).abs() / c_ref.grad.abs().clamp_min(1.0)).max()
    assert err < 1e-3, err


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("d", [1, 2, 5, 8])
def test_triton_backward_clamped_denominator(d):
    """Tiny nonzero coefficients: |den| << eps exercises the clamp branch."""
    c = torch.randn(128, 4, d + 1, device=DEVICE).requires_grad_(True)
    with torch.no_grad():
        c[..., 1:] *= 1e-6
    c_ref = c.detach().clone().requires_grad_(True)
    g = torch.randn(128, 4, device=DEVICE)

    (c_ref[..., 0] + continuant_fraction(c_ref[..., 1:], eps=0.1)).backward(g)
    ladder_apply(c, eps=0.1).backward(g)

    assert torch.isfinite(c.grad).all()
    assert torch.allclose(c.grad, c_ref.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
def test_triton_backward_at_zero_numerator():
    """d=2, a1=a2=0 -> num=a2=0, den=1, so frac=a2 and d frac/d a2 = 1.
    (eager autograd returns 0 at a2=0 only because torch.abs(0).grad == 0;
    the analytic adjoint gives the true derivative of num / den_c.)"""
    c = torch.zeros(16, 4, 3, device=DEVICE, requires_grad=True)
    g = torch.randn(16, 4, device=DEVICE)
    ladder_apply(c, eps=0.1).backward(g)
    assert torch.allclose(c.grad[..., 0], g)
    assert torch.allclose(c.grad[..., 1], torch.zeros_like(g))
    assert torch.allclose(c.grad[..., 2], g)


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
def test_triton_module_matches_eager_module():
    torch.manual_seed(0)
    eager = CoFrGeNetFNG(64, n_ladders=8, depth=5).to(DEVICE)
    fast = CoFrGeNetFNGTriton(64, n_ladders=8, depth=5).to(DEVICE)
    fast.load_state_dict(eager.state_dict())

    x = torch.randn(16, 7, 64, device=DEVICE)
    y_ref = eager(x)
    y = fast(x)
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

    y_ref.pow(2).mean().backward()
    y.pow(2).mean().backward()
    for (n1, p1), (n2, p2) in zip(eager.named_parameters(), fast.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, n1
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-2, atol=1e-4), (
            n1,
            (p1.grad - p2.grad).abs().max().item(),
        )


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_mixed_precision_fwd_bwd(dtype):
    """bf16/fp16 storage keeps native dtype; fp32 internal math is compared
    against the exact evaluation of the SAME quantized coefficients."""
    torch.manual_seed(0)
    c_low = (torch.randn(64, 8, 6, device=DEVICE) * 1.5).to(dtype).requires_grad_(True)
    c_ref = c_low.detach().float().requires_grad_(True)
    g32 = torch.randn(64, 8, device=DEVICE)

    out_ref = c_ref[..., 0] + continuant_fraction(c_ref[..., 1:], eps=0.1)
    out_ref.backward(g32)

    out = ladder_apply(c_low, eps=0.1)
    assert out.dtype == dtype
    out.backward(g32.to(dtype))
    assert c_low.grad.dtype == dtype
    assert torch.isfinite(c_low.grad.float()).all()

    tol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    fwd_err = ((out.float() - out_ref).abs() / out_ref.abs().clamp_min(1.0)).max()
    bwd_err = ((c_low.grad.float() - c_ref.grad).abs() / c_ref.grad.abs().clamp_min(1.0)).max()
    assert fwd_err < tol, fwd_err
    assert bwd_err < tol, bwd_err


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_module_mixed_precision(dtype):
    m = CoFrGeNetFNGTriton(64, n_ladders=8, depth=5).to(DEVICE).to(dtype)
    x = torch.randn(16, 7, 64, device=DEVICE, dtype=dtype)
    y = m(x)
    assert y.dtype == dtype
    y.float().pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad.float()).all(), name
