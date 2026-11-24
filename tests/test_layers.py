import pytest
import torch

from cofrgenet_fng import (
    HAS_TRITON,
    CoFrGeNetFNG,
    SwiGLUFFN,
    VanillaFFN,
    continuant_fraction,
    ladder_forward,
)


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
    c = torch.randn(4096, 8, d + 1) * 2
    ref = c[..., 0] + continuant_fraction(c[..., 1:], eps=0.1)
    got = ladder_forward(c, eps=0.1)
    # near a pole |f~| is huge and both fp32 evaluations carry ~1e-4 noise,
    # so judge by relative error with a depth-tolerant bound
    err = ((ref - got).abs() / ref.abs().clamp_min(1.0)).max().item()
    assert err < 1e-3
