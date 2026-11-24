"""Feed-forward layers: classic FFN baselines and the CoFrGeNet-FNG block.

CoFrGeNet-FNG ("Non-Expanding Gated Continued Fraction" FFN replacement,
arXiv:2611.26835) replaces the transformer FFN sublayer with:

1. a non-expanding gated projection x in R^p -> z in R^p (alpha = 1),
   z = a * silu(b) where [a, b] = split(Linear(x))               (2p^2 params)
2. L continued-fraction "ladders" of depth d. Each ladder has d+1 learned
   coefficient vectors; a_k = w_k^T z. The fractional part

       f~(a_1..a_d) = 1/(a_1 + 1/(a_2 + ... + 1/a_d))

   is evaluated with the continuant recursion (paper eqs. 3-6), so a depth-d
   ladder needs only ONE division instead of d.
3. that division replaced by the log/exp form (paper eq. 8), which is what
   makes d > 1 trainable without incremental training:

       f~ = sgn(K_{d-1}) * sgn(K_d)
            * exp( log|K_{d-1}| - log(max(|K_d|, eps)) )
4. a linear projection R^L -> R^p.

All modules share the same API: (..., p) -> (..., p), i.e. they are drop-in
replacements for the FFN sublayer of a transformer block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VanillaFFN",
    "SwiGLUFFN",
    "CoFrGeNetFNG",
    "continuant_fraction",
]


class VanillaFFN(nn.Module):
    """GPT-2 style feed-forward network. Params ~= 2 * alpha * p^2."""

    def __init__(self, p: int, alpha: float = 4.0):
        super().__init__()
        self.up = nn.Linear(p, int(alpha * p))
        self.down = nn.Linear(int(alpha * p), p)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLUFFN(nn.Module):
    """Llama style gated FFN. Params ~= 3 * alpha * p^2."""

    def __init__(self, p: int, alpha: float = 8 / 3):
        super().__init__()
        hidden = int(alpha * p)
        self.gate = nn.Linear(p, hidden)
        self.up = nn.Linear(p, hidden)
        self.down = nn.Linear(hidden, p)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def continuant_fraction(A: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """
    Evaluate f~(a_1, ..., a_d) = 1/(a_1 + 1/(a_2 + ... + 1/a_d)) stably.

    Uses the continuant recursion K_0 = 1, K_1 = a_d,
    K_k = a_{d-k+1} * K_{k-1} + K_{k-2}, which gives f~ = K_{d-1} / K_d
    with a single division, then rewrites that division in log/exp form
    (paper eq. 8) to avoid poles during training.

    A: (..., d) tensor of partial denominators a_1..a_d. Returns (...,).
    """
    d = A.shape[-1]
    Km2 = torch.ones_like(A[..., 0])  # K_0 = 1
    Km1 = A[..., -1]  # K_1 = a_d
    for k in range(2, d + 1):
        K = A[..., d - k] * Km1 + Km2  # K_k = a_{d-k+1} K_{k-1} + K_{k-2}
        Km2, Km1 = Km1, K
    # after the loop: Km1 = K_d(a_1..a_d), Km2 = K_{d-1}(a_2..a_d)
    num, den = Km2, Km1
    return (
        torch.sign(num)
        * torch.sign(den)
        * torch.exp(
            torch.log(num.abs().clamp_min(1e-12))
            - torch.log(den.abs().clamp_min(eps))
        )
    )


class CoFrGeNetFNG(nn.Module):
    """
    Non-Expanding Gated continued-fraction FFN replacement.

    Params: 2*p^2 (gate) + L*p*(d+1) (ladder coefficients) + L*p (down proj).
    With p=768, L=5, d=5 that is ~1.2M vs ~4.7M for a vanilla FFN (alpha=4).
    """

    def __init__(self, p: int, n_ladders: int = 5, depth: int = 5, eps: float = 0.1):
        super().__init__()
        self.p, self.L, self.d, self.eps = p, n_ladders, depth, eps
        self.gate_up = nn.Linear(p, 2 * p)  # non-expanding gated projection
        self.coeffs = nn.Linear(p, (depth + 1) * n_ladders)  # a_0..a_d per ladder
        self.down = nn.Linear(n_ladders, p)

    def forward(self, x):
        a, b = self.gate_up(x).chunk(2, dim=-1)
        z = a * F.silu(b)  # gated, still p-dim (expansion factor alpha = 1)
        c = self.coeffs(z).unflatten(-1, (self.L, self.d + 1))
        a0, A = c[..., 0], c[..., 1:]  # (..., L) and (..., L, d)
        ladder_out = a0 + continuant_fraction(A, self.eps)  # (..., L)
        return self.down(ladder_out)
