"""Single-GPU Muon optimizer (Newton-Schulz orthogonalized momentum) + AdamW hybrid.

Applies Muon to 2D hidden weight matrices (attention projections, FFN/FNG
linears) and keeps embeddings, positional embeddings, layer-norm gains and
biases on AdamW — the standard hybrid setup used in nanoGPT speedruns.
"""

import torch
from torch import Tensor


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Approximate the orthogonal factor U V^T of G = U S V^T."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                g = g.lerp_(buf, mom) if group["nesterov"] else buf
                if group["weight_decay"] != 0:
                    p.mul_(1 - lr * group["weight_decay"])
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u.view_as(p).to(p.dtype), alpha=-lr * scale)
        return loss


class MuonAdamW:
    """Wraps one Muon optimizer (2D hidden weights) + one AdamW (rest)."""

    def __init__(
        self,
        model,
        muon_lr=0.015,
        adam_lr=6e-4,
        weight_decay=0.0,
        adam_weight_decay=0.1,
        betas=(0.9, 0.95),
    ):
        hidden_2d, other = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_embed = any(k in name for k in ("wte", "wpe", "lm_head"))
            if p.ndim == 2 and not is_embed:
                hidden_2d.append(p)
            else:
                other.append(p)
        decay = [p for p in other if p.ndim >= 2]
        no_decay = [p for p in other if p.ndim < 2]
        self.muon = Muon(hidden_2d, lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=weight_decay)
        self.adam = torch.optim.AdamW(
            [
                dict(params=decay, weight_decay=adam_weight_decay),
                dict(params=no_decay, weight_decay=0.0),
            ],
            lr=adam_lr,
            betas=betas,
            fused=(torch.cuda.is_available()),
        )

    def set_lr(self, muon_lr, adam_lr):
        for g in self.muon.param_groups:
            g["lr"] = muon_lr
        for g in self.adam.param_groups:
            g["lr"] = adam_lr

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step()
        return self.adam.step(closure)
