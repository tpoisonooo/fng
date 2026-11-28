"""Variance-calibrated init for FNG inside nanoGPT.

In GPT-2 the FFN sublayer output has std ~= 0.02*sqrt(p)/sqrt(2*n_layer)
(approximately 0.073 for p=768, n_layer=12) thanks to the c_proj residual
scaling. FNG's gated ladder path has a depth-dependent variance: with flat
std=0.02 init, even depths attenuate it (~0.007) and odd depths amplify it
(~0.46) because the continuant denominator K_d crosses near the eps clamp.

This helper rescales each FNG block's `down.weight` so that the sublayer's
initial output std matches the FFN target, removing the depth/width-dependent
scale mismatch. See examples/nanogpt/README.md for why this is needed.
"""

import argparse
import math

import torch


@torch.no_grad()
def calibrate_fng_init(model, target_std: float | None = None, tokens: int = 2048) -> list[float]:
    """Rescale every MLPFNG.down.weight so the block output std ~= target_std."""
    cfg = model.config
    if target_std is None:
        target_std = 0.02 * math.sqrt(cfg.n_embd) / math.sqrt(2 * cfg.n_layer)

    dummy = torch.randn(1, min(tokens, cfg.block_size), cfg.n_embd, device=next(model.parameters()).device)
    if hasattr(model, "transformer"):
        dummy = model.transformer.h[0].ln_2(dummy)

    stds = []
    for blk in model.transformer.h:
        fng = blk.mlp.fng if hasattr(blk.mlp, "fng") else blk.mlp
        out = fng(dummy).float()
        cur_std = out.std().item()
        if cur_std > 0:
            fng.down.weight.mul_(target_std / cur_std)
        stds.append(cur_std)
    return stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nanogpt_dir")
    parser.add_argument("--n-ladders", type=int, default=8)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, args.nanogpt_dir)
    from model import GPT, GPTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = GPT(GPTConfig(mlp_type="fng", fng_n_ladders=args.n_ladders, fng_depth=args.depth, bias=True)).to(device)
    before = calibrate_fng_init(m)
    dummy = torch.randn(1, 64, m.config.n_embd, device=device)
    after_std = m.transformer.h[0].mlp(m.transformer.h[0].ln_2(dummy)).float().std().item()
    print(f"L={args.n_ladders}, d={args.depth}: init std {before[0]:.4f} -> {after_std:.4f} (target~=0.0736)")
