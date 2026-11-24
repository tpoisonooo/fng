# CoFrGeNet-FNG (unofficial PyTorch implementation)

Unofficial PyTorch implementation of **CoFrGeNet-FNG** — the *Non-Expanding
Gated Continued Fraction* feed-forward layer from IBM Research's paper
[Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers
in Language Models](https://arxiv.org/abs/2611.26835) (NeurIPS 2026) — plus
classic FFN baselines and a fused Triton kernel for the continued-fraction
ladders.

> **Note:** This is an independent, from-the-paper implementation. The
> authors' official repository, [IBM/CoFrNets](https://github.com/IBM/CoFrNets),
> had not published code at the time of writing.

## What is CoFrGeNet-FNG?

A drop-in replacement for the FFN sublayer of transformer (and SSM/MoE)
blocks with the same `(…, p) -> (…, p)` API, but ~4x fewer parameters:

1. **Non-expanding gated projection**: `x ∈ R^p → z ∈ R^p` (expansion
   factor α = 1), computed as `z = a ⊙ silu(b)` where `[a, b]` is a split
   of one linear map — 2p² parameters.
2. **L continued-fraction ladders of depth d**: each ladder has d+1 learned
   coefficient vectors, a_k = w_kᵀz, and evaluates the rational function

   ```
   f~(a_1..a_d) = 1/(a_1 + 1/(a_2 + ... + 1/a_d))
   ```

   via the continuant recursion (paper eqs. 3–6), needing only **one**
   division per ladder instead of d.
3. **log/exp division** (paper eq. 8): the remaining division is rewritten
   as `sgn·sgn·exp(log|num| − log max(|den|, ε))`, which stabilizes training
   of deep ladders (d > 1) **without** the incremental training schedule
   required by the original CoFrGeNet.
4. A linear projection `R^L → R^p`.

Parameter count: `2p² + L·p·(d+1) + L·p` vs. `2αp²` for a vanilla FFN or
`3αp²` for SwiGLU (α ≥ 2).

## Install

```bash
pip install -e .          # torch is the only hard dependency
```

Optional, for the fused Triton ladder kernel:

```bash
# GPU
pip install triton
# CPU (experimental triton-cpu backend, pick your platform/python)
pip install "https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0%2Bcpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

Note: install the triton-cpu wheel **last**. torch pins its own triton
build, so a later `pip install` that touches torch's dependencies may
replace the `+cpu` wheel with mainline triton (which has no CPU backend).

## Quickstart

```python
import torch
from cofrgenet_fng import CoFrGeNetFNG

ffn = CoFrGeNetFNG(p=768, n_ladders=5, depth=5)   # same API as an FFN
x = torch.randn(2, 128, 768)
y = ffn(x)                                        # (2, 128, 768)

sum(t.numel() for t in ffn.parameters())          # 1,208,862  (vs 4.7M vanilla FFN)
```

## Examples

Run from the repo root.

### `fng_vs_ffn.py` — FNG vs. FFN baselines

Verifies the continuant math, then compares parameters, speed, and learning
on a teacher-student task. Measured on an 8-core CPU (torch 2.9.1):

```
module                             params   vs FFN    fwd+bwd
VanillaFFN (alpha=4)            4,722,432    1.00x   259.90 ms
SwiGLUFFN (alpha=8/3)           4,723,456    1.00x   234.54 ms
CoFrGeNetFNG (L=5, d=5)         1,208,862    0.26x    69.60 ms
CoFrGeNetFNG (L=8, d=8)         1,243,464    0.26x    74.09 ms
```

On the toy fitting task the FNG block uses 3x fewer parameters but converges
slower than a full-size FFN — consistent with the paper, whose parity/better
results are at language-model pre-training scale, not tiny regression tasks.

### `fng_triton.py` — fused Triton ladder kernel

The ladder is a serial loop of tiny elementwise ops in eager PyTorch; the
kernel fuses it into a single pointwise pass (load d+1 coefficients, continuant
recursion in registers, one log/exp division, store). Forward-only; the eager
path remains the training reference. Measured with the triton-cpu backend
(numbers on a shared CPU vary run to run; the ladder-only speedup is the
consistent part — at whole-block level the p→2p GEMM dominates and the
difference is within noise):

```
(B=16, S=256, L=8, d=8)  N=32,768 ladders:     eager 0.679 ms | triton 0.164 ms (4.13x)
(B=64, S=1024, L=16, d=8) N=1,048,576 ladders: eager 16.262 ms | triton 5.945 ms (2.74x)
```

Run it with `TRITON_DEFAULT_BACKEND=cpu python fng_triton.py` (or on a GPU,
plain `python fng_triton.py`).

## Tests

```bash
pip install pytest
pytest -v          # triton tests are skipped automatically if unavailable
```

## Repository layout

```
cofrgenet_fng/        the package
  layers.py           VanillaFFN, SwiGLUFFN, CoFrGeNetFNG, continuant_fraction
  triton_ladder.py    fused Triton kernel (optional, forward-only)
fng_vs_ffn.py         example: FNG vs. FFN baselines
fng_triton.py         example: Triton kernel correctness + benchmarks
tests/                pytest suite
```

## Citation

```bibtex
@article{dhurandhar2026cofrgenetfng,
  title   = {Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models},
  author  = {Dhurandhar, Amit and Chenthamarakshan, Vijil and Ramamurthy, Karthikeyan Natesan and Wei, Dennis and Nair, Rahul},
  journal = {arXiv:2611.26835},
  year    = {2026}
}

@inproceedings{dhurandhar2026cofrgenet,
  title     = {CoFrGeNet: Continued Fraction Architectures for Language Generation},
  author    = {Dhurandhar, A. and Chenthamarakshan, V. and Wei, D. and Pedapati, T. and Ramamurthy, K. N. and Nair, R.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
