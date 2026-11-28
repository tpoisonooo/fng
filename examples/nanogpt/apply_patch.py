"""Patch a fresh nanoGPT clone so GPTConfig supports mlp_type="fng".

Usage:
    git clone https://github.com/karpathy/nanoGPT.git
    python examples/nanogpt/apply_patch.py /path/to/nanoGPT
"""

import sys


FNG_CLASS = '''
class MLPFNG(nn.Module):
    """Drop-in CoFrGeNet-FNG replacement for nanoGPT's MLP (keeps dropout)."""

    def __init__(self, config):
        super().__init__()
        from cofrgenet_fng import CoFrGeNetFNG, CoFrGeNetFNGTriton
        base = CoFrGeNetFNGTriton if config.fng_triton else CoFrGeNetFNG
        self.fng = base(config.n_embd, n_ladders=config.fng_n_ladders, depth=config.fng_depth)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fng(x))

'''


def patch(model_py: str) -> None:
    src = open(model_py, encoding="utf-8").read()
    if "MLPFNG" in src:
        print("already patched:", model_py)
        return
    src = src.replace("class Block(nn.Module):", FNG_CLASS + "class Block(nn.Module):", 1)
    src = src.replace(
        "        self.mlp = MLP(config)",
        '        self.mlp = MLP(config) if config.mlp_type == "ffn" else MLPFNG(config)',
        1,
    )
    src = src.replace(
        "    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster",
        "    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster\n"
        '    mlp_type: str = "ffn"       # "ffn" or "fng"\n'
        "    fng_n_ladders: int = 5\n"
        "    fng_depth: int = 5\n"
        "    fng_triton: bool = True",
        1,
    )
    open(model_py, "w", encoding="utf-8").write(src)
    print("patched:", model_py)


def patch_train(train_py: str) -> None:
    """nanoGPT's train.py has its own config whitelist; forward FNG flags too."""
    src = open(train_py, encoding="utf-8").read()
    if "fng_n_ladders" in src:
        print("already patched:", train_py)
        return
    src = src.replace(
        "bias = False # do we use bias inside LayerNorm and Linear layers?",
        "bias = False # do we use bias inside LayerNorm and Linear layers?\n"
        'mlp_type = "ffn"       # "ffn" or "fng"\n'
        "fng_n_ladders = 5\n"
        "fng_depth = 5\n"
        "fng_triton = True",
        1,
    )
    src = src.replace(
        "bias=bias, vocab_size=None, dropout=dropout)",
        "bias=bias, vocab_size=None, dropout=dropout, mlp_type=mlp_type,\n"
        "                  fng_n_ladders=fng_n_ladders, fng_depth=fng_depth, fng_triton=fng_triton)",
        1,
    )
    open(train_py, "w", encoding="utf-8").write(src)
    print("patched:", train_py)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python {sys.argv[0]} /path/to/nanoGPT")
    root = sys.argv[1].rstrip("/")
    patch(root + "/model.py")
    patch_train(root + "/train.py")
