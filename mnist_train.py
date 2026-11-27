"""
mnist_train.py

End-to-end forward + backward training test on MNIST (8-core CPU friendly).

Compares three channel-mixing blocks inside the same tiny residual classifier:
  * ffn        : VanillaFFN(p, alpha=4)
  * fng-eager  : CoFrGeNetFNG(p, n_ladders=16, depth=5)  (eager PyTorch)
  * fng-triton : CoFrGeNetFNGTriton(...)                 (fused Triton fwd+bwd)

MNIST is loaded directly from the IDX .gz files (no torchvision dependency,
which avoids torch/torchvision ABI coupling). Data is cached in --data-dir
(default /tmp/mnist).

Usage:
    TRITON_DEFAULT_BACKEND=cpu python mnist_train.py
"""

import argparse
import gzip
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cofrgenet_fng import (
    HAS_TRITON,
    CoFrGeNetFNG,
    CoFrGeNetFNGTriton,
    VanillaFFN,
    triton_backend,
)

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _download(fname: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        return path
    last_err = None
    for base in MIRRORS:
        try:
            print(f"downloading {base + fname} ...")
            urllib.request.urlretrieve(base + fname, path)
            return path
        except Exception as e:  # try the next mirror
            last_err = e
    raise RuntimeError(f"could not download {fname}: {last_err}")


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic >> 16 == 0, "not an IDX file"
        dtype = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}[
            (magic >> 8) & 0xFF
        ]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=dtype).reshape(dims)


def load_mnist(data_dir: str):
    tensors = {}
    for key, fname in FILES.items():
        arr = _read_idx(_download(fname, data_dir))
        if key.endswith("_x"):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0).flatten(1)
            t = (t - 0.1307) / 0.3081  # standard MNIST normalization
        else:
            t = torch.from_numpy(arr.astype(np.int64))
        tensors[key] = t
    return (
        TensorDataset(tensors["train_x"], tensors["train_y"]),
        TensorDataset(tensors["test_x"], tensors["test_y"]),
    )


class MnistNet(nn.Module):
    """x -> Linear(784, p) -> h + block(LayerNorm(h)) -> Linear(p, 10)."""

    def __init__(self, block: nn.Module, p: int = 128):
        super().__init__()
        self.inp = nn.Linear(28 * 28, p)
        self.norm = nn.LayerNorm(p)
        self.block = block
        self.head = nn.Linear(p, 10)

    def forward(self, x):
        h = self.inp(x)
        h = h + self.block(self.norm(h))
        return self.head(h)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(name, block, p, train_set, test_set, epochs, batch_size, lr, device):
    torch.manual_seed(0)
    model = MnistNet(block, p).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n--- {name} (params={n_params:,}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total_loss, n_steps = time.perf_counter(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_steps += 1
        acc = evaluate(model, test_loader)
        ms = (time.perf_counter() - t0) / n_steps * 1e3
        print(
            f"epoch {epoch}: loss={total_loss / n_steps:.4f}  "
            f"test_acc={acc:.4f}  step={ms:.1f} ms"
        )
        if epoch == epochs:
            assert acc > 0.90, f"{name}: test accuracy {acc:.4f} <= 0.90"
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="all", choices=["all", "ffn", "fng-eager", "fng-triton"]
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--n-ladders", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--data-dir", default="/tmp/mnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device={device}, triton={triton_backend()}")
    train_set, test_set = load_mnist(args.data_dir)
    p = args.p

    blocks = {
        "ffn": lambda: VanillaFFN(p, alpha=4.0),
        "fng-eager": lambda: CoFrGeNetFNG(p, n_ladders=args.n_ladders, depth=args.depth),
        "fng-triton": lambda: CoFrGeNetFNGTriton(
            p, n_ladders=args.n_ladders, depth=args.depth
        ),
    }
    names = list(blocks) if args.model == "all" else [args.model]
    if "fng-triton" in names and not HAS_TRITON:
        raise SystemExit("fng-triton selected but triton is not installed")

    results = {}
    for name in names:
        results[name] = train(
            name, blocks[name](), p, train_set, test_set,
            args.epochs, args.batch_size, args.lr, device,
        )

    # When both FNG variants are trained from the same seed/data order, report
    # how close the Triton fwd+bwd and eager paths stay after training.
    if "fng-eager" in results and "fng-triton" in results:
        diff = max(
            (p1 - p2).abs().max().item()
            for p1, p2 in zip(
                results["fng-eager"].parameters(), results["fng-triton"].parameters()
            )
        )
        print(f"\nmax param drift (eager vs triton) after training: {diff:.2e}")
    print("MNIST forward-backward training test passed.")


if __name__ == "__main__":
    main()
