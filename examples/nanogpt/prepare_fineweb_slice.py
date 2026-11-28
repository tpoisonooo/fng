"""Stream a FineWeb-Edu slice and write train.bin / val.bin for nanoGPT.

Usage:
    python examples/nanogpt/prepare_fineweb_slice.py /path/to/nanoGPT --tokens 20000000
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nanogpt_dir")
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--val-frac", type=float, default=0.005)
    args = parser.parse_args()

    out_dir = os.path.join(args.nanogpt_dir, "data", "fineweb_slice")
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    if os.path.exists(train_path):
        print("already prepared:", os.path.getsize(train_path) // 2, "tokens")
        return

    enc = tiktoken.get_encoding("gpt2")
    ds = iter(
        load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
    )
    buf, n_tokens = [], 0
    for ex in ds:
        ids = enc.encode_ordinary(ex["text"])
        buf.append(np.array(ids + [enc.eot_token], dtype=np.uint16))
        n_tokens += len(ids) + 1
        if len(buf) % 500 == 0:
            print(f"  {len(buf)} docs, {n_tokens / 1e6:.2f}M tokens")
        if n_tokens >= args.tokens:
            break

    all_ids = np.concatenate(buf)
    n_val = int(len(all_ids) * args.val_frac)
    all_ids[n_val:].tofile(train_path)
    all_ids[:n_val].tofile(os.path.join(out_dir, "val.bin"))
    print(f"done: train={len(all_ids) - n_val:,} tokens, val={n_val:,} tokens")


if __name__ == "__main__":
    main()
