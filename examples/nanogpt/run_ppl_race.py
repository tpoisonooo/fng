"""PPL-vs-wall-clock race: how long does each block take to hit the same PPL?

Trains GPT-2 Small nanoGPT variants on the same FineWeb-Edu slice, identical
batch order and LR schedule shape, and stops each run as soon as its val loss
on a FIXED held-out validation set reaches the target PPL (default 250).

Experiment management:
  * run name : gpt2small-<ffn|fng>-L<ladders>d<depth>-<adamw|muon>-lr<lr>
  * logs     : <exp-dir>/<run_name>/{config.json, history.jsonl, summary.json, log.txt}
  * renders  : <render-dir>/{results.json, ppl_vs_time.jpg, ppl_vs_tokens.jpg, RESULTS.md}
  Renders merge ALL runs found under <exp-dir>, so separate optimizer jobs
  end up on the same charts without re-running.

Usage:
    python examples/nanogpt/run_ppl_race.py /path/to/nanoGPT \
        --data-dir /path/to/fineweb_slice --optimizer adamw --target-ppl 250
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrate_fng_init import calibrate_fng_init  # noqa: E402
from muon import MuonAdamW  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


class Tee:
    def __init__(self, log_path):
        self.f = open(log_path, "a", encoding="utf-8")
        self.prev_stdout = sys.stdout

    def write(self, s):
        self.prev_stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.prev_stdout.flush()
        self.f.flush()

    def close(self):
        if sys.stdout is self:
            sys.stdout = self.prev_stdout
        self.f.close()


def fmt_lr(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")


class FixedBatches:
    def __init__(self, data_dir, batch_size=32, block_size=1024, device="cuda"):
        self.device = device
        self.bsz, self.T = batch_size, block_size
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(99)
        ix = torch.randint(len(val) - block_size, (50, batch_size), generator=g)
        self.val = []
        for row in ix:
            vx = torch.stack([torch.from_numpy(val[i : i + block_size].astype(np.int64)) for i in row]).to(device)
            vy = torch.stack([torch.from_numpy(val[i + 1 : i + 1 + block_size].astype(np.int64)) for i in row]).to(device)
            self.val.append((vx, vy))

    def train_batch(self):
        ix = torch.randint(len(self.train) - self.T, (self.bsz,))
        x = torch.stack([torch.from_numpy(self.train[i : i + self.T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.train[i + 1 : i + 1 + self.T].astype(np.int64)) for i in ix])
        return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)

    @torch.no_grad()
    def val_loss(self, model, dtype):
        model.eval()
        total = 0.0
        for vx, vy in self.val:
            with torch.amp.autocast(self.device, dtype=dtype):
                _, loss = model(vx, vy)
            total += loss.item()
        model.train()
        return total / len(self.val)


def run_one(run_name, run_dir, model, data, opt_args, max_steps, target_ppl, dtype, eval_every=50):
    os.makedirs(run_dir, exist_ok=True)
    tee = Tee(os.path.join(run_dir, "log.txt"))
    sys.stdout = tee

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": 1337,
                "n_params": model.get_num_params(),
                "peak_lr": opt_args["lr"],
                "max_steps": max_steps,
                "target_ppl": target_ppl,
                "dtype": str(dtype),
                "device": data.device,
            },
            f,
            indent=2,
        )

    history_f = open(os.path.join(run_dir, "history.jsonl"), "w")
    target_loss = math.log(target_ppl)

    # warm up the Triton JIT pipeline before timing
    X, Y = data.train_batch()
    with torch.amp.autocast(data.device, dtype=dtype):
        _, loss = model(X, Y)
    loss.backward()
    if hasattr(opt_args["opt"], "zero_grad"):
        opt_args["opt"].zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[{run_name}] start (target ppl={target_ppl}, loss={target_loss:.4f}, max_steps={max_steps})", flush=True)
    train_seconds = 0.0
    hit = False
    for step in range(1, max_steps + 1):
        progress_lr = min(1.0, step / 100.0)
        if step > 100:
            decay_p = (step - 100) / (max_steps - 100)
            progress_lr = 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * decay_p))
        lr = opt_args["lr"] * progress_lr
        opt_args["set_lr"](lr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        X, Y = data.train_batch()
        with torch.amp.autocast(data.device, dtype=dtype):
            _, loss = model(X, Y)
        opt_args["opt"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_args["opt"].step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - t0

        if step % eval_every == 0 or step == 1:
            vl = data.val_loss(model, dtype)
            ppl = math.exp(vl)
            rec = {
                "step": step,
                "train_seconds": round(train_seconds, 3),
                "tokens": step * data.bsz * data.T,
                "val_loss": vl,
                "ppl": ppl,
            }
            history_f.write(json.dumps(rec) + "\n")
            history_f.flush()
            print(f"[{run_name}] step {step:>4} | {train_seconds:6.1f}s | val_loss={vl:.4f} | ppl={ppl:7.2f}", flush=True)
            if vl <= target_loss:
                hit = True
                break

    history_f.close()
    last = rec
    summary = {
        "run_name": run_name,
        "n_params": model.get_num_params(),
        "hit_target": hit,
        "final_step": last["step"],
        "train_seconds_to_target": last["train_seconds"] if hit else None,
        "steps_to_target": last["step"] if hit else None,
        "tokens_to_target": last["tokens"] if hit else None,
        "best_val_loss": last["val_loss"],
        "best_ppl": last["ppl"],
        "throughput_tok_per_sec": round(data.bsz * data.T * last["step"] / max(last["train_seconds"], 1e-6), 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    tee.close()
    print(f"[{run_name}] summary: {summary}", flush=True)
    return summary


def load_all_histories(exp_dir):
    runs = {}
    if not os.path.isdir(exp_dir):
        return runs
    for name in sorted(os.listdir(exp_dir)):
        hfile = os.path.join(exp_dir, name, "history.jsonl")
        sfile = os.path.join(exp_dir, name, "summary.json")
        cfile = os.path.join(exp_dir, name, "config.json")
        if os.path.isfile(hfile) and os.path.isfile(sfile) and os.path.isfile(cfile):
            with open(cfile) as f:
                cfg = json.load(f)
            with open(sfile) as f:
                summ = json.load(f)
            with open(hfile) as f:
                hist = [json.loads(l) for l in f if l.strip()]
            runs[name] = {"config": cfg, "summary": summ, "history": hist}
    return runs


def render_all(exp_dir, render_dir, target_ppl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(render_dir, exist_ok=True)

    runs = load_all_histories(exp_dir)
    with open(os.path.join(render_dir, "results.json"), "w") as f:
        json.dump({n: r["summary"] for n, r in runs.items()}, f, indent=2)

    palette = ["#3B6CB4", "#E07A1F", "#3E884F", "#8E44AD", "#16A085", "#C0392B", "#7F8C8D", "#2C3E50"]
    for xkey, xlabel, filename in [
        ("train_seconds", "纯训练墙钟耗时 (秒)", "ppl_vs_time.jpg"),
        ("tokens", "训练 token 数", "ppl_vs_tokens.jpg"),
    ]:
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for idx, (name, r) in enumerate(runs.items()):
            color = palette[idx % len(palette)]
            xs = [h[xkey] for h in r["history"]]
            ys = [h["ppl"] for h in r["history"]]
            hit = r["summary"]["hit_target"]
            short = name.replace("gpt2small-", "")
            ax.plot(xs, ys, lw=2, color=color, label=f"{short} ({'hit' if hit else 'miss'})")
            if hit:
                ax.scatter([xs[-1]], [ys[-1]], color=color, zorder=5)
        ax.axhline(target_ppl, color="#B23B3B", ls="--", lw=1.3, label=f"目标 PPL = {target_ppl}")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("固定验证集 PPL (log)")
        ax.set_title(f"nanoGPT GPT-2 Small PPL 竞速 ({filename.split('_vs_')[1].split('.')[0]})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(render_dir, filename), format="jpg", dpi=160, facecolor="white")
        if filename == "ppl_vs_time.jpg":
            fig.savefig("/tmp/ppl_race_preview.png", dpi=90, facecolor="white")
        plt.close(fig)

    lines = [
        "# nanoGPT PPL race results",
        "",
        f"Target PPL: **{target_ppl}** (fixed validation subset, same batch order and seed).",
        "",
        "| Run | Params | Opt | Hit target? | Steps to target | Train seconds to target | Throughput (tok/s) | Final/Best PPL |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        s = r["summary"]
        step_s = s["steps_to_target"] if s["hit_target"] else f"{s['final_step']}+"
        sec_s = f"{s['train_seconds_to_target']:.1f}s" if s["hit_target"] else "not hit"
        hit_s = "yes" if s["hit_target"] else "no"
        lines.append(
            f"| `{name}` | {s['n_params'] / 1e6:.2f}M | {s.get('optimizer', 'adamw')} | {hit_s} | {step_s} | {sec_s} | {s['throughput_tok_per_sec']:,.0f} | {s['best_ppl']:.1f} |"
        )
    with open(os.path.join(render_dir, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"rendered {len(runs)} runs -> {render_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nanogpt_dir")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.015)
    parser.add_argument("--target-ppl", type=float, default=250.0)
    parser.add_argument("--ffn-max-steps", type=int, default=2500)
    parser.add_argument("--fng-max-steps", type=int, default=4500)
    parser.add_argument("--exp-dir", default=os.path.join(REPO_ROOT, "experiments"))
    parser.add_argument("--render-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.nanogpt_dir)
    from model import GPT, GPTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    data = FixedBatches(args.data_dir, device=device)

    specs = [
        dict(block="ffn", L=0, d=0, lr=6e-4, max_steps=args.ffn_max_steps),
        dict(block="fng", L=16, d=5, lr=3e-4, max_steps=args.fng_max_steps),
        dict(block="fng", L=8, d=5, lr=6e-4, max_steps=args.fng_max_steps),
    ]
    if args.only:
        specs = [s for s in specs if s["block"] in args.only.split(",")]

    for sp in specs:
        spec_lr = sp["lr"] if args.optimizer == "adamw" else args.muon_lr
        run_name = f"gpt2small-{sp['block']}-L{sp['L']}d{sp['d']}-{args.optimizer}-lr{fmt_lr(spec_lr)}"
        if args.only and run_name not in args.only.split(","):
            continue

        torch.manual_seed(1337)
        if sp["block"] == "ffn":
            cfg = GPTConfig(mlp_type="ffn", bias=True)
        else:
            cfg = GPTConfig(mlp_type="fng", fng_n_ladders=sp["L"], fng_depth=sp["d"], bias=True)
        model = GPT(cfg).to(device)
        if sp["block"] != "ffn":
            calibrate_fng_init(model)
        model = model.to(dtype)

        if args.optimizer == "adamw":
            opt = model.configure_optimizers(
                weight_decay=0.1, learning_rate=sp["lr"], betas=(0.9, 0.95), device_type="cuda"
            )
            opt_args = {
                "lr": sp["lr"],
                "opt": opt,
                "set_lr": lambda lr: [g.__setitem__("lr", lr) for g in opt.param_groups],
            }
        else:
            opt = MuonAdamW(model, muon_lr=args.muon_lr, adam_lr=6e-4, weight_decay=0.0, adam_weight_decay=0.1)
            opt_args = {
                "lr": args.muon_lr,
                "opt": opt,
                "set_lr": lambda lr: opt.set_lr(lr, 6e-4 * (lr / args.muon_lr)),
            }

        summ = run_one(run_name, os.path.join(args.exp_dir, run_name), model, data, opt_args,
                       sp["max_steps"], args.target_ppl, dtype)
        summ["optimizer"] = args.optimizer
        with open(os.path.join(args.exp_dir, run_name, "summary.json"), "w") as f:
            json.dump(summ, f, indent=2)
        render_all(args.exp_dir, args.render_dir, args.target_ppl)

    render_all(args.exp_dir, args.render_dir, args.target_ppl)


if __name__ == "__main__":
    main()
