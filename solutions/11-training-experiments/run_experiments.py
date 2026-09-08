"""Chapter 11 solution: controlled hyperparameter experiments.

Changes ONE variable at a time against a fixed baseline and tabulates the
result. Everything else -- seed, data, step count -- is held constant, or the
comparison means nothing.

    python3 solutions/11-training-experiments/run_experiments.py --steps 300
    python3 solutions/11-training-experiments/run_experiments.py --sweep lr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "solutions" / "10-tiny-gpt"))

from dataset import get_batch, load_corpus, make_splits  # noqa: E402
from model import TinyTransformer  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

CORPUS = REPO_ROOT / "sample-data" / "shakespeare.txt"

BASELINE = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "block_size": 64,
    "n_embed": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
}

SWEEPS = {
    "lr": ("learning_rate", [1e-5, 1e-4, 3e-4, 1e-3, 1e-2]),
    "batch": ("batch_size", [8, 16, 32, 64]),
    "block": ("block_size", [16, 32, 64, 128]),
    "embed": ("n_embed", [32, 64, 128, 256]),
    "layers": ("n_layers", [1, 2, 3, 4]),
}


@torch.no_grad()
def evaluate(model, data, config, device, iters=40) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, config["block_size"], config["batch_size"], device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def run_one(config: dict, train_data, val_data, vocab_size: int, steps: int, seed: int) -> dict:
    """Train one configuration from scratch with a fixed seed."""
    torch.manual_seed(seed)          # same init and same batches every run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformer(
        vocab_size=vocab_size,
        n_embed=config["n_embed"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        block_size=config["block_size"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    start = time.time()

    diverged = False
    for _ in range(steps):
        x, y = get_batch(train_data, config["block_size"], config["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            diverged = True
            break

    runtime = time.time() - start
    if diverged:
        return {
            "params": sum(p.numel() for p in model.parameters()),
            "train": float("nan"),
            "val": float("nan"),
            "runtime": runtime,
            "diverged": True,
        }

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "train": evaluate(model, train_data, config, device),
        "val": evaluate(model, val_data, config, device),
        "runtime": runtime,
        "diverged": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--sweep", choices=[*SWEEPS, "all"], default="lr", help="which variable to vary"
    )
    args = parser.parse_args()

    text = load_corpus(CORPUS)
    tokenizer = CharTokenizer(text)
    train_data, val_data = make_splits(text, tokenizer)

    sweeps = list(SWEEPS) if args.sweep == "all" else [args.sweep]

    print(f"Corpus {len(text):,} chars | {args.steps} steps/run | seed {args.seed}")
    print("Everything except the swept variable is held constant.\n")

    for sweep_name in sweeps:
        key, values = SWEEPS[sweep_name]
        print("=" * 74)
        print(f"SWEEP: {key}   (baseline {BASELINE[key]})")
        print("=" * 74)
        print(f"{key:>12} {'params':>10} {'train':>8} {'val':>8} {'gap':>8} {'time':>8}")
        print("-" * 74)

        results = []
        for value in values:
            config = {**BASELINE, key: value}
            r = run_one(config, train_data, val_data, tokenizer.vocab_size, args.steps, args.seed)
            marker = " <- baseline" if value == BASELINE[key] else ""

            if r["diverged"]:
                print(f"{value:>12} {r['params']:>10,} {'nan':>8} {'nan':>8} {'--':>8} "
                      f"{r['runtime']:>7.1f}s  DIVERGED{marker}")
            else:
                gap = r["val"] - r["train"]
                print(f"{value:>12} {r['params']:>10,} {r['train']:>8.4f} {r['val']:>8.4f} "
                      f"{gap:>+8.4f} {r['runtime']:>7.1f}s{marker}")
            results.append((value, r))

        finite = [(v, r) for v, r in results if not r["diverged"]]
        if finite:
            best = min(finite, key=lambda item: item[1]["val"])
            print(f"\n  Best validation loss: {key}={best[0]} ({best[1]['val']:.4f})")
        diverged = [v for v, r in results if r["diverged"]]
        if diverged:
            print(f"  Diverged (loss -> nan): {key}={diverged}")
        print()

    print("=" * 74)
    print("How to read these")
    print("=" * 74)
    print(
        "  LEARNING RATE has an optimum, not a direction. Too low and the model\n"
        "  barely moves in the step budget; too high and it overshoots or\n"
        "  diverges to nan. This is the single most important hyperparameter.\n\n"
        "  MODEL SIZE lowers training loss almost monotonically. Watch the GAP\n"
        "  column instead -- when validation stops improving while training keeps\n"
        "  falling, the extra capacity is going into memorisation.\n\n"
        "  BATCH SIZE mostly trades noise for speed. Larger batches give smoother\n"
        "  gradients per step but fewer steps for the same compute; the noise in\n"
        "  small batches is not purely harmful, it acts as regularisation.\n\n"
        "  CONTEXT LENGTH helps until the model has more context than it can use.\n\n"
        "  Record what you expected BEFORE each run, in experiments.md. Being\n"
        "  wrong on paper is how the intuition actually forms."
    )


if __name__ == "__main__":
    main()
