"""Chapter 10 solution: train the tiny GPT.

    python3 solutions/10-tiny-gpt/train.py
    python3 solutions/10-tiny-gpt/train.py --steps 5000    # better output

Run sample-data/download_corpus.py first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dataset import get_batch, load_corpus, make_splits
from model import TinyTransformer
from tokenizer import CharTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "sample-data" / "shakespeare.txt"
CHECKPOINT = Path(__file__).parent / "checkpoint.pt"

# Small enough to train on a laptop CPU in a couple of minutes.
BLOCK_SIZE = 64
BATCH_SIZE = 32
N_EMBED = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 250
EVAL_ITERS = 50


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Average loss over several batches.

    A single batch is far too noisy to judge progress by -- you would see
    random fluctuation and read it as a trend.
    """
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(CORPUS)
    tokenizer = CharTokenizer(text)
    train_data, val_data = make_splits(text, tokenizer)

    print(f"Device:     {device}")
    print(f"Corpus:     {len(text):,} characters, vocab {tokenizer.vocab_size}")
    print(f"Train/val:  {len(train_data):,} / {len(val_data):,} tokens")

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Steps:      {args.steps}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"{'step':>6}  {'train':>8}  {'val':>8}  {'gap':>7}  {'elapsed':>8}")
    print("-" * 46)
    start = time.time()
    for step in range(args.steps + 1):
        if step % EVAL_INTERVAL == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, device)
            gap = losses["val"] - losses["train"]
            print(
                f"{step:>6}  {losses['train']:>8.4f}  {losses['val']:>8.4f}  "
                f"{gap:>+7.4f}  {time.time() - start:>7.1f}s"
            )

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model, train_data, val_data, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": tokenizer.chars,
            "config": {
                "n_embed": N_EMBED,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "block_size": BLOCK_SIZE,
                "dropout": DROPOUT,
            },
            "losses": losses,
        },
        CHECKPOINT,
    )
    print(f"\nSaved checkpoint: {CHECKPOINT}")

    gap = losses["val"] - losses["train"]
    print(f"\nFinal train {losses['train']:.4f} / val {losses['val']:.4f} (gap {gap:+.4f})")
    if gap > 0.15:
        print("  Validation loss is well above training loss -- overfitting.")
        print("  More data, more dropout, or a smaller model would help.")
    else:
        print("  Train and validation are close -- the model is still learning")
        print("  general structure rather than memorising. You could train longer.")

    print("\nSample:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    ids = model.generate(context, max_new_tokens=400)[0].tolist()
    print("-" * 46)
    print(tokenizer.decode(ids))
    print("-" * 46)


if __name__ == "__main__":
    main()
