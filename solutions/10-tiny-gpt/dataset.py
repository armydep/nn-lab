"""Data pipeline: text -> train/validation splits -> batches."""

from __future__ import annotations

from pathlib import Path

import torch

from tokenizer import CharTokenizer


def load_corpus(path: Path) -> str:
    if not path.exists():
        raise SystemExit(
            f"Corpus not found: {path}\n"
            "Run:  python3 sample-data/download_corpus.py"
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 100_000:
        print(
            f"WARNING: corpus is only {len(text):,} characters. A Transformer\n"
            "will memorise this. See sample-data/README.md.\n"
        )
    return text


def make_splits(
    text: str, tokenizer: CharTokenizer, train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split BEFORE batching, and split by position, not randomly.

    Random splitting would leak: adjacent windows overlap almost entirely, so
    a random validation window is nearly identical to a training one, and
    validation loss would flatter the model.
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(train_fraction * len(data))
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows.

    Targets are inputs shifted by one: predicting the next character at every
    position at once. One window of length T gives T training signals, not 1.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
