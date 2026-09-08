"""Chapter 10 solution: generate text from a saved checkpoint.

    python3 solutions/10-tiny-gpt/generate.py
    python3 solutions/10-tiny-gpt/generate.py --prompt "ROMEO:" --temperature 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import TinyTransformer
from tokenizer import CharTokenizer

CHECKPOINT = Path(__file__).parent / "checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="<1 = more predictable, >1 = more random",
    )
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT}. Run train.py first.")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    # Rebuild the tokenizer from the saved vocabulary. Rebuilding it from the
    # corpus would be a bug waiting to happen: a different corpus gives a
    # different id ordering, and the checkpoint's embeddings would be
    # silently mismatched.
    tokenizer = CharTokenizer("")
    tokenizer.chars = ckpt["chars"]
    tokenizer.vocab_size = len(ckpt["chars"])
    tokenizer._stoi = {ch: i for i, ch in enumerate(ckpt["chars"])}
    tokenizer._itos = {i: ch for ch, i in tokenizer._stoi.items()}

    model = TinyTransformer(vocab_size=tokenizer.vocab_size, **ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    losses = ckpt.get("losses", {})
    if losses:
        print(f"Checkpoint: train {losses['train']:.4f} / val {losses['val']:.4f}")
    print(f"Prompt: {args.prompt!r}   temperature {args.temperature}\n")

    unknown = set(args.prompt) - set(tokenizer.chars)
    if unknown:
        raise SystemExit(f"Prompt has characters not in the vocabulary: {unknown}")

    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    ids = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature)

    print("-" * 60)
    print(tokenizer.decode(ids[0].tolist()))
    print("-" * 60)


if __name__ == "__main__":
    main()
