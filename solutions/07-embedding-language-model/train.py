"""Chapter 7 solution: embeddings + MLP language model.

Widens the context from 1 character (Chapter 6) to BLOCK_SIZE characters, and
replaces the lookup table with learned vectors.

    python3 solutions/07-embedding-language-model/train.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

CORPUS = Path(__file__).resolve().parents[2] / "sample-data" / "tiny.txt"
BLOCK_SIZE = 4        # characters of context
EMBED_DIM = 16
HIDDEN = 64
STEPS = 3000
LEARNING_RATE = 0.01


class EmbeddingLM(nn.Module):
    """Embed each context character, concatenate, feed through an MLP."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # The embedding table IS a lookup table -- token id -> vector. It is
        # trained by gradient descent like any other parameter, so "meaning"
        # here is simply whatever makes next-character prediction work.
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(BLOCK_SIZE * EMBED_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, vocab_size),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, BLOCK_SIZE) -> emb: (B, BLOCK_SIZE, EMBED_DIM)
        emb = self.embedding(idx)
        # Flatten the context into one vector. Note this is order-aware:
        # position 0 and position 3 land in different input slots, so the
        # model can tell them apart. Chapter 9 replaces this crude approach
        # with positional embeddings.
        return self.net(emb.view(emb.shape[0], -1))


def build_dataset(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding windows of BLOCK_SIZE characters -> the character that follows."""
    xs, ys = [], []
    for i in range(len(ids) - BLOCK_SIZE):
        xs.append(ids[i : i + BLOCK_SIZE])
        ys.append(ids[i + BLOCK_SIZE])
    return torch.tensor(xs), torch.tensor(ys)


def main() -> None:
    text = CORPUS.read_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    ids = [stoi[c] for c in text]
    X, Y = build_dataset(ids)
    print(f"Corpus: {len(text)} chars, vocab {vocab_size}")
    print(f"Context window: {BLOCK_SIZE} characters")
    print(f"Examples: {len(X)}\n")

    model = EmbeddingLM(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Uniform-model baseline loss: {torch.log(torch.tensor(float(vocab_size))):.4f}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training:")
    for step in range(STEPS + 1):
        logits = model(X)
        # F.cross_entropy applies log_softmax internally, in the numerically
        # stable form you wrote by hand in Chapter 6.
        loss = F.cross_entropy(logits, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  step {step:5d}   loss {loss.item():.6f}")

    final_loss = loss.item()
    print(f"\nFinal loss: {final_loss:.6f}")
    print("  (Chapter 6's bigram reached ~1.57 on this corpus with 1 char of")
    print("   context. More context -> lower loss, on the same data.)")

    if final_loss < 0.5:
        print(
            "\n  *** LOOK AT HOW LOW THAT LOSS IS. ***\n"
            "  This is not a better model -- it is MEMORISATION. With 144\n"
            "  characters and thousands of parameters, the model can store the\n"
            "  corpus outright. The generated text below will echo the training\n"
            "  data almost verbatim.\n\n"
            "  This is the exact failure mode sample-data/README.md warns about,\n"
            "  and it is worth seeing once, here, where the corpus is small enough\n"
            "  to recognise it instantly. In Chapter 10 the same thing happens\n"
            "  invisibly unless you hold out a validation split.\n"
        )
    else:
        print()

    # --- Generate -------------------------------------------------------
    print("Generated text (sampled):")
    model.eval()
    with torch.no_grad():
        for trial in range(3):
            context = [stoi[c] for c in "hell"[:BLOCK_SIZE]]
            out = list(context)
            for _ in range(60):
                logits = model(torch.tensor([context[-BLOCK_SIZE:]]))
                probabilities = F.softmax(logits[0], dim=-1)
                nxt = int(torch.multinomial(probabilities, num_samples=1))
                out.append(nxt)
                context.append(nxt)
            print(f"  {trial + 1}: {''.join(itos[i] for i in out)!r}")

    # --- What the embeddings learned ------------------------------------
    print("\nNearest neighbours in embedding space (cosine similarity):")
    with torch.no_grad():
        emb = model.embedding.weight
        normalised = emb / emb.norm(dim=1, keepdim=True)
        similarity = normalised @ normalised.T
        similarity.fill_diagonal_(-2.0)
        for ch in ["a", "e", "o"]:
            if ch not in stoi:
                continue
            i = stoi[ch]
            top = int(similarity[i].argmax())
            print(
                f"  {repr(ch)} is closest to {repr(itos[top])}  "
                f"(similarity {similarity[i][top]:.3f})"
            )
    print(
        "\n  Nothing told the model that vowels are related. Any structure here\n"
        "  emerged purely from characters being interchangeable for the task of\n"
        "  predicting what comes next. That is what 'learned representation'\n"
        "  means -- and on a 144-character corpus, expect it to be weak."
    )


if __name__ == "__main__":
    main()
