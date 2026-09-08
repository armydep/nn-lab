"""Chapter 9 solution: a tiny Transformer language model.

Every component traces back to an earlier chapter:
  - Embeddings          -> Chapter 7
  - Causal attention    -> Chapter 8
  - Feed-forward MLP    -> Chapter 4/5
  - Cross-entropy       -> Chapter 6

New here: multi-head attention, residual connections, layer normalisation.

    python3 solutions/09-tiny-transformer/model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One causal self-attention head -- Chapter 8, batched."""

    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # A buffer is state that is not a parameter: saved with the model,
        # moved across devices, but never receives a gradient.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)                                     # (B, T, head)
        q = self.query(x)                                   # (B, T, head)

        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # Slice the mask to T: the last batch of a corpus may be shorter.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))

        return weights @ self.value(x)                      # (B, T, head)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated.

    One head learns one notion of "what to look at". Multiple heads let the
    model track several relationships at once -- one might follow syntax,
    another recent tokens -- and the projection mixes their findings.
    """

    def __init__(self, n_heads: int, n_embed: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)
        )
        self.proj = nn.Linear(head_size * n_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP -- the Chapter 4/5 network, applied to each position.

    Attention moves information BETWEEN positions; this layer does the
    thinking WITHIN a position. The 4x expansion is the standard ratio.
    """

    def __init__(self, n_embed: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: communicate, then compute."""

    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, n_embed, block_size, dropout)
        self.feed_forward = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two design decisions worth understanding:
        #
        # RESIDUAL (x + ...): gradient flows straight through the addition to
        # earlier layers. Without it, deep stacks train very poorly -- the
        # gradient signal degrades as it passes through each layer.
        #
        # PRE-NORM (ln applied BEFORE the sublayer, not after): keeps the
        # residual path clean and makes training far more stable. The original
        # 2017 paper used post-norm; modern GPTs use pre-norm.
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """A miniature GPT."""

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        block_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        # Attention is permutation-invariant -- it has no inherent notion of
        # order. Without positional embeddings, "cat sat" and "sat cat" would
        # produce identical representations.
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        token_emb = self.token_embedding(idx)                              # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = token_emb + pos_emb                                            # broadcast
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)                                           # (B,T,vocab)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # Flatten to (B*T, V) -- cross_entropy wants 2D logits.
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Autoregressive sampling: predict, append, repeat."""
        for _ in range(max_new_tokens):
            # Never feed more than block_size -- positional embeddings only
            # exist for that many positions.
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            # Only the LAST position predicts the next token.
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def main() -> None:
    torch.manual_seed(0)

    vocab_size, block_size, batch_size = 65, 32, 4
    model = TinyTransformer(vocab_size=vocab_size, block_size=block_size)

    print("=" * 62)
    print("Shape check")
    print("=" * 62)
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(idx, targets)

    print(f"  input   {tuple(idx.shape)}   (batch, time)")
    print(f"  logits  {tuple(logits.shape)}   (batch, time, vocab)")
    assert logits.shape == (batch_size, block_size, vocab_size)
    print("  matches the definition of done: [batch, time, vocab]")

    print(f"\n  loss at init: {loss.item():.4f}")
    expected = torch.log(torch.tensor(float(vocab_size)))
    print(f"  -log(1/{vocab_size}) = {expected:.4f}")
    print(
        "\n  These should be close. An untrained model is near-uniform over the\n"
        "  vocabulary, so its loss is log(vocab_size). If your loss at init is\n"
        "  far from this, something is wrong BEFORE you start training --\n"
        "  the fastest sanity check there is."
    )
    assert abs(loss.item() - expected.item()) < 0.5, "init loss is suspicious"

    print("\n" + "=" * 62)
    print("Parameter count")
    print("=" * 62)
    total = sum(p.numel() for p in model.parameters())
    for name, module in [
        ("token embedding", model.token_embedding),
        ("position embedding", model.position_embedding),
        ("blocks", model.blocks),
        ("lm_head", model.lm_head),
    ]:
        count = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {count:>8,}  ({100 * count / total:4.1f}%)")
    print(f"  {'TOTAL':<20} {total:>8,}")

    print("\n" + "=" * 62)
    print("Generation runs")
    print("=" * 62)
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=20)
    print(f"  generated shape {tuple(generated.shape)} (1 seed + 20 new)")
    assert generated.shape == (1, 21)
    print("  Untrained, so the ids are random -- Chapter 10 trains it on real text.")


if __name__ == "__main__":
    main()
