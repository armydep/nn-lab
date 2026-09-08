"""Chapter 8 solution: causal self-attention from first principles.

Deliberately standalone -- no model, no training. Just the calculation, with
the attention matrix printed at every stage so you can see what it does.

    python3 solutions/08-attention/attention.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.manual_seed(0)

T = 5      # sequence length (time steps)
C = 8      # embedding dimension
HEAD = 4   # attention head size

TOKENS = ["the", "cat", "sat", "on", "mat"]


def show(matrix: torch.Tensor, title: str, labels: list[str]) -> None:
    print(f"\n{title}")
    print("        " + "".join(f"{label:>8}" for label in labels))
    for i, row in enumerate(matrix):
        cells = "".join(
            "    -inf" if torch.isinf(v) and v < 0 else f"{v:8.3f}" for v in row
        )
        print(f"  {labels[i]:>5} {cells}")


def main() -> None:
    x = torch.randn(T, C)
    print(f"Input: {T} tokens, {C} dims each -> shape {tuple(x.shape)}")
    print(f"Tokens: {TOKENS}")

    # --- Q, K, V --------------------------------------------------------
    # Three different learned views of the same token:
    #   Query -- "what am I looking for?"
    #   Key   -- "what do I offer?"
    #   Value -- "what do I actually contribute?"
    query_proj = torch.nn.Linear(C, HEAD, bias=False)
    key_proj = torch.nn.Linear(C, HEAD, bias=False)
    value_proj = torch.nn.Linear(C, HEAD, bias=False)

    q = query_proj(x)      # (T, HEAD)
    k = key_proj(x)        # (T, HEAD)
    v = value_proj(x)      # (T, HEAD)
    print(f"Q, K, V shapes: {tuple(q.shape)}")

    # --- Scores ---------------------------------------------------------
    # scores[i][j] = how much token i wants to attend to token j.
    scores = q @ k.T                                    # (T, T)
    show(scores, "1. Raw scores (Q @ K.T)", TOKENS)

    # --- Scale ----------------------------------------------------------
    # Without this, dot products grow with HEAD, softmax saturates towards
    # one-hot, and gradients vanish. Dividing by sqrt(HEAD) keeps the
    # variance of the scores at roughly 1 regardless of head size.
    scaled = scores / (HEAD**0.5)
    show(scaled, f"2. Scaled (/ sqrt({HEAD}))", TOKENS)

    # --- Causal mask ----------------------------------------------------
    # A language model predicts the next token. If position i could see
    # position j > i, it would be reading the answer -- training loss would
    # collapse and generation would produce nonsense, because at inference
    # the future does not exist yet.
    mask = torch.tril(torch.ones(T, T))
    masked = scaled.masked_fill(mask == 0, float("-inf"))
    show(masked, "3. Causally masked (upper triangle -> -inf)", TOKENS)
    print(
        "\n  -inf, not a large negative number and not zero: exp(-inf) is\n"
        "  exactly 0, so masked positions get exactly zero weight. A large\n"
        "  finite value leaks a small amount of future information."
    )

    # --- Softmax --------------------------------------------------------
    weights = F.softmax(masked, dim=-1)
    show(weights, "4. Attention weights (softmax over each row)", TOKENS)

    row_sums = weights.sum(dim=-1)
    print(f"\n  Row sums: {[f'{s:.3f}' for s in row_sums]}")
    assert torch.allclose(row_sums, torch.ones(T), atol=1e-6)
    print("  Every row sums to 1 -- each row is a probability distribution")
    print("  describing how token i divides its attention over tokens 0..i.")

    upper = torch.triu(weights, diagonal=1)
    assert torch.all(upper == 0), "causal mask leaked"
    print("  Upper triangle is exactly zero -- no token sees the future.")
    print(f"  Token 0 must attend entirely to itself: {weights[0][0]:.3f}")

    # --- Weighted sum of values -----------------------------------------
    out = weights @ v                                   # (T, HEAD)
    print(f"\n5. Output = weights @ V -> shape {tuple(out.shape)}")
    print(
        "\n  Each output row is a weighted average of the VALUE vectors of\n"
        "  that token and everything before it. Token i's representation is\n"
        "  now built from its own context -- which is the entire point:\n"
        "  'sat' can mean something different depending on what preceded it."
    )

    # --- What a row means -----------------------------------------------
    print("\n" + "=" * 62)
    print("Reading the attention matrix")
    print("=" * 62)
    for i in range(T):
        contributions = [
            f"{TOKENS[j]} {weights[i][j]:.2f}" for j in range(i + 1) if weights[i][j] > 0.01
        ]
        print(f"  '{TOKENS[i]}' builds itself from: {', '.join(contributions)}")
    print(
        "\n  Row i answers: 'when computing the new representation of token i,\n"
        "  how much does each earlier token contribute?' The weights are not\n"
        "  meaningful yet -- these projections are random. After training they\n"
        "  become the model's learned notion of which context matters."
    )


if __name__ == "__main__":
    main()
