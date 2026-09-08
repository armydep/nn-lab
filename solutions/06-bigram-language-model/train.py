"""Chapter 6 solution: a bigram character language model.

NumPy, not PyTorch -- the point is to implement softmax and cross-entropy
yourself and confirm dL/dlogits = p - onehot(y).

    python3 solutions/06-bigram-language-model/train.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)

CORPUS = Path(__file__).resolve().parents[2] / "sample-data" / "tiny.txt"
LEARNING_RATE = 10.0        # large LR is fine: one linear layer, tiny data
EPOCHS = 400


# --- Stable softmax / log-softmax --------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Subtracting the max is free and prevents exp() overflow.

    The constant cancels between numerator and denominator, so the result is
    unchanged -- only the floating-point range changes. See NUMERICS.md.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(x)) without ever forming a probability that could underflow."""
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean -log(probability assigned to the correct class)."""
    return float(-np.mean(log_softmax(logits)[np.arange(len(targets)), targets]))


def main() -> None:
    text = CORPUS.read_text()

    # --- Vocabulary ----------------------------------------------------
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    print(f"Corpus: {len(text)} characters, vocabulary of {vocab_size}")
    print(f"Vocab: {''.join(repr(c)[1:-1] for c in chars)}")
    assert decode(encode("hello")) == "hello", "encode/decode must round-trip"
    print("encode/decode round-trips\n")

    # --- Training pairs: (current char, next char) ----------------------
    ids = encode(text)
    xs = np.array(ids[:-1])
    ys = np.array(ids[1:])
    print(f"Training pairs: {len(xs)}\n")

    # --- The model: one logit row per input character -------------------
    # A bigram model is a lookup table. Row i holds the logits for "what
    # follows character i". No hidden layer, no nonlinearity.
    W = rng.normal(0, 0.01, size=(vocab_size, vocab_size))

    # --- The gradient ---------------------------------------------------
    # L = -log(softmax(logits)[correct])
    #
    # Working through the softmax derivative, almost everything cancels:
    #
    #   dL/dlogits = p - onehot(target)
    #
    # The same "prediction minus target" rule as Chapter 2's dL/dz = p - y,
    # generalised from 2 classes to vocab_size classes.
    def gradient(W: np.ndarray) -> np.ndarray:
        logits = W[xs]                                  # (N, vocab)
        p = softmax(logits)
        p[np.arange(len(ys)), ys] -= 1.0                # p - onehot(y)
        p /= len(ys)

        dW = np.zeros_like(W)
        # Rows repeat (a character appears many times), so accumulate.
        np.add.at(dW, xs, p)
        return dW

    # Verify against finite differences before training.
    analytic = gradient(W)
    eps = 1e-5
    numeric = np.zeros_like(W)
    for i in range(vocab_size):
        for j in range(vocab_size):
            original = W[i, j]
            W[i, j] = original + eps
            up = cross_entropy(W[xs], ys)
            W[i, j] = original - eps
            down = cross_entropy(W[xs], ys)
            W[i, j] = original
            numeric[i, j] = (up - down) / (2 * eps)

    max_diff = np.max(np.abs(analytic - numeric))
    print(f"Gradient check (p - onehot(y)): max diff {max_diff:.2e}")
    assert max_diff < 1e-6, "gradient derivation is wrong"
    print("  match\n")

    # --- Train ----------------------------------------------------------
    # A uniform model over `vocab_size` characters would have loss log(vocab).
    print(f"Baseline loss for a uniform model: {np.log(vocab_size):.4f}")
    print("Training:")
    for epoch in range(EPOCHS + 1):
        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}   loss {cross_entropy(W[xs], ys):.6f}")
        W -= LEARNING_RATE * gradient(W)

    final_loss = cross_entropy(W[xs], ys)
    print(f"\nFinal loss: {final_loss:.6f}")

    # --- Hand-count sanity check ----------------------------------------
    # A bigram model has a closed-form optimum: the empirical frequencies.
    # If the learned probabilities match counts, every component is correct.
    print("\nLearned probabilities vs hand-counted frequencies:")
    print(f"  {'bigram':<12} {'counted':>9} {'learned':>9}")
    checked = 0
    for ch in ["l", "e", "n", "h"]:
        if ch not in stoi:
            continue
        i = stoi[ch]
        following = [ys[k] for k in range(len(xs)) if xs[k] == i]
        if not following:
            continue
        counts = np.bincount(following, minlength=vocab_size)
        best = int(np.argmax(counts))
        empirical = counts[best] / counts.sum()
        learned = softmax(W[i])[best]
        label = f"{repr(ch)[1:-1]} -> {repr(itos[best])[1:-1]}"
        print(f"  {label:<12} {empirical:>9.4f} {learned:>9.4f}")
        assert abs(empirical - learned) < 0.05, f"bigram {label} did not converge"
        checked += 1
    print(f"  {checked} bigrams match their empirical frequencies\n")

    # --- Generate -------------------------------------------------------
    # Sample from the distribution. argmax would immediately enter a loop,
    # because the most likely successor of a character never changes.
    print("Generated text (sampled):")
    for trial in range(3):
        current = stoi["h"] if "h" in stoi else 0
        out = [current]
        for _ in range(60):
            probabilities = softmax(W[current])
            current = int(rng.choice(vocab_size, p=probabilities))
            out.append(current)
        print(f"  {trial + 1}: {decode(out)!r}")

    print(
        "\nA bigram model sees exactly one character of context, so it cannot\n"
        "produce real words -- only locally plausible letter transitions.\n"
        "Chapter 7 widens the context window, which is the next thing that has\n"
        "to change for the output to improve."
    )


if __name__ == "__main__":
    main()
