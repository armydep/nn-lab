"""Chapter 2, Task 3 solution -- the log(0) trap."""

import numpy as np

X = np.array(
    [
        [0.5, 0.5], [1.0, 0.8], [0.7, 1.1], [1.2, 0.6],
        [6.0, 6.2], [6.5, 5.8], [5.9, 6.6], [6.3, 6.1],
    ]
)
Y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

EPS = 1e-12
LEARNING_RATE = 3.0
EPOCHS = 60


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce_naive(p, y):
    """The formula exactly as written. Correct maths, unsafe in float64."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def bce_safe(p, y):
    """Identical, except no log ever receives exactly 0."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def forward(x, w, b):
    return sigmoid(x @ w + b)


def gradients(x, y, w, b):
    dz = forward(x, w, b) - y
    return x.T @ dz / len(y), np.mean(dz)


def train(loss_fn, label):
    w, b = np.zeros(2), 0.0
    print(f"\n{label}")
    print(f"  {'epoch':>6} {'loss':>14} {'min p':>12} {'1 - max p':>12}")
    print("  " + "-" * 50)

    first_bad = None
    for epoch in range(EPOCHS + 1):
        p = forward(X, w, b)
        loss = loss_fn(p, Y)

        if epoch % 10 == 0 or (not np.isfinite(loss) and first_bad is None):
            text = f"{loss:.8f}" if np.isfinite(loss) else "     nan/inf"
            print(f"  {epoch:>6} {text:>14} {p.min():>12.2e} {1 - p.max():>12.2e}")

        if not np.isfinite(loss) and first_bad is None:
            first_bad = epoch
            print(f"  ^^^ loss stopped being a number at epoch {epoch}")

        dw, db = gradients(X, Y, w, b)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    return first_bad


if __name__ == "__main__":
    ordinary = np.array([0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8, 0.1])
    print(f"On ordinary probabilities: naive {bce_naive(ordinary, Y):.6f}   "
          f"safe {bce_safe(ordinary, Y):.6f}")
    assert np.isclose(bce_naive(ordinary, Y), bce_safe(ordinary, Y))
    print("  Identical. Clipping is invisible until it matters.\n")

    print("The one line responsible for all of it:")
    with np.errstate(divide="ignore"):
        print(f"  np.log(0.0) = {np.log(0.0)}")

    bad_naive = train(bce_naive, "=== NAIVE (no clipping) ===")
    bad_safe = train(bce_safe, "=== SAFE (clipped) ===")

    print("\n" + "=" * 58)
    print(f"Naive broke at epoch {bad_naive}.  Safe broke: {bad_safe}.")
    assert bad_naive is not None, "naive should have produced nan"
    assert bad_safe is None, "safe should never break"
    print("=" * 58)

    print("\nAnswers:\n")

    print("  1. Does a smaller learning rate hide the bug?")
    for lr in (0.1, 1.0, 3.0):
        w, b = np.zeros(2), 0.0
        broke = None
        for epoch in range(EPOCHS + 1):
            if not np.isfinite(bce_naive(forward(X, w, b), Y)) and broke is None:
                broke = epoch
            dw, db = gradients(X, Y, w, b)
            w -= lr * dw
            b -= lr * db
        print(f"     lr={lr}: naive breaks at epoch {broke}")
    print(
        "     Yes -- and that is what makes this bug nasty. It hides at small\n"
        "     learning rates and appears the moment you tune upward, so it\n"
        "     looks like the tuning broke things rather than the loss.\n"
    )

    print("  2. Does nan ever recover?")
    print("     It depends entirely on WHERE the nan is, and the run above\n"
          "     shows the surprising case: the naive loss was nan at epoch 1\n"
          "     and perfectly healthy by epoch 10. Training never noticed.\n")
    print("     That happened because the WEIGHTS never went nan. Gradients\n"
          "     here are p - y, which takes no log, so the poison stayed in\n"
          "     the reported number and never reached the parameters.\n")
    print("     Contrast that with nan actually reaching the weights:")
    w, b = np.array([np.nan, 0.0]), 0.0
    for epoch in range(3):
        dw, db = gradients(X, Y, w, b)
        w = w - 1.0 * dw
        b = b - 1.0 * db
        print(f"       epoch {epoch}: w = {w},  loss = {bce_safe(forward(X, w, b), Y)}")
    print("     Never recovers, and clipping cannot help -- nan * anything is\n"
          "     nan, so it spreads to every parameter and stays. THAT is the\n"
          "     case worth detecting early and abandoning.\n")
    print("     Practical rule: a nan loss is a symptom, not a diagnosis.\n"
          "     Check whether your weights are still finite before concluding\n"
          "     the run is dead.\n")

    print("  3. Is EPS = 1e-300 still safe?")
    tiny = np.clip(np.array([0.0]), 1e-300, 1.0 - 1e-300)
    print(f"     clip(0.0, 1e-300, ...) = {tiny[0]:.1e}, log = {np.log(tiny)[0]:.2f}")
    print(f"     But: 1.0 - 1e-300 == 1.0 is {1.0 - 1e-300 == 1.0} in float64,")
    print("     so the UPPER clip silently does nothing and log(1-p) still\n"
          "     hits log(0). The lower bound works, the upper one does not.\n"
          "     Clip values must be large enough to survive the arithmetic;\n"
          "     1e-12 is the usual safe choice.\n")

    print("  4. Why can a model train while its reported loss is nan?")
    p_saturated = np.array([1.0, 0.0])
    y_example = np.array([0.0, 1.0])
    print(f"     Loss at p={p_saturated.tolist()}: {bce_naive(p_saturated, y_example)}")
    print(f"     Gradient dL/dz = p - y = {(p_saturated - y_example).tolist()}")
    print(
        "     The gradient is perfectly finite. Only the LOSS VALUE overflows,\n"
        "     because it takes a log and the gradient does not. So a model\n"
        "     using the fused form can keep learning while its printed loss is\n"
        "     nan -- and conversely, a nan loss does not always mean the model\n"
        "     is dead. Check which quantity actually broke before panicking."
    )
