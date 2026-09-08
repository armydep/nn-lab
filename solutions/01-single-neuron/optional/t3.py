"""Chapter 1, Task 3 solution -- learning rate regimes."""

import numpy as np

X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = 2.0 * X + 1.0

EPOCHS = 300
RATES = [0.001, 0.01, 0.05, 0.1, 0.145, 0.15, 0.2]
RUNAWAY = 1e6


def forward(x, w, b):
    return w * x + b


def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)


def gradients(x, y, w, b):
    error = forward(x, w, b) - y
    return np.mean(2 * error * x), np.mean(2 * error)


def train(learning_rate, epochs=EPOCHS):
    """Returns (final_loss, w, b, diverged).

    Divergence is defined as the loss GROWING, not as it overflowing to inf.
    Overflow is just what happens eventually; growth is the actual failure,
    and it starts at a much lower learning rate.
    """
    w, b = 0.0, 0.0
    start_loss = mse(forward(X, w, b), Y)
    loss = start_loss

    # Diverging runs overflow float64, which is expected here -- the
    # warnings would otherwise bury the output.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            dw, db = gradients(X, Y, w, b)
            w -= learning_rate * dw
            b -= learning_rate * db
            loss = mse(forward(X, w, b), Y)

            # Abandon a dead run immediately rather than burning compute.
            if not np.isfinite(loss) or loss > RUNAWAY:
                return loss, w, b, True

    return loss, w, b, loss > start_loss


if __name__ == "__main__":
    start_loss = mse(forward(X, 0.0, 0.0), Y)
    print(f"Starting loss (w=0, b=0): {start_loss:.4f}\n")
    print(f"{'lr':>8} {'final loss':>14} {'w':>9} {'b':>9}   verdict")
    print("-" * 62)

    for lr in RATES:
        loss, w, b, diverged = train(lr)
        if diverged:
            print(f"{lr:>8} {'exploded':>14} {'--':>9} {'--':>9}   DIVERGED")
        else:
            verdict = "too slow" if loss > 1.0 else ("learning" if loss > 0.01 else "converged")
            print(f"{lr:>8} {loss:>14.6f} {w:>+9.4f} {b:>+9.4f}   {verdict}")

    # Locate the cliff by bisection on "does the loss grow?".
    low, high = 0.05, 0.5
    for _ in range(50):
        mid = (low + high) / 2
        if train(mid)[3]:
            high = mid
        else:
            low = mid
    print(f"\nDivergence threshold, by bisection: lr = {low:.6f}")

    # The theoretical value, for comparison.
    hessian = 2 * np.array([[np.mean(X**2), np.mean(X)], [np.mean(X), 1.0]])
    lambda_max = np.max(np.linalg.eigvalsh(hessian))
    print(f"Theory (2 / largest curvature) =     {2 / lambda_max:.6f}")
    print(
        "\n  These agree, and the value is not arbitrary. For mean squared\n"
        "  error the cliff sits at exactly 2 / (largest eigenvalue of the\n"
        "  loss curvature). Past it, each step overshoots the minimum by\n"
        "  more than it began away from it, so the error grows every step."
    )

    print("\nWatching a diverging run (lr = 0.2), first 8 steps:")
    w, b = 0.0, 0.0
    for step in range(8):
        dw, db = gradients(X, Y, w, b)
        w -= 0.2 * dw
        b -= 0.2 * db
        print(f"  step {step + 1}:  w = {w:+12.2f}   b = {b:+10.2f}   loss = {mse(forward(X, w, b), Y):.3e}")
    print(
        "\n  The sign flips every step while the magnitude grows. That is what\n"
        "  divergence IS -- overshooting so hard it lands further out on the\n"
        "  other side, then further still. Note it is already hopeless by\n"
        "  step 2; the overflow much later is only the bookkeeping catching up."
    )

    print("\nDoes lr = 0.001 eventually converge?")
    for epochs in (300, 3000, 30000):
        loss, w, b, _ = train(0.001, epochs=epochs)
        print(f"  {epochs:>6} epochs:  loss {loss:.8f}   w {w:.4f}   b {b:.4f}")
    print(
        "\n  Yes. A small learning rate is not wrong, only slow -- roughly 100x\n"
        "  the steps to reach what lr=0.05 reached in 300. Compute is the price\n"
        "  of caution. A rate past the cliff is a different matter entirely:\n"
        "  no number of steps will rescue it."
    )
