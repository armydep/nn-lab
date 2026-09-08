"""Chapter 1, Task 3 -- learning rate: the parameter that decides everything.

Same problem as Task 1 (y = 2x + 1). Nothing about the model changes.
The ONLY thing you vary is how big a step you take.

You will see all three regimes: too small to get anywhere, about right,
and large enough to explode.

    python3 01-single-neuron/optional/t3.py
"""

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


# ----------------------------------------------------------------------
# YOUR TURN -- write the training loop as a reusable function.
#
# Run `epochs` steps of gradient descent from w=0, b=0, and return
# (final_loss, w, b, diverged).
#
# On that last value: divergence means the loss GREW, not that it became
# inf. A run whose loss climbs from 33 to 17,000 has already failed --
# waiting for it to overflow is just waiting for bookkeeping. So:
#
#   - bail out early if the loss stops being finite OR exceeds RUNAWAY
#   - otherwise, report diverged = (final loss > starting loss)
#
# Getting this distinction right is the point of the task. A detector
# that only checks isfinite() will call a badly diverging run "slow".
#
# One practical note: diverging runs overflow float64 and numpy prints
# warnings. Wrap your loop in:
#
#   with np.errstate(over="ignore", invalid="ignore"):
#
# so the output stays readable.
# ----------------------------------------------------------------------
def train(learning_rate, epochs=EPOCHS):
    w, b = 0.0, 0.0
    # TODO: record the starting loss
    # TODO: loop, computing gradients and updating w and b
    # TODO: bail out early on non-finite or runaway loss
    # return (loss, w, b, diverged)
    pass  # <- replace


if __name__ == "__main__":
    if train(0.05) is None:
        raise SystemExit("Write train() first.")

    start_loss = mse(forward(X, 0.0, 0.0), Y)
    print(f"Fitting y = 2x + 1, {EPOCHS} epochs per run. Target: w=2, b=1")
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

    print(
        "\nThe pattern: raising the learning rate helps, up to a point -- then\n"
        "it stops helping, then it explodes. There is an optimum, and it is\n"
        "not 'as large as possible'.\n\n"
        "Look at where the cliff falls. Between 0.145 and 0.15 the behaviour\n"
        "changes completely. That edge is not luck: for mean squared error it\n"
        "sits at exactly 2 / (largest curvature of the loss). Exercise 1 has\n"
        "you find it, and the solution shows it matching theory to 6 decimals.\n\n"
        "Real models have a cliff too -- you just cannot compute where it is,\n"
        "which is why learning rate is the first thing anyone tunes and why\n"
        "Chapter 11 sweeps it before anything else."
    )

# ----------------------------------------------------------------------
# Definition of done
#
#   You see all three regimes, and your detector correctly labels lr=0.15
#   as DIVERGED rather than "slow".
#
# Then explore:
#
#   1. Find the cliff by bisection: repeatedly halve the interval between
#      a rate that converges and one that diverges. You should land near
#      0.1492. Compare it to 2 / largest eigenvalue of
#      2 * [[mean(x^2), mean(x)], [mean(x), 1]]  -- they match exactly.
#
#   2. At lr = 0.2, print w and b every step for the first 8 steps. Watch
#      the sign FLIP each step while the magnitude grows. That is what
#      divergence is: overshooting so badly it lands further out on the
#      other side, every single time.
#
#   3. Run lr = 0.001 for 30000 epochs. Does it converge eventually? What
#      does that say about whether a small learning rate is WRONG, versus
#      merely expensive?
#
#   4. Read NUMERICS.md. Loss going bad after a few hundred steps usually
#      means exploding gradients, like this. Loss going bad instantly
#      usually means a domain error like log(0) -- which is exactly what
#      Chapter 2's Task 3 has you produce on purpose.
# ----------------------------------------------------------------------
