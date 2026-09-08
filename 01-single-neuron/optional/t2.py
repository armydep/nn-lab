"""Chapter 1, Task 2 -- a neuron with TWO inputs.

    y = 3*x1 - 2*x2 + 5

New idea: one gradient per weight. Note that w2's true value is NEGATIVE --
the model has to discover that x2 pushes the output DOWN.

This is the direct stepping stone to Chapter 2 (two inputs + sigmoid) and to
Chapter 4 (many inputs, as matrices).

    python3 01-single-neuron/optional/t2.py
"""

import numpy as np

# ----------------------------------------------------------------------
# Data. Six points. Column 0 is x1, column 1 is x2.
# ----------------------------------------------------------------------
X = np.array(
    [
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [3.0, 2.0],
        [2.0, 3.0],
        [4.0, 1.0],
    ]
)
Y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 5.0

LEARNING_RATE = 0.02
EPOCHS = 2000


# ----------------------------------------------------------------------
# YOUR TURN (1 of 3) -- forward pass
#
# Return w1*x1 + w2*x2 + b for every row at once.
# X[:, 0] is all the x1 values; X[:, 1] is all the x2 values.
# ----------------------------------------------------------------------
def forward(x, w1, w2, b):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 3) -- loss
#
# Same mean squared error as before.
# ----------------------------------------------------------------------
def mse(y_hat, y):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 3) -- the gradients
#
# Derive these on paper FIRST, then write them.
#
#   L = mean( (w1*x1 + w2*x2 + b - y)^2 )
#
# Start from what you know for the one-input case:
#     dL/dw = mean(2 * error * x)
#     dL/db = mean(2 * error)
#
# Ask yourself: when you differentiate with respect to w1, what does the
# w2*x2 term become? What multiplies the error for w1 versus for w2?
#
# Return the three gradients as a tuple (dw1, dw2, db).
# ----------------------------------------------------------------------
def gradients(x, y, w1, w2, b):
    pass  # <- replace, return (dw1, dw2, db)


# ----------------------------------------------------------------------
# A gift: finite-difference checking, taught properly in Chapter 3.
# Nudge a parameter slightly, see how much the loss moves. That ratio IS
# the gradient. If your formulas disagree with this, your formulas are
# wrong -- and you never have to guess about it again.
# ----------------------------------------------------------------------
def numeric_gradients(x, y, w1, w2, b, eps=1e-6):
    def loss_at(a, c, d):
        return mse(forward(x, a, c, d), y)

    dw1 = (loss_at(w1 + eps, w2, b) - loss_at(w1 - eps, w2, b)) / (2 * eps)
    dw2 = (loss_at(w1, w2 + eps, b) - loss_at(w1, w2 - eps, b)) / (2 * eps)
    db = (loss_at(w1, w2, b + eps) - loss_at(w1, w2, b - eps)) / (2 * eps)
    return dw1, dw2, db


if __name__ == "__main__":
    w1, w2, b = 0.0, 0.0, 0.0

    if forward(X, w1, w2, b) is None or mse(np.zeros(6), Y) is None:
        raise SystemExit("Write forward() and mse() first.")

    print(f"Starting loss: {mse(forward(X, w1, w2, b), Y):.4f}\n")

    if gradients(X, Y, w1, w2, b) is None:
        raise SystemExit("Now write gradients() -- derive it on paper first.")

    # Check the derivation before trusting it.
    analytic = gradients(X, Y, w1, w2, b)
    numeric = numeric_gradients(X, Y, w1, w2, b)
    print("Gradient check:")
    print(f"  yours:   dw1={analytic[0]:+.6f}  dw2={analytic[1]:+.6f}  db={analytic[2]:+.6f}")
    print(f"  numeric: dw1={numeric[0]:+.6f}  dw2={numeric[1]:+.6f}  db={numeric[2]:+.6f}")
    if not np.allclose(analytic, numeric, atol=1e-4):
        raise SystemExit("\n  MISMATCH -- your gradient formulas are wrong. Re-derive.")
    print("  match\n")

    print("Training:")
    for epoch in range(EPOCHS + 1):
        if epoch % 250 == 0:
            loss = mse(forward(X, w1, w2, b), Y)
            print(f"  epoch {epoch:5d}  loss {loss:10.6f}  w1 {w1:+.4f}  w2 {w2:+.4f}  b {b:+.4f}")
        dw1, dw2, db = gradients(X, Y, w1, w2, b)
        w1 -= LEARNING_RATE * dw1
        w2 -= LEARNING_RATE * dw2
        b -= LEARNING_RATE * db

    print(f"\nLearned:  w1={w1:.4f} (target  3.0)")
    print(f"          w2={w2:.4f} (target -2.0)")
    print(f"          b ={b:.4f} (target  5.0)")

    ok = abs(w1 - 3) < 0.05 and abs(w2 + 2) < 0.05 and abs(b - 5) < 0.05
    print("\nDONE" if ok else "\nNot converged yet -- try more epochs.")

# ----------------------------------------------------------------------
# Definition of done
#
#   w1 -> 3, w2 -> -2, b -> 5, and your gradients match the numeric check.
#
# Then answer these:
#
#   1. b converges more slowly than w1 and w2 here. Why might a parameter
#      that every example depends on equally move slower than one tied to
#      a specific input?
#
#   2. w2 has to become negative from a start of 0. Nothing in the code
#      says "allow negatives". What in the update rule makes that happen
#      automatically?
#
#   3. If you added a third input x3 that was always 0.0, what would
#      dL/dw3 be? Would w3 ever change? What does that tell you about
#      features that carry no information?
# ----------------------------------------------------------------------
