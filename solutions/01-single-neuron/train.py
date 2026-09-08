"""Chapter 1 solution: a single trainable linear neuron.

Learns y = 2x + 1 from scratch. No autograd -- gradients derived by hand.

    python3 solutions/01-single-neuron/train.py
"""

from __future__ import annotations

import numpy as np

# --- Data: y = 2x + 1 --------------------------------------------------
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = 2.0 * X + 1.0

# --- Parameters --------------------------------------------------------
w = 0.0
b = 0.0
LEARNING_RATE = 0.05
EPOCHS = 200


def forward(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """y_hat = w*x + b"""
    return w * x + b


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_hat - y) ** 2))


# --- Gradient derivation -----------------------------------------------
#   L    = mean((w*x + b - y)^2)
#   dL/dw = mean(2 * (w*x + b - y) * x)
#   dL/db = mean(2 * (w*x + b - y) * 1)
#
# The (y_hat - y) term is the error. Both gradients are "error, weighted by
# how much each parameter influenced the output". For w that influence is x;
# for b it is 1, since b shifts the output one-for-one.


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    error = forward(x, w, b) - y
    dw = float(np.mean(2.0 * error * x))
    db = float(np.mean(2.0 * error))
    return dw, db


def numeric_gradients(
    x: np.ndarray, y: np.ndarray, w: float, b: float, eps: float = 1e-5
) -> tuple[float, float]:
    """Finite-difference check -- the tool introduced properly in Chapter 3."""
    dw = (mse(forward(x, w + eps, b), y) - mse(forward(x, w - eps, b), y)) / (2 * eps)
    db = (mse(forward(x, w, b + eps), y) - mse(forward(x, w, b - eps), y)) / (2 * eps)
    return dw, db


def main() -> None:
    global w, b

    # Verify the derivation before trusting the training loop.
    analytic = gradients(X, Y, w, b)
    numeric = numeric_gradients(X, Y, w, b)
    print("Gradient check at start:")
    print(f"  analytic dw={analytic[0]:+.6f}  db={analytic[1]:+.6f}")
    print(f"  numeric  dw={numeric[0]:+.6f}  db={numeric[1]:+.6f}")
    assert np.allclose(analytic, numeric, atol=1e-4), "gradient derivation is wrong"
    print("  match\n")

    print("Training:")
    for epoch in range(EPOCHS + 1):
        loss = mse(forward(X, w, b), Y)
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}   loss {loss:10.6f}   w {w:+.4f}   b {b:+.4f}")

        dw, db = gradients(X, Y, w, b)
        # Gradient descent: step *against* the gradient, because the gradient
        # points uphill and we want to go down.
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    print(f"\nLearned:  w = {w:.4f}  (target 2.0)")
    print(f"          b = {b:.4f}  (target 1.0)")
    print(f"Final loss: {mse(forward(X, w, b), Y):.8f}")

    print("\nPredictions:")
    for x_i, y_i in zip(X, Y):
        print(f"  x={x_i:.0f}  predicted={forward(x_i, w, b):7.4f}  actual={y_i:.1f}")


if __name__ == "__main__":
    main()
