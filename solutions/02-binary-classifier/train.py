"""Chapter 2 solution: a sigmoid neuron for binary classification.

Demonstrates the dL/dz = p - y result and the log(0) stability trap.

    python3 solutions/02-binary-classifier/train.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

# --- A tiny linearly separable 2D dataset ------------------------------
# Class 0 clusters near (1, 1); class 1 clusters near (3, 3).
X = np.array(
    [
        [1.0, 1.0], [1.5, 0.8], [0.8, 1.4], [1.2, 1.6],
        [3.0, 3.0], [3.2, 2.7], [2.8, 3.3], [3.4, 3.1],
    ]
)
Y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

EPS = 1e-12
LEARNING_RATE = 0.5
EPOCHS = 400


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    The naive 1/(1+exp(-z)) overflows for large negative z, because exp(-z)
    becomes inf. Branching on the sign keeps every exp() argument <= 0.
    """
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce(p: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy, clipped to avoid log(0) -> -inf -> nan."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def forward(x: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray]:
    z = x @ w + b
    return z, sigmoid(z)


# --- The gradient ------------------------------------------------------
# L = -[y*log(p) + (1-y)*log(1-p)],  p = sigmoid(z)
#
#   dL/dp = (p - y) / (p * (1 - p))
#   dp/dz = p * (1 - p)
#
# Multiplying them, the p*(1-p) cancels completely:
#
#   dL/dz = p - y
#
# Prediction minus target. This cancellation is also *why* the fused
# "sigmoid + BCE" loss in every framework is numerically safe: the
# dangerous p*(1-p) division never has to be computed at all.


def gradients(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    _, p = forward(x, w, b)
    dz = p - y                      # the whole gradient, in one line
    dw = x.T @ dz / len(y)
    db = float(np.mean(dz))
    return dw, db


def numeric_gradients(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, eps: float = 1e-6
) -> tuple[np.ndarray, float]:
    dw = np.zeros_like(w)
    for i in range(len(w)):
        up, down = w.copy(), w.copy()
        up[i] += eps
        down[i] -= eps
        dw[i] = (bce(forward(x, up, b)[1], y) - bce(forward(x, down, b)[1], y)) / (2 * eps)
    db = (bce(forward(x, w, b + eps)[1], y) - bce(forward(x, w, b - eps)[1], y)) / (2 * eps)
    return dw, db


def main() -> None:
    w = rng.normal(0, 0.5, size=2)
    b = 0.0

    analytic_w, analytic_b = gradients(X, Y, w, b)
    numeric_w, numeric_b = numeric_gradients(X, Y, w, b)
    print("Gradient check (confirms dL/dz = p - y):")
    print(f"  analytic dw={np.round(analytic_w, 6)}  db={analytic_b:+.6f}")
    print(f"  numeric  dw={np.round(numeric_w, 6)}  db={numeric_b:+.6f}")
    assert np.allclose(analytic_w, numeric_w, atol=1e-5)
    assert np.isclose(analytic_b, numeric_b, atol=1e-5)
    print("  match\n")

    print("Training:")
    for epoch in range(EPOCHS + 1):
        _, p = forward(X, w, b)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}   loss {bce(p, Y):8.6f}")
        dw, db = gradients(X, Y, w, b)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    _, p = forward(X, w, b)
    print(f"\nFinal loss: {bce(p, Y):.6f}")
    print(f"Weights: w = {np.round(w, 4)}, b = {b:.4f}\n")

    print("Predicted probabilities:")
    correct = 0
    for x_i, y_i, p_i in zip(X, Y, p):
        predicted = 1 if p_i >= 0.5 else 0
        correct += predicted == int(y_i)
        print(
            f"  x=({x_i[0]:.1f}, {x_i[1]:.1f})  p={p_i:.4f}  "
            f"predicted={predicted}  actual={int(y_i)}"
        )
    print(f"\nAccuracy: {correct}/{len(Y)}")

    # The decision boundary is where z = 0, i.e. w1*x1 + w2*x2 + b = 0.
    print(
        f"Decision boundary: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f} = 0\n"
        "  Points on one side get p > 0.5, the other side p < 0.5.\n"
        "  Sigmoid turns the signed distance from this line into a probability."
    )


if __name__ == "__main__":
    main()
