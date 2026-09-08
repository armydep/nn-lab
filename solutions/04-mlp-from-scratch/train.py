"""Chapter 4 solution: a two-layer MLP that learns XOR.

Chapter 3's two-layer graph, with matrices and a batch. The chain rule is
identical -- only the shapes changed.

    python3 solutions/04-mlp-from-scratch/train.py
"""

from __future__ import annotations

import numpy as np

# --- XOR: the smallest problem no linear model can solve ---------------
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 4
LEARNING_RATE = 0.5
EPOCHS = 5000
SEED = 1


def init_params(seed: int = SEED) -> dict[str, np.ndarray]:
    """Random init -- NOT zeros.

    With zero (or otherwise identical) weights, every hidden unit computes the
    same thing, therefore receives the same gradient, therefore stays identical
    forever. The network collapses to a single neuron and XOR stays unsolvable.
    Random values break that symmetry.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1.0, size=(2, HIDDEN)),
        "b1": np.zeros((1, HIDDEN)),      # biases may safely start at zero:
        "W2": rng.normal(0, 1.0, size=(HIDDEN, 1)),
        "b2": np.zeros((1, 1)),           # the weights already broke symmetry
    }


def forward(x: np.ndarray, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Store every intermediate -- the backward pass needs them."""
    z1 = x @ p["W1"] + p["b1"]        # (N, HIDDEN)
    h = np.tanh(z1)                   # (N, HIDDEN)
    z2 = h @ p["W2"] + p["b2"]        # (N, 1)
    return {"z1": z1, "h": h, "z2": z2}


def loss_fn(z2: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((z2 - y) ** 2))


def backward(
    x: np.ndarray, y: np.ndarray, p: dict[str, np.ndarray], cache: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Backpropagation. Compare line by line with Chapter 3 step 3.

    Shape rule: every gradient has the same shape as the thing it
    differentiates. dW1 is (2, HIDDEN) because W1 is (2, HIDDEN).
    """
    n = len(x)
    h, z2 = cache["h"], cache["z2"]

    # Through the squared error. The 2/n is the derivative of the mean.
    dz2 = 2.0 * (z2 - y) / n                 # (N, 1)

    dW2 = h.T @ dz2                          # (HIDDEN,N)@(N,1) -> (HIDDEN,1)
    db2 = dz2.sum(axis=0, keepdims=True)     # (1, 1) -- sum over the batch

    dh = dz2 @ p["W2"].T                     # (N,1)@(1,HIDDEN) -> (N,HIDDEN)
    dz1 = dh * (1.0 - h**2)                  # d/dz tanh = 1 - tanh^2
    dW1 = x.T @ dz1                          # (2,N)@(N,HIDDEN) -> (2,HIDDEN)
    db1 = dz1.sum(axis=0, keepdims=True)     # (1, HIDDEN)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def gradient_check(p: dict[str, np.ndarray], eps: float = 1e-6) -> None:
    """Verify every parameter matrix against finite differences."""
    analytic = backward(X, Y, p, forward(X, p))

    print("Gradient check:")
    for name in ("W1", "b1", "W2", "b2"):
        numeric = np.zeros_like(p[name])
        # np.nditer walks every element of the matrix regardless of its shape.
        it = np.nditer(p[name], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = p[name][idx]

            p[name][idx] = original + eps
            up = loss_fn(forward(X, p)["z2"], Y)
            p[name][idx] = original - eps
            down = loss_fn(forward(X, p)["z2"], Y)
            p[name][idx] = original          # always restore

            numeric[idx] = (up - down) / (2 * eps)
            it.iternext()

        max_diff = np.max(np.abs(analytic[name] - numeric))
        status = "OK" if max_diff < 1e-6 else "MISMATCH"
        print(f"  {name:3s} shape {str(p[name].shape):9s} max diff {max_diff:.2e}  {status}")
        assert max_diff < 1e-6, f"gradient for {name} is wrong"
    print()


def main() -> None:
    p = init_params()
    gradient_check(p)

    print("Training:")
    for epoch in range(EPOCHS + 1):
        cache = forward(X, p)
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d}   loss {loss_fn(cache['z2'], Y):.8f}")

        grads = backward(X, Y, p, cache)
        for name in p:
            p[name] -= LEARNING_RATE * grads[name]

    cache = forward(X, p)
    predictions = cache["z2"]
    print(f"\nFinal loss: {loss_fn(predictions, Y):.8f}\n")

    print("XOR truth table:")
    correct = 0
    for x_i, y_i, pred in zip(X, Y, predictions):
        rounded = 1 if pred[0] >= 0.5 else 0
        correct += rounded == int(y_i[0])
        print(
            f"  {int(x_i[0])} XOR {int(x_i[1])} = {int(y_i[0])}   "
            f"predicted {pred[0]:+.4f} -> {rounded}"
        )
    print(f"\nCorrect: {correct}/4")

    print("\nWhy a single linear neuron cannot do this:")
    print(
        "  A single neuron computes w1*x1 + w2*x2 + b, then thresholds it.\n"
        "  That draws ONE straight line and calls everything on one side 1.\n"
        "  XOR needs (0,1) and (1,0) as class 1, with (0,0) and (1,1) as\n"
        "  class 0 -- the two positives sit on opposite corners. No single\n"
        "  straight line separates them; you would need two.\n\n"
        "  The hidden layer builds the extra lines. Each hidden unit is its\n"
        "  own linear boundary, tanh bends the result, and the output layer\n"
        "  combines them. Depth buys the ability to represent a decision\n"
        "  boundary that is not a single straight cut."
    )

    print("\nWhat the hidden units learned (pre-activation z1 per input):")
    for x_i, z1_row in zip(X, cache["z1"]):
        formatted = "  ".join(f"{v:+.2f}" for v in z1_row)
        print(f"  input ({int(x_i[0])},{int(x_i[1])}) -> [{formatted}]")


if __name__ == "__main__":
    main()
