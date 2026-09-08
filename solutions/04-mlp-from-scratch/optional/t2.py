"""Chapter 4, Task 2 solution -- symmetry, and why random init is not superstition.

Three initialisations of the same network:

    zeros      nothing moves at all
    constant   everything moves together, and stays identical forever
    random     hidden units differentiate, XOR is learned

The lesson is that the problem is SYMMETRY, not the value zero.

    python3 solutions/04-mlp-from-scratch/optional/t2.py
"""

from __future__ import annotations

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 4
LEARNING_RATE = 0.2
EPOCHS = 5000


def init_zeros():
    return {
        "W1": np.zeros((2, HIDDEN)),
        "b1": np.zeros((1, HIDDEN)),
        "W2": np.zeros((HIDDEN, 1)),
        "b2": np.zeros((1, 1)),
    }


def init_constant(c: float = 0.5):
    """Non-zero, but every hidden unit still identical to every other."""
    return {
        "W1": np.full((2, HIDDEN), c),
        "b1": np.full((1, HIDDEN), c),
        "W2": np.full((HIDDEN, 1), c),
        "b2": np.zeros((1, 1)),
    }


def init_random(seed: int = 1):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1.0, size=(2, HIDDEN)),
        "b1": np.zeros((1, HIDDEN)),
        "W2": rng.normal(0, 1.0, size=(HIDDEN, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]
    h = np.tanh(z1)
    z2 = h @ p["W2"] + p["b2"]
    return z1, h, z2


def train(p, epochs=EPOCHS):
    n = X.shape[0]
    for _ in range(epochs):
        _, h, z2 = forward(X, p)

        dz2 = 2 * (z2 - Y) / n
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dh = dz2 @ p["W2"].T
        dz1 = dh * (1 - h**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        for name, grad in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            p[name] -= LEARNING_RATE * grad
    return p


def evaluate(p):
    _, h, z2 = forward(X, p)
    loss = float(np.mean((z2 - Y) ** 2))
    correct = int(np.sum((z2 >= 0.5) == (Y == 1)))
    distinct = np.unique(h.round(9), axis=1).shape[1]   # distinct hidden-unit columns
    return loss, correct, h, distinct


def report(label, p):
    loss, correct, h, distinct = evaluate(p)
    print(f"\n--- {label} ---")
    print(f"loss {loss:.6f}   correct {correct}/4   distinct hidden units {distinct}/{HIDDEN}")
    print("hidden activations (one row per input):")
    for row in h:
        print("   " + "  ".join(f"{v:+.6f}" for v in row))
    return loss, correct, distinct


def main() -> None:
    print("=" * 62)
    print("SYMMETRY BREAKING")
    print("=" * 62)

    p_zero = train(init_zeros())
    loss_z, correct_z, distinct_z = report("zeros", p_zero)

    p_const = train(init_constant())
    loss_c, correct_c, distinct_c = report("constant 0.5 (non-zero, still identical)", p_const)

    p_rand = train(init_random())
    loss_r, correct_r, distinct_r = report("random", p_rand)

    print("\n" + "=" * 62)
    print(f"{'init':<12}{'loss':>12}{'correct':>10}{'distinct units':>16}")
    print(f"{'zeros':<12}{loss_z:>12.6f}{correct_z:>10}{distinct_z:>16}")
    print(f"{'constant':<12}{loss_c:>12.6f}{correct_c:>10}{distinct_c:>16}")
    print(f"{'random':<12}{loss_r:>12.6f}{correct_r:>10}{distinct_r:>16}")

    # Zeros: W2 starts at 0, so dh = dz2 @ W2.T = 0 and W1 never receives
    # any gradient at all. Only b2 can move, to the mean of Y.
    assert distinct_z == 1, "zero init should leave every hidden unit identical"
    assert np.allclose(p_zero["W1"], 0.0), "W1 should never have moved from zero"
    assert correct_z < 4, "zero init must not solve XOR"

    # Constant: gradients are non-zero, but identical across units, so the
    # units move together and stay indistinguishable. Non-zero did not help.
    assert distinct_c == 1, "constant init should keep hidden units identical"
    assert correct_c < 4, "constant init must not solve XOR either"

    # Random: units differentiate and XOR falls.
    assert distinct_r > 1, "random init should differentiate the hidden units"
    assert correct_r == 4, "random init should solve XOR"
    assert loss_r < 0.01, "random init should reach a low loss"

    print(
        "\nAnswers:\n"
        "  1. With zeros, W2 is zero, so dh = dz2 @ W2.T is zero and W1 gets\n"
        "     no gradient whatsoever. Only b2 moves -- to the mean of Y, 0.5.\n"
        "     The network predicts 0.5 for everything and loss sticks at 0.25.\n\n"
        "  2. The constant run is the interesting one. Its gradients are NOT\n"
        "     zero -- the weights genuinely change every epoch. But every\n"
        "     hidden unit sees identical inputs and identical gradients, so\n"
        "     they move in lockstep and remain identical forever. Four hidden\n"
        "     units, one distinct function. That proves the problem is\n"
        "     symmetry, not the number zero.\n\n"
        "  3. Biases can safely start at zero because the WEIGHTS have\n"
        "     already broken the symmetry -- each unit receives a different\n"
        "     weighted sum, so identical biases do not make them identical.\n\n"
        "  4. Scale matters as well as randomness. Initialise with\n"
        "     rng.normal(0, 50) and tanh saturates immediately: (1 - h**2)\n"
        "     goes to zero and gradients vanish. Symmetry broken, learning\n"
        "     still dead. Schemes like Xavier and He initialisation exist to\n"
        "     choose that scale rather than guess it."
    )


if __name__ == "__main__":
    main()
