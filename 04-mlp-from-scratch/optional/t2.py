"""Chapter 4, Task 2 -- why random initialization, and what actually breaks.

The main exercise told you to initialize randomly, "NOT zeros". This task
makes you prove why, and corrects the usual half-understanding of it.

You will train the same network three ways:

    zeros       every weight 0.0
    constant    every weight 0.5   <- non-zero, but still all identical
    random      drawn from a normal distribution

Most people believe the problem is the value zero. Run the constant case
before you decide.

    python3 04-mlp-from-scratch/optional/t2.py
"""

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


# ----------------------------------------------------------------------
# YOUR TURN (1 of 4) -- the constant initialization.
#
# Every weight and bias set to the same non-zero value c. Same shapes as
# init_zeros. np.full is the tool.
#
# Predict, before running: will this solve XOR? It is not zero, so the
# usual advice says it should be fine.
# ----------------------------------------------------------------------
def init_constant(c=0.5):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 4) -- the random initialization.
#
# W1 and W2 from rng.normal(0, 1.0, size=...); biases zero. Use
# np.random.default_rng(seed) so the run is reproducible.
# ----------------------------------------------------------------------
def init_random(seed=1):
    pass  # <- replace


def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]
    h = np.tanh(z1)
    z2 = h @ p["W2"] + p["b2"]
    return z1, h, z2


# ----------------------------------------------------------------------
# YOUR TURN (3 of 4) -- the training loop.
#
# This is Chapter 4's backward pass, unchanged. Work backwards:
#
#   dz2 = 2*(z2 - Y) / n        dW2 = h.T @ dz2      db2 = dz2.sum(0)
#   dh  = dz2 @ W2.T            dz1 = dh * (1-h^2)
#   dW1 = X.T @ dz1             db1 = dz1.sum(0)
#
# Keep the sums as keepdims=True so the bias shapes stay (1, n).
# ----------------------------------------------------------------------
def train(p, epochs=EPOCHS):
    pass  # <- replace, return p


# ----------------------------------------------------------------------
# YOUR TURN (4 of 4) -- the measurement that makes the point.
#
# Return (loss, correct, h, distinct) where `distinct` is the number of
# DIFFERENT hidden units -- the number of unique columns in h:
#
#   np.unique(h.round(9), axis=1).shape[1]
#
# That single number is the whole lesson. Watch it, not the loss.
# ----------------------------------------------------------------------
def evaluate(p):
    pass  # <- replace


def main():
    for label, init in (
        ("zeros", init_zeros),
        ("constant 0.5", init_constant),
        ("random", init_random),
    ):
        p = train(init())
        loss, correct, h, distinct = evaluate(p)
        print(f"\n--- {label} ---")
        print(f"loss {loss:.6f}   correct {correct}/4   distinct hidden units {distinct}/{HIDDEN}")
        for row in h:
            print("   " + "  ".join(f"{v:+.6f}" for v in row))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   Random reaches 4/4 with 4 distinct hidden units. Zeros and constant
#   both reach fewer than 4/4 with exactly 1 distinct hidden unit.
#
# Then think about:
#
#   1. With zeros, W2 starts at zero, so dh = dz2 @ W2.T is zero and W1
#      never receives any gradient at all. Check it: is W1 still exactly
#      zero after 5000 epochs? What is the only parameter that moved, and
#      why does the loss settle at 0.25 specifically?
#
#   2. The constant run's gradients are NOT zero -- the weights really do
#      change every epoch. So why do the hidden units stay identical? Say
#      it in one sentence, and notice that your sentence never mentions
#      the number zero.
#
#   3. If symmetry is the problem, why is it safe to start the BIASES at
#      zero, as the main exercise does?
#
#   4. Randomness is necessary but not sufficient. Try rng.normal(0, 50).
#      Symmetry is broken -- so why does learning still fail? (Look at
#      what 1 - h**2 is when tanh saturates.) This is why initialization
#      schemes specify a scale, not just "random".
# ----------------------------------------------------------------------
