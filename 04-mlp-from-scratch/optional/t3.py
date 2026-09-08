"""Chapter 4, Task 3 -- delete the nonlinearity, and watch depth stop mattering.

Chapter 4 said a hidden layer plus a nonlinear activation increases what
the network can represent. This task isolates the second half of that
claim by removing only the activation.

    with tanh:      z1 = X@W1 + b1  ->  h = tanh(z1)  ->  z2 = h@W2 + b2
    with identity:  z1 = X@W1 + b1  ->  h = z1        ->  z2 = h@W2 + b2

Everything else is identical: same data, same shapes, same loop, same
number of parameters. One network solves XOR and the other provably
cannot.

    python3 04-mlp-from-scratch/optional/t3.py
"""

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 4
LEARNING_RATE = 0.2
EPOCHS = 5000
SEED = 1


def init_params(seed=SEED):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1.0, size=(2, HIDDEN)),
        "b1": np.zeros((1, HIDDEN)),
        "W2": rng.normal(0, 1.0, size=(HIDDEN, 1)),
        "b2": np.zeros((1, 1)),
    }


# ----------------------------------------------------------------------
# YOUR TURN (1 of 3) -- one training loop, both activations.
#
# Take a `nonlinear` flag and switch two lines on it:
#
#   forward:   h = np.tanh(z1)  if nonlinear else  z1
#   backward:  dz1 = dh * (1 - h**2)  if nonlinear else  dh
#
# That second line is the point: the derivative of the identity function
# is 1, so the gradient passes through the activation untouched.
# ----------------------------------------------------------------------
def train(p, nonlinear, epochs=EPOCHS):
    pass  # <- replace, return p


# ----------------------------------------------------------------------
# YOUR TURN (2 of 3) -- predictions, honouring the same flag.
# ----------------------------------------------------------------------
def predict(p, nonlinear):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 3) -- fold the linear network into ONE layer.
#
# If h = z1, then substituting gives
#
#   z2 = (X@W1 + b1)@W2 + b2 = X@(W1@W2) + (b1@W2 + b2)
#
# Compute W_eff = W1 @ W2 and b_eff = b1 @ W2 + b2, then verify that
# X @ W_eff + b_eff reproduces the two-layer predictions exactly -- to
# 1e-12, not approximately. Do not take this on faith; print the
# differences.
# ----------------------------------------------------------------------
def collapse(p):
    pass  # <- replace, return (W_eff, b_eff)


def main():
    p_tanh = train(init_params(), nonlinear=True)
    p_lin = train(init_params(), nonlinear=False)

    for label, p, nl in (("tanh", p_tanh, True), ("identity", p_lin, False)):
        z2 = predict(p, nl)
        loss = float(np.mean((z2 - Y) ** 2))
        correct = int(np.sum((z2 >= 0.5) == (Y == 1)))
        print(f"\n--- {label} ---")
        print(f"loss {loss:.6f}   correct {correct}/4")
        for (a, b), target, pred in zip(X, Y.ravel(), z2.ravel()):
            mark = "" if (pred >= 0.5) == (target == 1) else "   <- wrong"
            print(f"   XOR({a:.0f}, {b:.0f}) = {target:.0f}   predicted {pred:+.4f}{mark}")

    W_eff, b_eff = collapse(p_lin)
    print(f"\nfolded: W_eff = {W_eff.ravel()}   b_eff = {b_eff.ravel()}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   tanh reaches 4/4. The identity network does not, its loss stalls near
#   0.25, and your folded single layer reproduces its predictions to
#   within 1e-12.
#
# Then think about:
#
#   1. The identity network's loss settles at almost exactly 0.25. That is
#      not arbitrary -- it is the variance of Y. What is the network
#      predicting for every input, and why is that the best a straight
#      line can do here?
#
#   2. You collapsed two layers into one. What happens if you add a third
#      and fourth identity layer? Write out W1@W2@W3@W4 and say what kind
#      of object it is.
#
#   3. Chapter 4 called the hidden layer a "new representation". In your
#      own words, what does the nonlinearity let that representation do
#      that a linear map cannot?
#
#   4. Swap tanh for relu -- np.maximum(0, z1), with derivative (z1 > 0).
#      XOR should still fall. What does that tell you about which
#      nonlinearity matters, versus that there is one at all?
# ----------------------------------------------------------------------
