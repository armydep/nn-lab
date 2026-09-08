"""Chapter 4, Task 3 solution -- take out the nonlinearity and watch depth die.

Same network, same data, same training loop. The only change is the
activation: tanh becomes the identity.

    with tanh:      z1 = X@W1 + b1  ->  h = tanh(z1)  ->  z2 = h@W2 + b2
    with identity:  z1 = X@W1 + b1  ->  h = z1        ->  z2 = h@W2 + b2

The second one is algebraically a SINGLE linear layer:

    z2 = (X@W1 + b1)@W2 + b2 = X@(W1@W2) + (b1@W2 + b2)

Two layers, one effective matrix. XOR is not linearly separable, so it
cannot be solved -- no matter how many layers you stack.

    python3 solutions/04-mlp-from-scratch/optional/t3.py
"""

from __future__ import annotations

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 4
LEARNING_RATE = 0.2
EPOCHS = 5000
SEED = 1


def init_params(seed: int = SEED):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1.0, size=(2, HIDDEN)),
        "b1": np.zeros((1, HIDDEN)),
        "W2": rng.normal(0, 1.0, size=(HIDDEN, 1)),
        "b2": np.zeros((1, 1)),
    }


def train(p, nonlinear: bool, epochs: int = EPOCHS):
    n = X.shape[0]
    for _ in range(epochs):
        z1 = X @ p["W1"] + p["b1"]
        h = np.tanh(z1) if nonlinear else z1
        z2 = h @ p["W2"] + p["b2"]

        dz2 = 2 * (z2 - Y) / n
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dh = dz2 @ p["W2"].T
        # d(tanh)/dz = 1 - h^2;  d(identity)/dz = 1
        dz1 = dh * (1 - h**2) if nonlinear else dh
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        for name, grad in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            p[name] -= LEARNING_RATE * grad
    return p


def predict(p, nonlinear: bool):
    z1 = X @ p["W1"] + p["b1"]
    h = np.tanh(z1) if nonlinear else z1
    return h @ p["W2"] + p["b2"]


def report(label, p, nonlinear):
    z2 = predict(p, nonlinear)
    loss = float(np.mean((z2 - Y) ** 2))
    correct = int(np.sum((z2 >= 0.5) == (Y == 1)))
    print(f"\n--- {label} ---")
    print(f"loss {loss:.6f}   correct {correct}/4")
    for (a, b), target, pred in zip(X, Y.ravel(), z2.ravel()):
        mark = "" if (pred >= 0.5) == (target == 1) else "   <- wrong"
        print(f"   XOR({a:.0f}, {b:.0f}) = {target:.0f}   predicted {pred:+.4f}{mark}")
    return loss, correct


def main() -> None:
    print("=" * 62)
    print("WHAT THE NONLINEARITY IS FOR")
    print("=" * 62)

    p_tanh = train(init_params(), nonlinear=True)
    loss_t, correct_t = report("tanh", p_tanh, nonlinear=True)

    p_lin = train(init_params(), nonlinear=False)
    loss_l, correct_l = report("identity (no nonlinearity)", p_lin, nonlinear=False)

    # The collapse, demonstrated rather than asserted: fold the two layers
    # into one matrix and one bias, then compare predictions.
    W_eff = p_lin["W1"] @ p_lin["W2"]              # (2, 1)
    b_eff = p_lin["b1"] @ p_lin["W2"] + p_lin["b2"]  # (1, 1)
    collapsed = X @ W_eff + b_eff

    print("\n--- the two linear layers, folded into one ---")
    print(f"W_eff = [{W_eff[0, 0]:+.6f}, {W_eff[1, 0]:+.6f}]   b_eff = {b_eff[0, 0]:+.6f}")
    print(f"{'input':>10}{'2-layer':>14}{'1-layer':>14}{'difference':>14}")
    for (a, b), two, one in zip(X, predict(p_lin, False).ravel(), collapsed.ravel()):
        print(f"  ({a:.0f}, {b:.0f}){two:>14.9f}{one:>14.9f}{abs(two - one):>14.2e}")

    print("\n" + "=" * 62)
    print(f"{'activation':<14}{'loss':>12}{'correct':>10}")
    print(f"{'tanh':<14}{loss_t:>12.6f}{correct_t:>10}")
    print(f"{'identity':<14}{loss_l:>12.6f}{correct_l:>10}")

    assert correct_t == 4, "tanh should solve XOR"
    assert loss_t < 0.01, "tanh should reach a low loss"
    assert correct_l < 4, "a linear network must NOT solve XOR"
    assert loss_l > 0.2, "linear loss should stall near 0.25 -- the variance of Y"
    assert np.allclose(predict(p_lin, False), collapsed, atol=1e-12), (
        "two linear layers must be exactly equivalent to one"
    )

    print("\nThe linear network's predictions are identical to a single layer's,")
    print("to within floating-point noise. The second layer bought nothing.")

    print(
        "\nAnswers:\n"
        "  1. Loss stalls at about 0.25 because the best a linear model can\n"
        "     do on XOR is predict 0.5 everywhere, and the variance of Y is\n"
        "     0.25. That number is not a coincidence -- it is the floor.\n\n"
        "  2. Stacking more identity layers changes nothing. W1@W2@W3 is\n"
        "     still one matrix. Depth without a nonlinearity is not depth;\n"
        "     it is an expensive way to write a single matrix product.\n\n"
        "  3. The nonlinearity is what lets the hidden layer BEND the input\n"
        "     space, so that the output layer's straight line can separate\n"
        "     classes that were not separable in the original coordinates.\n"
        "     Chapter 4's hidden representation is exactly that bend.\n\n"
        "  4. Any nonlinearity works -- try relu in place of tanh and XOR\n"
        "     still falls. What matters is that it is not linear, not which\n"
        "     one you pick. Modern models choose between them on gradient\n"
        "     behaviour and speed, not on whether they work at all."
    )


if __name__ == "__main__":
    main()
