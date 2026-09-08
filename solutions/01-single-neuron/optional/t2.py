"""Chapter 1, Task 2 solution -- two-input neuron."""

import numpy as np

X = np.array(
    [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0], [2.0, 3.0], [4.0, 1.0]]
)
Y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 5.0

LEARNING_RATE = 0.02
EPOCHS = 2000


def forward(x, w1, w2, b):
    return w1 * x[:, 0] + w2 * x[:, 1] + b


def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)


# L = mean( (w1*x1 + w2*x2 + b - y)^2 )
#
# Differentiating with respect to w1, the w2*x2 term is a constant -- it
# contributes nothing. So the ONLY change from the one-input case is which
# input multiplies the error:
#
#   dL/dw1 = mean(2 * error * x1)
#   dL/dw2 = mean(2 * error * x2)
#   dL/db  = mean(2 * error)          (b's "input" is always 1)
#
# Each weight's gradient is the error scaled by the input that weight
# controls. That pattern holds for every network in this repository.
def gradients(x, y, w1, w2, b):
    error = forward(x, w1, w2, b) - y
    dw1 = np.mean(2 * error * x[:, 0])
    dw2 = np.mean(2 * error * x[:, 1])
    db = np.mean(2 * error)
    return dw1, dw2, db


def numeric_gradients(x, y, w1, w2, b, eps=1e-6):
    def loss_at(a, c, d):
        return mse(forward(x, a, c, d), y)

    return (
        (loss_at(w1 + eps, w2, b) - loss_at(w1 - eps, w2, b)) / (2 * eps),
        (loss_at(w1, w2 + eps, b) - loss_at(w1, w2 - eps, b)) / (2 * eps),
        (loss_at(w1, w2, b + eps) - loss_at(w1, w2, b - eps)) / (2 * eps),
    )


if __name__ == "__main__":
    w1, w2, b = 0.0, 0.0, 0.0

    analytic = gradients(X, Y, w1, w2, b)
    numeric = numeric_gradients(X, Y, w1, w2, b)
    print("Gradient check:")
    print(f"  analytic: {tuple(round(v, 6) for v in analytic)}")
    print(f"  numeric:  {tuple(round(v, 6) for v in numeric)}")
    assert np.allclose(analytic, numeric, atol=1e-4)
    print("  match\n")

    for epoch in range(EPOCHS + 1):
        if epoch % 500 == 0:
            loss = mse(forward(X, w1, w2, b), Y)
            print(f"  epoch {epoch:5d}  loss {loss:10.6f}  w1 {w1:+.4f}  w2 {w2:+.4f}  b {b:+.4f}")
        dw1, dw2, db = gradients(X, Y, w1, w2, b)
        w1 -= LEARNING_RATE * dw1
        w2 -= LEARNING_RATE * dw2
        b -= LEARNING_RATE * db

    print(f"\nLearned: w1={w1:.4f} (3.0)  w2={w2:.4f} (-2.0)  b={b:.4f} (5.0)")
    assert abs(w1 - 3) < 0.05 and abs(w2 + 2) < 0.05 and abs(b - 5) < 0.05

    print(
        "\nAnswers:\n"
        "  1. b's gradient is mean(2*error), with no input scaling it. The\n"
        "     weights get multiplied by x values of 1-4, so their gradients\n"
        "     are several times larger and they move faster. Inputs with\n"
        "     larger magnitudes produce larger gradients -- which is exactly\n"
        "     why feature scaling matters in real models.\n\n"
        "  2. Nothing constrains the sign. w2 starts at 0, its gradient comes\n"
        "     out positive, and w2 -= lr * positive drives it negative. The\n"
        "     update rule follows the gradient wherever it leads.\n\n"
        "  3. dL/dw3 = mean(2 * error * 0) = 0, always. w3 would never move\n"
        "     from its initial value. A feature that never varies cannot\n"
        "     influence the loss, so it receives no gradient and is never\n"
        "     learned -- the model ignores it automatically."
    )
