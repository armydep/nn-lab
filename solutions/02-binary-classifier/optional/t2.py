"""Chapter 2, Task 2 solution -- non-separable data."""

import numpy as np

X = np.array(
    [
        [1.0, 1.0], [1.5, 1.2], [0.8, 1.6], [1.3, 0.9], [2.6, 2.4],
        [3.0, 3.0], [3.2, 2.7], [2.8, 3.3], [3.4, 3.1], [1.4, 1.5],
    ]
)
Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

EPS = 1e-12
LEARNING_RATE = 0.3
EPOCHS = 3000


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def forward(x, w, b):
    z = x @ w + b
    return z, sigmoid(z)


def gradients(x, y, w, b):
    _, p = forward(x, w, b)
    dz = p - y                       # dL/dz = p - y
    return x.T @ dz / len(y), np.mean(dz)


def plot_boundary(w, b):
    print("\n  Decision boundary ('.' predicts 0, '#' predicts 1)")
    print("  Data points: 0 / 1 = correct, X = misclassified\n")
    for row in range(16, -1, -1):
        x2 = row * 0.25
        line = ""
        for col in range(17):
            x1 = col * 0.25
            marker = " "
            for (px1, px2), py in zip(X, Y):
                if abs(px1 - x1) < 0.13 and abs(px2 - x2) < 0.13:
                    _, p = forward(np.array([[px1, px2]]), w, b)
                    marker = str(int(py)) if (1 if p[0] >= 0.5 else 0) == int(py) else "X"
                    break
            if marker == " ":
                _, p = forward(np.array([[x1, x2]]), w, b)
                marker = "#" if p[0] >= 0.5 else "."
            line += marker
        print(f"  {x2:4.1f} |{line}")
    print("       " + "-" * 17)
    print("        0   1   2   3   4")


if __name__ == "__main__":
    w, b = np.zeros(2), 0.0

    for epoch in range(EPOCHS + 1):
        _, p = forward(X, w, b)
        if epoch % 1000 == 0:
            accuracy = np.mean((p >= 0.5) == (Y == 1))
            print(f"  epoch {epoch:5d}   loss {bce(p, Y):.6f}   accuracy {accuracy:.0%}")
        dw, db = gradients(X, Y, w, b)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    _, p = forward(X, w, b)
    accuracy = np.mean((p >= 0.5) == (Y == 1))
    print(f"\nFinal loss {bce(p, Y):.6f}   accuracy {accuracy:.0%}")

    print("\nPer-point:")
    for (x1, x2), y_i, p_i in zip(X, Y, p):
        predicted = 1 if p_i >= 0.5 else 0
        flag = "  <- wrong" if predicted != int(y_i) else ""
        print(f"  ({x1:.1f}, {x2:.1f})  actual {int(y_i)}  p={p_i:.3f} -> {predicted}{flag}")

    plot_boundary(w, b)

    print(
        "\nAnswers:\n"
        "  1. Accuracy is a step function -- it only changes when a point\n"
        "     crosses p = 0.5. Loss is continuous: pushing a correct point\n"
        "     from p=0.7 to p=0.9 lowers the loss while changing no label.\n"
        "     This is precisely why we train on cross-entropy and not on\n"
        "     accuracy: accuracy has zero gradient almost everywhere, so\n"
        "     gradient descent would have nothing to follow.\n\n"
        "  2. Moving the outlier out of the opposing cluster lets the line\n"
        "     separate more of the data, and the boundary rotates because it\n"
        "     no longer has to compromise for that point.\n\n"
        "  3. A wrong point at p=0.99 contributes -log(0.01) = 4.6 to the\n"
        "     loss; at p=0.51 it contributes only 0.71. Cross-entropy grows\n"
        "     without bound as confidence in a wrong answer rises, while\n"
        "     squared error caps out at 1.0. Confident and wrong is the\n"
        "     failure worth punishing hardest.\n\n"
        "  4. Same limitation as XOR: one straight line, and this data needs\n"
        "     a bent one. The fix is the same too -- a hidden layer, which is\n"
        "     exactly what Chapter 4 builds."
    )
