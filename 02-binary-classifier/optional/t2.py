"""Chapter 2, Task 2 -- data that does NOT separate cleanly.

The Chapter 2 exercise used two tidy clusters. Real data overlaps.

Here, three points sit on the wrong side of any line you could draw. The
model cannot reach 100% accuracy, and it should not. Your job is to watch
what it does instead -- and to see the decision boundary it settles on.

    python3 02-binary-classifier/optional/t2.py
"""

import numpy as np

# ----------------------------------------------------------------------
# Two overlapping clusters. The starred points are the troublemakers:
# class-0 points sitting inside class-1 territory, and vice versa.
# ----------------------------------------------------------------------
X = np.array(
    [
        [1.0, 1.0], [1.5, 1.2], [0.8, 1.6], [1.3, 0.9], [2.6, 2.4],  # <- * class 0, deep in 1
        [3.0, 3.0], [3.2, 2.7], [2.8, 3.3], [3.4, 3.1], [1.4, 1.5],  # <- * class 1, deep in 0
    ]
)
Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

EPS = 1e-12
LEARNING_RATE = 0.3
EPOCHS = 3000


# ----------------------------------------------------------------------
# YOUR TURN (1 of 4) -- sigmoid
#
# 1 / (1 + exp(-z)). Squashes any real number into (0, 1).
# ----------------------------------------------------------------------
def sigmoid(z):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 4) -- binary cross-entropy
#
#   L = -mean( y*log(p) + (1-y)*log(1-p) )
#
# Clip p into [EPS, 1-EPS] before the logs. Task 3 shows you exactly what
# happens if you don't.
# ----------------------------------------------------------------------
def bce(p, y):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 4) -- forward pass
#
# Return BOTH z (the raw score) and p (the probability), as a tuple.
# ----------------------------------------------------------------------
def forward(x, w, b):
    pass  # <- replace, return (z, p)


# ----------------------------------------------------------------------
# YOUR TURN (4 of 4) -- gradients
#
# Use the result from Chapter 2: dL/dz = p - y. That is the whole thing.
# From dz, the rest follows exactly as in Chapter 1:
#
#   dL/dw = (how much each input contributed) averaged over examples
#   dL/db = mean(dz)
#
# x.T @ dz gives you the per-weight sums in one operation. Divide by the
# number of examples to make it a mean.
# ----------------------------------------------------------------------
def gradients(x, y, w, b):
    pass  # <- replace, return (dw, db)


# ----------------------------------------------------------------------
# Provided: draws the decision boundary as ASCII so you can SEE it.
# Not the lesson -- don't spend time reading it.
# ----------------------------------------------------------------------
def plot_boundary(w, b):
    print("\n  Decision boundary ('.' = predicts 0, '#' = predicts 1)")
    print("  Your data: 0 = class 0, 1 = class 1, X = misclassified\n")
    for row in range(16, -1, -1):
        x2 = row * 0.25
        line = ""
        for col in range(17):
            x1 = col * 0.25
            marker = " "
            for (px1, px2), py in zip(X, Y):
                if abs(px1 - x1) < 0.13 and abs(px2 - x2) < 0.13:
                    _, p = forward(np.array([[px1, px2]]), w, b)
                    predicted = 1 if p[0] >= 0.5 else 0
                    marker = str(int(py)) if predicted == int(py) else "X"
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

    if sigmoid(np.zeros(1)) is None or forward(X, w, b) is None:
        raise SystemExit("Write sigmoid() and forward() first.")
    if bce(np.full(10, 0.5), Y) is None or gradients(X, Y, w, b) is None:
        raise SystemExit("Now write bce() and gradients().")

    print(f"Sanity: sigmoid(0) should be 0.5 -> {sigmoid(np.array([0.0]))[0]:.4f}\n")

    print("Training:")
    for epoch in range(EPOCHS + 1):
        _, p = forward(X, w, b)
        if epoch % 500 == 0:
            accuracy = np.mean((p >= 0.5) == (Y == 1))
            print(f"  epoch {epoch:5d}   loss {bce(p, Y):.6f}   accuracy {accuracy:.0%}")
        dw, db = gradients(X, Y, w, b)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    _, p = forward(X, w, b)
    accuracy = np.mean((p >= 0.5) == (Y == 1))
    print(f"\nFinal loss: {bce(p, Y):.6f}    accuracy: {accuracy:.0%}")

    print("\nPer-point predictions:")
    for (x1, x2), y_i, p_i in zip(X, Y, p):
        predicted = 1 if p_i >= 0.5 else 0
        flag = "  <- wrong" if predicted != int(y_i) else ""
        print(f"  ({x1:.1f}, {x2:.1f})  actual {int(y_i)}  p={p_i:.3f}  ->  {predicted}{flag}")

    plot_boundary(w, b)

    print(
        "\n  The loss did not reach zero and the accuracy did not reach 100%.\n"
        "  That is the correct outcome. No straight line separates this data,\n"
        "  so the model settles on the line that is least wrong overall.\n\n"
        "  Compare the probabilities on the two misclassified points against\n"
        "  the points it gets right. The wrong ones are noticeably LESS\n"
        "  extreme -- the model hedges on them rather than committing. It is\n"
        "  wrong, but it is not confidently wrong, and cross-entropy is what\n"
        "  pushed it toward that hedge."
    )

# ----------------------------------------------------------------------
# Definition of done
#
#   Training converges, accuracy lands at 80% (8 of 10), and you can point
#   at the two points it gets wrong and say why no line could fix them.
#
# Then think about:
#
#   1. Loss keeps falling after accuracy stops improving. How can the loss
#      improve when no additional point flips to correct?
#
#   2. Try moving one troublemaker -- change [2.6, 2.4] to [2.6, 0.5] and
#      re-run. What happens to accuracy AND to the boundary's angle?
#
#   3. The misclassified points get p near 0.5. What would it mean if a
#      wrong point got p = 0.99 instead? Which is worse, and why does
#      cross-entropy punish that far harder than squared error would?
#
#   4. This data needs a bent boundary, not a straight one. That is the
#      same limitation as XOR in Chapter 4 -- and the same fix: a hidden
#      layer. You now have a concrete reason to want one.
# ----------------------------------------------------------------------
