"""Chapter 2, Task 3 -- break it on purpose: the log(0) trap.

Every guide tells you to clip probabilities before taking a log. Almost
nobody shows you the failure, so the advice never sticks.

Here you write BOTH versions -- unclipped and clipped -- and watch the
unclipped one turn into nan on data that is perfectly reasonable.

Breaking it deliberately, once, in a place you control, is much cheaper
than meeting it at midnight in Chapter 10.

    python3 02-binary-classifier/optional/t3.py
"""

import numpy as np

# Cleanly separable, and deliberately far apart. The model can drive its
# probabilities all the way to 0 and 1 -- which is exactly the problem.
X = np.array(
    [
        [0.5, 0.5], [1.0, 0.8], [0.7, 1.1], [1.2, 0.6],
        [6.0, 6.2], [6.5, 5.8], [5.9, 6.6], [6.3, 6.1],
    ]
)
Y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

EPS = 1e-12
LEARNING_RATE = 3.0      # large on purpose -- drives sigmoid into saturation
EPOCHS = 60


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ----------------------------------------------------------------------
# YOUR TURN (1 of 2) -- the naive version
#
# Write BCE exactly as the formula reads, with NO clipping:
#
#   L = -mean( y*log(p) + (1-y)*log(1-p) )
#
# This is the version almost everyone writes first. It is not stupid --
# it is the correct formula. It is just not safe in floating point.
# ----------------------------------------------------------------------
def bce_naive(p, y):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 2) -- the safe version
#
# Same thing, but clip p into [EPS, 1 - EPS] first, so neither log can
# ever receive exactly 0.
# ----------------------------------------------------------------------
def bce_safe(p, y):
    pass  # <- replace


def forward(x, w, b):
    return sigmoid(x @ w + b)


def gradients(x, y, w, b):
    dz = forward(x, w, b) - y          # dL/dz = p - y, from Chapter 2
    return x.T @ dz / len(y), np.mean(dz)


def train(loss_fn, label):
    """Train with the given loss function, reporting where it breaks."""
    w, b = np.zeros(2), 0.0
    print(f"\n{label}")
    print(f"  {'epoch':>6} {'loss':>14} {'min p':>12} {'max p':>12}")
    print("  " + "-" * 48)

    first_bad = None
    for epoch in range(EPOCHS + 1):
        p = forward(X, w, b)
        loss = loss_fn(p, Y)

        if epoch % 10 == 0 or (not np.isfinite(loss) and first_bad is None):
            loss_text = f"{loss:.8f}" if np.isfinite(loss) else "  nan/inf"
            print(f"  {epoch:>6} {loss_text:>14} {p.min():>12.2e} {1 - p.max():>12.2e}")

        if not np.isfinite(loss) and first_bad is None:
            first_bad = epoch
            print(f"  ^^^ loss stopped being a number at epoch {epoch}")

        dw, db = gradients(X, Y, w, b)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    return first_bad


if __name__ == "__main__":
    if bce_naive(np.full(8, 0.5), Y) is None or bce_safe(np.full(8, 0.5), Y) is None:
        raise SystemExit("Write bce_naive() and bce_safe() first.")

    # Both must agree on ordinary inputs -- clipping changes nothing there.
    ordinary = np.array([0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8, 0.1])
    naive_val, safe_val = bce_naive(ordinary, Y), bce_safe(ordinary, Y)
    print(f"On ordinary probabilities:  naive {naive_val:.6f}   safe {safe_val:.6f}")
    if not np.isclose(naive_val, safe_val):
        raise SystemExit("These should be identical -- check your implementations.")
    print("  Identical, as expected. Clipping is invisible until it matters.\n")

    # The direct demonstration.
    print("The single line that causes all of it:")
    with np.errstate(divide="ignore"):
        print(f"  np.log(0.0) = {np.log(0.0)}")
    print("  Any probability that reaches exactly 0.0 makes the loss infinite.")
    print("\n  Note the 'min p' column below: watch it hit 0.00e+00. Float64")
    print("  cannot represent sigmoid(-800) as anything but exactly zero.")

    bad_naive = train(bce_naive, "=== NAIVE (no clipping) ===")
    bad_safe = train(bce_safe, "=== SAFE (clipped) ===")

    print("\n" + "=" * 56)
    if bad_naive is not None and bad_safe is None:
        print(f"Naive broke at epoch {bad_naive}. Safe never broke.")
    elif bad_naive is None:
        print("Naive did not break this run -- raise LEARNING_RATE or EPOCHS.")
    print("=" * 56)
    print(
        "\nWhat happened: the data separates easily and the learning rate is\n"
        "large, so one step drove the weights up, so z grew, so sigmoid(z)\n"
        "saturated to exactly 1.0. Then log(1 - 1.0) = log(0) = -inf.\n\n"
        "Nothing was wrong with the model, the data, or the maths. The formula\n"
        "was right. Floating point simply cannot hold it.\n\n"
        "Now look again at the naive table -- and notice it RECOVERS. Explaining\n"
        "why is exercise 2, and it is the most useful thing in this file.\n\n"
        "This is why PyTorch ships BCEWithLogitsLoss and warns you off\n"
        "Sigmoid + BCELoss. See NUMERICS.md for the fused form that avoids\n"
        "forming p at all."
    )

# ----------------------------------------------------------------------
# Definition of done
#
#   The naive version produces nan, the safe version does not, and you can
#   explain the chain: big weights -> big z -> p = exactly 1.0 -> log(0).
#
# Then explore:
#
#   1. Lower LEARNING_RATE to 0.1. Does naive still break? Now you know why
#      this bug is so nasty: it hides at small learning rates and appears
#      the moment you tune upward.
#
#   2. Look carefully at the naive table -- the loss goes bad, and then
#      COMES BACK. By epoch 10 it is healthy again and matches the safe
#      version exactly. Why did training survive a nan loss?
#
#      Hint: which quantity actually went nan, and which one does the
#      weight update depend on? Then try starting from w = [nan, 0.0] and
#      see whether THAT ever recovers. The difference between those two
#      cases is the whole lesson.
#
#   3. Try EPS = 1e-300 instead of 1e-12. Does clipping still save you?
#      What does that say about how small a clip value can usefully be?
#
#   4. NUMERICS.md gives dL/dz = sigmoid(z) - y for the fused loss -- the
#      same expression this file already uses in gradients(). The GRADIENT
#      was never in danger. Only the loss VALUE was. Why does that mean a
#      model can sometimes keep training while its reported loss is nan?
# ----------------------------------------------------------------------
