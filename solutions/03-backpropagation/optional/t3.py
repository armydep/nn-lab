"""Chapter 3, Task 3 solution -- find the planted bug, then tune eps.

Part 1: four analytic gradients, one wrong local derivative. Numeric
        checking says which are wrong; the PATTERN says where the bug is.
Part 2: eps is not free. Too large and the difference quotient is a bad
        approximation; too small and floating point cancels the signal.

    python3 solutions/03-backpropagation/optional/t3.py
"""

from __future__ import annotations

import numpy as np

X = 0.9
TARGET = 0.4
W1, B1, W2, B2 = 0.6, -0.3, 1.4, 0.2


def forward(x, w1, b1, w2, b2):
    z1 = w1 * x + b1
    h = np.tanh(z1)
    z2 = w2 * h + b2
    loss = (z2 - TARGET) ** 2
    return z1, h, z2, loss


def loss_only(x, w1, b1, w2, b2):
    return forward(x, w1, b1, w2, b2)[3]


def numeric_grad(f, value, eps=1e-5):
    return (f(value + eps) - f(value - eps)) / (2 * eps)


def buggy_gradients(x, w1, b1, w2, b2):
    """Contains exactly ONE incorrect local derivative. Do not read past
    the return statement until you have found it with numeric checking."""
    z1, h, z2, _ = forward(x, w1, b1, w2, b2)

    dL_dz2 = 2 * (z2 - TARGET)
    dL_dw2 = dL_dz2 * h
    dL_db2 = dL_dz2

    dL_dh = dL_dz2 * w2
    dL_dz1 = dL_dh * (1 - h)        # <-- BUG: d(tanh)/dz is 1 - h**2, not 1 - h
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    return {"w1": dL_dw1, "b1": dL_db1, "w2": dL_dw2, "b2": dL_db2}


def correct_gradients(x, w1, b1, w2, b2):
    z1, h, z2, _ = forward(x, w1, b1, w2, b2)

    dL_dz2 = 2 * (z2 - TARGET)
    dL_dw2 = dL_dz2 * h
    dL_db2 = dL_dz2

    dL_dh = dL_dz2 * w2
    dL_dz1 = dL_dh * (1 - h**2)     # the real derivative of tanh
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    return {"w1": dL_dw1, "b1": dL_db1, "w2": dL_dw2, "b2": dL_db2}


PARAMS = {"w1": W1, "b1": B1, "w2": W2, "b2": B2}


def numeric_all(eps=1e-5):
    out = {}
    for name, value in PARAMS.items():
        def f(v, name=name):
            kwargs = dict(PARAMS, **{name: v})
            return loss_only(X, **kwargs)
        out[name] = numeric_grad(f, value, eps)
    return out


def part_one() -> set[str]:
    print("=" * 62)
    print("PART 1 -- find the planted bug")
    print("=" * 62)

    buggy = buggy_gradients(X, **PARAMS)
    numeric = numeric_all()

    print(f"{'param':>6}{'analytic':>14}{'numeric':>14}{'':>10}")
    wrong = set()
    for name in PARAMS:
        ok = np.isclose(buggy[name], numeric[name], atol=1e-6)
        if not ok:
            wrong.add(name)
        print(f"{name:>6}{buggy[name]:>14.6f}{numeric[name]:>14.6f}"
              f"{'  ok' if ok else '  WRONG':>10}")

    print(f"\nwrong: {sorted(wrong)}")
    print("w2 and b2 are correct, w1 and b1 are not.")
    print("Everything downstream of z2 is fine; everything that flows")
    print("through tanh is broken -- so the bug is dh/dz1, and nowhere else.")

    assert wrong == {"w1", "b1"}, f"expected w1 and b1 to fail, got {sorted(wrong)}"

    fixed = correct_gradients(X, **PARAMS)
    for name in PARAMS:
        assert np.isclose(fixed[name], numeric[name], atol=1e-6), f"{name} still wrong"
    print("\nAll four gradients match once dh/dz1 = 1 - h**2 is restored.")
    return wrong


def part_two() -> float:
    print("\n" + "=" * 62)
    print("PART 2 -- choosing eps")
    print("=" * 62)

    exact = correct_gradients(X, **PARAMS)["w1"]

    def f(v):
        return loss_only(X, v, B1, W2, B2)

    print(f"analytic dL/dw1 = {exact:.12f}\n")
    print(f"{'eps':>10}{'numeric':>20}{'relative error':>18}")

    results = []
    for power in range(1, 13):
        eps = 10.0 ** (-power)
        approx = numeric_grad(f, W1, eps)
        rel = abs(approx - exact) / abs(exact)
        results.append((eps, rel))
        print(f"{eps:>10.0e}{approx:>20.12f}{rel:>18.2e}")

    best_eps, best_err = min(results, key=lambda r: r[1])
    worst_large = results[0][1]     # eps = 1e-1, truncation error
    worst_small = results[-1][1]    # eps = 1e-12, cancellation error

    print(f"\nbest eps = {best_eps:.0e}  (relative error {best_err:.2e})")
    print(f"  eps too large (1e-1):  error {worst_large:.2e}  -- the quotient")
    print("                                       approximates a curve badly")
    print(f"  eps too small (1e-12): error {worst_small:.2e}  -- L(w+eps) and")
    print("                                       L(w-eps) round to nearly")
    print("                                       the same float; the")
    print("                                       difference is mostly noise")

    assert worst_large > best_err * 100, "large eps should be clearly worse"
    assert worst_small > best_err * 100, "tiny eps should be clearly worse"
    assert 1e-8 <= best_eps <= 1e-3, f"unexpected sweet spot: {best_eps:.0e}"
    return best_eps


def main() -> None:
    part_one()
    best = part_two()

    print(
        "\nAnswers:\n"
        "  1. Which gradients failed tells you WHERE the bug is, not just\n"
        "     that one exists. Gradients computed before the faulty node in\n"
        "     the backward walk are fine; everything after it is wrong. Read\n"
        "     the pattern and you have localised the bug without a debugger.\n\n"
        "  2. 1 - h and 1 - h**2 agree at h = 0 and diverge as |h| grows, so\n"
        "     a bug like this can look almost right near initialisation and\n"
        "     get worse as the network trains. That is the worst failure\n"
        "     profile there is.\n\n"
        f"  3. The error curve is U-shaped and bottoms out around {best:.0e}.\n"
        "     Large eps: the central difference approximates the tangent of a\n"
        "     curve over too wide an interval -- error falls as eps^2.\n"
        "     Tiny eps: L(w+eps) and L(w-eps) differ in their last few bits,\n"
        "     so subtracting them cancels almost all the significant digits\n"
        "     and you divide the surviving noise by a tiny number.\n\n"
        "  4. This is why 1e-5 is the conventional default, and why gradient\n"
        "     checks compare with a tolerance rather than for equality."
    )


if __name__ == "__main__":
    main()
