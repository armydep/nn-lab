"""Chapter 3, Task 2 solution -- one parameter, two paths.

    z1 = w*x + b  ->  h = tanh(z1)  ->  z2 = w*h + b  ->  L = (z2 - target)^2

w and b each appear twice, so each gradient is the SUM of two routes.

    python3 solutions/03-backpropagation/optional/t2.py
"""

from __future__ import annotations

import numpy as np

EPS = 1e-5

X = 1.3
TARGET = 0.7
W = 0.8
B = -0.2


def forward(x: float, w: float, b: float):
    z1 = w * x + b
    h = np.tanh(z1)
    z2 = w * h + b
    loss = (z2 - TARGET) ** 2
    return z1, h, z2, loss


def numeric_grad(f, value: float, eps: float = EPS) -> float:
    return (f(value + eps) - f(value - eps)) / (2 * eps)


def dw_one_path(x: float, w: float, b: float) -> float:
    """The mistake: count only the direct route, treating h as a constant."""
    _, h, z2, _ = forward(x, w, b)
    dL_dz2 = 2 * (z2 - TARGET)
    return dL_dz2 * h


def dw_both_paths(x: float, w: float, b: float) -> float:
    """Route A (direct) + route B (through z1 -> h -> z2)."""
    _, h, z2, _ = forward(x, w, b)
    dL_dz2 = 2 * (z2 - TARGET)

    route_a = h                          # dz2/dw, h held fixed
    route_b = w * (1 - h**2) * x         # dz2/dh * dh/dz1 * dz1/dw

    return dL_dz2 * (route_a + route_b)


def db_both_paths(x: float, w: float, b: float) -> float:
    _, h, z2, _ = forward(x, w, b)
    dL_dz2 = 2 * (z2 - TARGET)

    route_a = 1.0                        # dz2/db, h held fixed
    route_b = w * (1 - h**2) * 1.0       # dz2/dh * dh/dz1 * dz1/db

    return dL_dz2 * (route_a + route_b)


def main() -> None:
    z1, h, z2, loss = forward(X, W, B)
    print("=" * 62)
    print("TWO PATHS INTO ONE PARAMETER")
    print("=" * 62)
    print(f"forward:  z1={z1:+.6f}  h={h:+.6f}  z2={z2:+.6f}  L={loss:.6f}\n")

    num_w = numeric_grad(lambda v: forward(X, v, B)[3], W)
    num_b = numeric_grad(lambda v: forward(X, W, v)[3], B)

    one = dw_one_path(X, W, B)
    both_w = dw_both_paths(X, W, B)
    both_b = db_both_paths(X, W, B)

    print(f"{'':24}{'analytic':>12}{'numeric':>12}   verdict")
    print(f"{'dL/dw  direct route only':24}{one:>12.6f}{num_w:>12.6f}   WRONG")
    print(f"{'dL/dw  both routes':24}{both_w:>12.6f}{num_w:>12.6f}   ok")
    print(f"{'dL/db  both routes':24}{both_b:>12.6f}{num_b:>12.6f}   ok")

    # The routes, separately -- so the size of the omission is visible.
    dL_dz2 = 2 * (z2 - TARGET)
    print(f"\nroute A (direct)      {dL_dz2 * h:+.6f}")
    print(f"route B (through h)   {dL_dz2 * W * (1 - h**2) * X:+.6f}")
    print(f"sum                   {both_w:+.6f}")

    assert np.isclose(both_w, num_w, atol=1e-6), "dL/dw does not match finite differences"
    assert np.isclose(both_b, num_b, atol=1e-6), "dL/db does not match finite differences"
    assert not np.isclose(one, num_w, atol=1e-6), "the one-path version should FAIL the check"

    # Saturation: a large w pushes tanh flat, and route B disappears.
    w_big = 3.0
    _, h_big, z2_big, _ = forward(X, w_big, B)
    route_b_big = 2 * (z2_big - TARGET) * w_big * (1 - h_big**2) * X
    print(f"\nwith w = {w_big}:  h = {h_big:.6f},  route B = {route_b_big:+.8f}")
    assert abs(route_b_big) < abs(dL_dz2 * W * (1 - h**2) * X), "saturation should shrink route B"

    print("\nAll gradient checks passed.")
    print(
        "\nAnswers:\n"
        "  1. A gradient that is wrong but still points broadly downhill\n"
        "     produces a model that trains and underperforms, with no error\n"
        "     to trace. A crash tells you where to look; a silently wrong\n"
        "     gradient costs you weeks. This is why gradient checking is a\n"
        "     habit and not an occasional tool.\n\n"
        "  2. Route A dominates here. Raising w saturates tanh: h approaches\n"
        "     1, so (1 - h^2) approaches 0 and route B nearly vanishes --\n"
        "     printed above. That is the vanishing gradient, visible in a\n"
        "     three-node graph. Residual connections (Chapter 9) exist to\n"
        "     keep a route like A open through many layers.\n\n"
        "  3. With `=` instead of `+=`, whichever route the engine happens\n"
        "     to process last overwrites the other. The order depends on the\n"
        "     topological sort, so the bug is not even deterministic across\n"
        "     graph shapes -- and nothing raises an error.\n\n"
        "  4. Weight tying like this is not a toy case. Embedding matrices\n"
        "     shared with the output head, and any recurrent weight, receive\n"
        "     gradient from many paths at once."
    )


if __name__ == "__main__":
    main()
