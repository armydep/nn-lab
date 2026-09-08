"""Chapter 3, Task 2 -- when one parameter feeds two paths.

Every graph in the main exercise used each parameter exactly once, so each
gradient had exactly one route back from the loss. That made the chain rule
a single chain.

Here `w` and `b` are each used TWICE:

    z1 = w*x + b
    h  = tanh(z1)
    z2 = w*h + b        <- the same w and b again
    L  = (z2 - target)**2

Gradient flows back along BOTH routes and the two contributions must be
ADDED. Miss one and your gradient is wrong -- not crashed, just wrong, and
the model will still train, only worse.

This is the multivariable chain rule, and it is the reason autograd engines
accumulate with `+=` instead of assigning with `=`.

    python3 03-backpropagation/optional/t2.py
"""

import numpy as np

EPS = 1e-5

X = 1.3
TARGET = 0.7
W = 0.8
B = -0.2


def forward(x, w, b):
    """Return (z1, h, z2, loss) -- every intermediate, as in the main exercise."""
    z1 = w * x + b
    h = np.tanh(z1)
    z2 = w * h + b
    loss = (z2 - TARGET) ** 2
    return z1, h, z2, loss


def numeric_grad(f, value, eps=EPS):
    return (f(value + eps) - f(value - eps)) / (2 * eps)


# ----------------------------------------------------------------------
# YOUR TURN (1 of 3) -- the WRONG gradient, on purpose.
#
# Compute dL/dw counting ONLY the direct route through `z2 = w*h + b`,
# pretending h is a constant. This is the mistake the task is about, and
# you will confirm below that it fails the numeric check.
#
#   dL/dz2 = 2*(z2 - target)
#   dz2/dw = h              <- treating h as constant
# ----------------------------------------------------------------------
def dw_one_path(x, w, b):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 3) -- the correct dL/dw.
#
# Add the second route. w also reaches the loss through z1 -> h -> z2:
#
#   route A (direct):   dz2/dw = h
#   route B (via h):    dz2/dh * dh/dz1 * dz1/dw
#                     =    w   * (1-h^2) *    x
#
#   dL/dw = dL/dz2 * (route A + route B)
# ----------------------------------------------------------------------
def dw_both_paths(x, w, b):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 3) -- the correct dL/db.
#
# Same shape of argument. b reaches the loss directly (dz2/db = 1) and
# through z1 (dz2/dh * dh/dz1 * dz1/db, where dz1/db = 1).
# ----------------------------------------------------------------------
def db_both_paths(x, w, b):
    pass  # <- replace


def main():
    z1, h, z2, loss = forward(X, W, B)
    print(f"forward:  z1={z1:+.6f}  h={h:+.6f}  z2={z2:+.6f}  L={loss:.6f}\n")

    loss_wrt_w = lambda w_val: forward(X, w_val, B)[3]
    loss_wrt_b = lambda b_val: forward(X, W, b_val)[3]

    num_w = numeric_grad(loss_wrt_w, W)
    num_b = numeric_grad(loss_wrt_b, B)

    one = dw_one_path(X, W, B)
    both_w = dw_both_paths(X, W, B)
    both_b = db_both_paths(X, W, B)

    print(f"{'':22}{'analytic':>12}{'numeric':>12}{'':>8}")
    print(f"{'dL/dw (one path)':22}{one:>12.6f}{num_w:>12.6f}   <- wrong")
    print(f"{'dL/dw (both paths)':22}{both_w:>12.6f}{num_w:>12.6f}")
    print(f"{'dL/db (both paths)':22}{both_b:>12.6f}{num_b:>12.6f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   `dw_both_paths` and `db_both_paths` match the numeric gradients to
#   about six decimal places, and `dw_one_path` visibly does not.
#
# Then think about:
#
#   1. The one-path gradient is not slightly off -- it is a different
#      number. Gradient descent using it will often still reduce the loss,
#      just along the wrong direction. Why is a bug that still "works"
#      worse than one that crashes?
#
#   2. Which route contributes more here? Push W up to 3.0 and re-run:
#      tanh saturates, (1 - h^2) collapses toward zero, and the second
#      route almost vanishes. That is a vanishing gradient, in one line.
#
#   3. An autograd engine writes `parent.grad += local * out.grad`. If you
#      changed that `+=` to `=`, which of the two routes would survive --
#      and would you notice?
#
#   4. Chapter 5 builds that engine. This task is the reason its accumulation
#      step is written the way it is.
# ----------------------------------------------------------------------
