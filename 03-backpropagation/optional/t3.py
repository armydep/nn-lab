"""Chapter 3, Task 3 -- hunt a planted bug, then find the right eps.

Two things the main exercise gave you for free.

PART 1. `buggy_gradients()` below computes four gradients, and exactly one
        of its local derivatives is wrong. You are not told which. Find it
        with numeric checking -- and notice that WHICH gradients fail tells
        you where the bug is, not merely that there is one.

PART 2. eps is a real choice, not a constant to copy. Too large and the
        difference quotient is a poor approximation of a curve. Too small
        and floating point destroys the answer. You will see both walls.

    python3 03-backpropagation/optional/t3.py
"""

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


def buggy_gradients(x, w1, b1, w2, b2):
    """Four gradients. One local derivative in here is wrong.

    Do not squint at it -- find it the way you would find it in code you
    did not write, by checking numerically.
    """
    z1, h, z2, _ = forward(x, w1, b1, w2, b2)

    dL_dz2 = 2 * (z2 - TARGET)
    dL_dw2 = dL_dz2 * h
    dL_db2 = dL_dz2

    dL_dh = dL_dz2 * w2
    dL_dz1 = dL_dh * (1 - h)
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    return {"w1": dL_dw1, "b1": dL_db1, "w2": dL_dw2, "b2": dL_db2}


PARAMS = {"w1": W1, "b1": B1, "w2": W2, "b2": B2}


# ----------------------------------------------------------------------
# YOUR TURN (1 of 4) -- central difference, with eps as an argument.
#
#   (f(v + eps) - f(v - eps)) / (2 * eps)
#
# Part 2 varies eps over twelve orders of magnitude, so it must be a
# parameter here, not a module constant.
# ----------------------------------------------------------------------
def numeric_grad(f, value, eps=1e-5):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 4) -- numeric gradients for all four parameters.
#
# Return {"w1": ..., "b1": ..., "w2": ..., "b2": ...}.
#
# The awkward part is varying ONE parameter while holding the other three
# fixed. A closure over the parameter name does it:
#
#   def f(v, name=name):
#       kwargs = dict(PARAMS, **{name: v})
#       return loss_only(X, **kwargs)
#
# The `name=name` default is not decoration -- without it every closure
# captures the same final value of the loop variable.
# ----------------------------------------------------------------------
def numeric_all(eps=1e-5):
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 4) -- compare, and report which parameters fail.
#
# Print analytic vs numeric per parameter and return the set of names that
# disagree beyond atol=1e-6. Before you look at the answer: predict which
# ones will fail, given a bug somewhere in the tanh step.
# ----------------------------------------------------------------------
def part_one():
    pass  # <- replace, return the set of failing names


# ----------------------------------------------------------------------
# YOUR TURN (4 of 4) -- sweep eps from 1e-1 down to 1e-12.
#
# Fix the bug you found, take the now-correct analytic dL/dw1 as the exact
# value, and for each eps print the numeric estimate and the relative
# error |approx - exact| / |exact|.
#
# Expect a U: error falls as eps shrinks, reaches a floor, then rises
# again. Both sides of that U have a different cause -- name them.
# ----------------------------------------------------------------------
def part_two():
    pass  # <- replace, return the best eps


def main():
    part_one()
    part_two()


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   You identified the single wrong local derivative from the pattern of
#   failures, and your eps sweep shows a clear U-shaped error curve.
#
# Then think about:
#
#   1. w2 and b2 check out while w1 and b1 do not. In the backward walk,
#      which nodes come before the bug and which come after? Convince
#      yourself you could have located the bug from that pattern alone,
#      without reading the code.
#
#   2. The planted bug uses (1 - h) where tanh needs (1 - h**2). Those two
#      agree exactly at h = 0. What does that mean for a network whose
#      weights start near zero and grow during training?
#
#   3. Name the two error sources at the ends of the U. One is about
#      approximating a curve with a straight line; the other is about
#      subtracting two nearly equal floats. Which is which?
#
#   4. Given that curve, why does every gradient check compare with a
#      tolerance instead of testing for equality?
# ----------------------------------------------------------------------
