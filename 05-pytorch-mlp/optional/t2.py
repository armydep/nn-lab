"""Chapter 5, Task 2 -- extend the engine, and watch it re-derive p - y.

Part A gave you an engine with +, *, ** and tanh. Real engines grow by
adding operations, and each operation is exactly two things: how to
compute it forwards, and how to push gradient to its parents.

You will add exp and log, then build sigmoid and binary cross-entropy out
of primitives ONLY -- no hand-derived shortcut anywhere. The engine should
then reproduce Chapter 2's result, dL/dz = p - y, without ever being told
it.

That is the moment autograd stops being a convenience and starts being an
answer to "where do gradients come from".

    python3 05-pytorch-mlp/optional/t2.py
"""

from __future__ import annotations

import math

class Value:
    """A scalar that remembers how it was computed."""

    def __init__(self, data: float, _parents: tuple["Value", ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value({self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float):
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # YOUR TURN (1 of 4) -- exp
    #
    # forward:  math.exp(self.data)
    # backward: d(e^x)/dx = e^x, so the local derivative IS the output.
    #           Capture it in the closure and accumulate with +=.
    # ------------------------------------------------------------------
    def exp(self):
        pass  # <- replace

    # ------------------------------------------------------------------
    # YOUR TURN (2 of 4) -- log
    #
    # forward:  math.log(self.data)
    # backward: d(ln x)/dx = 1/x. Note this needs the INPUT, where exp
    #           needed the output. Both are just values captured during
    #           the forward pass -- Chapter 3's "store every intermediate",
    #           automated.
    # ------------------------------------------------------------------
    def log(self):
        pass  # <- replace

    # --- conveniences, all built on the four operations above ----------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def backward(self):
        order, seen = [], set()

        def build(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                build(parent)
            order.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
        return order


# ----------------------------------------------------------------------
# YOUR TURN (3 of 4) -- sigmoid and binary cross-entropy, from primitives.
#
#   sigmoid(z) = 1 / (1 + exp(-z))     ->  (1 + (-z).exp()) ** -1
#   bce(p, y)  = -(y*log(p) + (1-y)*log(1-p))
#
# Do NOT shortcut either of these. The whole point is that the engine
# composes local derivatives and lands on p - y by itself.
# ----------------------------------------------------------------------
def sigmoid(z):
    pass  # <- replace


def bce(p, y):
    pass  # <- replace


def numeric_grad(f, value, eps=1e-6):
    return (f(value + eps) - f(value - eps)) / (2 * eps)


# ----------------------------------------------------------------------
# YOUR TURN (4 of 4) -- the three checks.
#
# 1. exp and log gradient-check against finite differences. Include one
#    case where the SAME leaf feeds two branches, e.g. x.exp() * x.log():
#
#       x = Value(x0)
#       out = build(x)          # build receives the leaf, not a number
#       out.backward()
#       compare x.grad with numeric_grad(lambda v: build(Value(v)).data, x0)
#
#    Building two separate Value(x0) objects would silently test something
#    else -- two independent inputs, each with one path.
#
# 2. For z in (-2, -0.5, 0, 0.9, 2.5) and y in (0, 1): build
#    bce(sigmoid(z), y), call backward, and confirm z.grad equals p - y.
#
# 3. Set y = 0.5 so BOTH log terms are live. Now p feeds the loss twice.
#    Confirm the gradient still matches finite differences -- this is the
#    case that only works because every _backward accumulates.
# ----------------------------------------------------------------------
def main():
    pass  # <- replace


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   exp and log gradient-check, z.grad equals p - y to nine decimal places
#   across all ten (z, y) combinations, and the y = 0.5 reuse case checks
#   out too.
#
# Then think about:
#
#   1. You added two operations without touching backward(). What is the
#      contract an operation must satisfy for that to work?
#
#   2. exp stores its output for the backward pass, log stores its input.
#      Look at your two closures and say which value each captured, and
#      why the choice is forced.
#
#   3. Nowhere in this file is p - y written down as a rule, yet it comes
#      out exactly. So why do PyTorch and every other framework still
#      special-case cross-entropy instead of composing it like you just
#      did? (NUMERICS.md has the answer.)
#
#   4. Change every += to = in the engine and re-run. Which of the three
#      checks fails, and which still pass? That is what a weak test suite
#      looks like from the inside.
# ----------------------------------------------------------------------
