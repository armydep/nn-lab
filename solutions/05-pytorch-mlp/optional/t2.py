"""Chapter 5, Task 2 solution -- extend the engine, and let it re-derive p - y.

Part A gave you an engine with +, *, ** and tanh. Real engines grow by
adding operations, and each new operation is exactly two things: how to
compute it forwards, and how to push gradient backwards.

Here you add exp and log, then build sigmoid and binary cross-entropy out
of primitives only -- no hand-derived shortcut anywhere. The engine then
reproduces Chapter 2's result, dL/dz = p - y, without being told it.

    python3 solutions/05-pytorch-mlp/optional/t2.py
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

    def exp(self):
        """d(e^x)/dx = e^x -- the output itself is the local derivative."""
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """d(ln x)/dx = 1/x -- note it uses the INPUT, not the output."""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

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


def sigmoid(z: Value) -> Value:
    """1 / (1 + exp(-z)) -- from primitives, so the engine derives it."""
    return (1 + (-z).exp()) ** -1


def bce(p: Value, y: float) -> Value:
    """-(y*log(p) + (1-y)*log(1-p)) -- also from primitives."""
    return -(y * p.log() + (1 - y) * (1 - p).log())


def numeric_grad(f, value: float, eps: float = 1e-6) -> float:
    return (f(value + eps) - f(value - eps)) / (2 * eps)


def check_exp_and_log() -> None:
    print("=" * 62)
    print("PART 1 -- the two new operations")
    print("=" * 62)

    # Each builder receives ONE leaf, so the third case genuinely shares x
    # between two branches -- a gradient that only comes out right because
    # every _backward accumulates.
    for label, build, x0 in (
        ("exp(x)", lambda x: x.exp(), 0.7),
        ("log(x)", lambda x: x.log(), 2.3),
        ("exp(x)*log(x)", lambda x: x.exp() * x.log(), 1.7),
    ):
        x = Value(x0)
        out = build(x)
        out.backward()
        numeric = numeric_grad(lambda v: build(Value(v)).data, x0)
        print(f"{label:>16}  engine {x.grad:+.9f}   numeric {numeric:+.9f}")
        assert abs(x.grad - numeric) < 1e-5, f"{label} gradient is wrong"

    print("\nBoth operations gradient-check.")


def check_cross_entropy() -> None:
    print("\n" + "=" * 62)
    print("PART 2 -- the engine re-derives dL/dz = p - y")
    print("=" * 62)
    print(f"{'z':>8}{'y':>5}{'p':>12}{'engine dL/dz':>16}{'p - y':>12}{'numeric':>14}")

    for z0 in (-2.0, -0.5, 0.0, 0.9, 2.5):
        for y in (0.0, 1.0):
            z = Value(z0)
            loss = bce(sigmoid(z), y)
            loss.backward()

            p = sigmoid(Value(z0)).data
            expected = p - y
            numeric = numeric_grad(lambda v: bce(sigmoid(Value(v)), y).data, z0)

            print(f"{z0:>8.1f}{y:>5.0f}{p:>12.6f}{z.grad:>16.9f}"
                  f"{expected:>12.6f}{numeric:>14.6f}")

            assert abs(z.grad - expected) < 1e-9, "engine disagrees with p - y"
            assert abs(z.grad - numeric) < 1e-5, "engine disagrees with finite differences"

    print("\nThe engine was never told the p - y shortcut. It composed the")
    print("local derivatives of exp, log, +, * and ** and arrived there anyway.")


def check_reuse() -> None:
    print("\n" + "=" * 62)
    print("PART 3 -- one input, many paths")
    print("=" * 62)

    z = Value(0.6)
    p = sigmoid(z)
    # z reaches the loss through a long chain, and p is used TWICE inside
    # bce (once as log(p), once as log(1-p)).
    loss = bce(p, 0.5)
    order = loss.backward()

    numeric = numeric_grad(lambda v: bce(sigmoid(Value(v)), 0.5).data, 0.6)
    print(f"nodes in graph {len(order)}")
    print(f"engine dL/dz {z.grad:+.9f}   numeric {numeric:+.9f}")
    assert abs(z.grad - numeric) < 1e-5, "reuse path gradient is wrong"
    print("\nWith y = 0.5 both log terms are live, so p feeds the loss twice.")
    print("The += in every _backward is what makes this come out right.")


def main() -> None:
    check_exp_and_log()
    check_cross_entropy()
    check_reuse()

    print(
        "\nAnswers:\n"
        "  1. An operation needs exactly two things: a forward value, and a\n"
        "     rule for pushing gradient to its parents. That is the entire\n"
        "     contract -- which is why engines can be extended without\n"
        "     touching backward() itself.\n\n"
        "  2. exp captures its OUTPUT for the backward pass (d(e^x)/dx = e^x)\n"
        "     while log needs its INPUT (d(ln x)/dx = 1/x). Both are closures\n"
        "     over values from the forward pass -- this is precisely the\n"
        "     'store every intermediate' rule from Chapter 3, automated.\n\n"
        "  3. p - y is not built into the engine anywhere. It emerges from\n"
        "     composing local derivatives. Frameworks special-case it for\n"
        "     numerical stability and speed, not because the chain rule\n"
        "     needs help.\n\n"
        "  4. With y = 0.5 both log terms are active, so p has two paths to\n"
        "     the loss. Change every += to = in this file and Part 3 breaks\n"
        "     while Parts 1 and 2 may still pass -- exactly the kind of bug\n"
        "     that survives a weak test suite."
    )


if __name__ == "__main__":
    main()
