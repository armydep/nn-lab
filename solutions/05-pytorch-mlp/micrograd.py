"""Chapter 5 Part A solution: a minimal autograd engine.

The one new idea versus Chapters 3-4: the graph RECORDS itself as the forward
pass runs, so the backward order can be worked out automatically instead of
being hardcoded by you.

    python3 solutions/05-pytorch-mlp/micrograd.py
"""

from __future__ import annotations

import math


class Value:
    """A scalar that remembers how it was computed."""

    def __init__(self, data: float, _parents: tuple[Value, ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        # How to push gradient from this node to its parents. Overwritten by
        # each operation below; a leaf node has nothing to push.
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # Addition passes gradient through unchanged to both parents.
            # NOTE the +=, not =. If a value is used twice, gradient arrives
            # from both paths and must ACCUMULATE. Using = silently discards
            # one path -- the classic autograd bug.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, and vice versa.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # Convenience operators so expressions read naturally.
    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def backward(self) -> None:
        """Topologically sort the graph, then walk it backwards."""
        order: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            order.append(node)      # appended only after all parents

        build(self)

        # Seed: dL/dL = 1.
        self.grad = 1.0
        # Reversed topological order guarantees a node's own gradient is
        # complete before it pushes anything to its parents. Process them in
        # the wrong order and a node pushes a partial gradient downstream --
        # every gradient below it is then silently wrong.
        for node in reversed(order):
            node._backward()


def main() -> None:
    print("=" * 62)
    print("Rebuilding Chapter 3's two-layer graph with Value objects")
    print("=" * 62)

    # Identical numbers to solutions/03-backpropagation/gradients.py step 3.
    x, target = 1.5, 0.8
    w1, b1 = Value(0.4), Value(-0.2)
    w2, b2 = Value(0.9), Value(0.1)

    # Forward pass -- written as ordinary arithmetic. The graph builds itself.
    z1 = w1 * x + b1
    h = z1.tanh()
    z2 = w2 * h + b2
    loss = (z2 - target) ** 2

    print(f"forward:  z1 = {z1.data:+.6f}  h = {h.data:+.6f}  z2 = {z2.data:+.6f}")
    print(f"          L  = {loss.data:.6f}")

    # One call replaces the entire hand-written backward pass.
    loss.backward()

    print("\nGradients from the engine:")
    print(f"  dL/dw1 = {w1.grad:+.6f}   dL/db1 = {b1.grad:+.6f}")
    print(f"  dL/dw2 = {w2.grad:+.6f}   dL/db2 = {b2.grad:+.6f}")

    # These are the numbers Chapter 3 derived by hand.
    expected = {"w1": -0.827167, "b1": -0.551444, "w2": -0.272078, "b2": -0.716092}
    print("\nChapter 3 hand-derived:")
    print(f"  dL/dw1 = {expected['w1']:+.6f}   dL/db1 = {expected['b1']:+.6f}")
    print(f"  dL/dw2 = {expected['w2']:+.6f}   dL/db2 = {expected['b2']:+.6f}")

    for name, value in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
        assert abs(value.grad - expected[name]) < 1e-5, f"{name} mismatch"
    print("\n  match -- the engine reproduces the hand derivation")

    # Demonstrate why accumulation matters.
    print("\n" + "=" * 62)
    print("Why gradients must accumulate (+= not =)")
    print("=" * 62)
    a = Value(3.0)
    y = a * a                  # 'a' is used TWICE in one expression
    y.backward()
    print(f"  y = a * a  with a = {a.data}")
    print(f"  dy/da = {a.grad:+.1f}   (correct: 2a = 6.0)")
    print(
        "  Gradient arrives once from each use of 'a'. With = instead of +=,\n"
        "  the second assignment overwrites the first and you get 3.0 -- half\n"
        "  the true gradient, with no error raised anywhere.\n\n"
        "  This is also why PyTorch needs optimizer.zero_grad(): accumulation\n"
        "  is the correct default, so gradients from the PREVIOUS step must be\n"
        "  cleared explicitly or they sum across steps."
    )


if __name__ == "__main__":
    main()
