"""Chapter 3 solution: the chain rule on a computation graph.

Three steps: one neuron by hand, finite-difference checking, then a two-layer
graph -- the exact structure Chapter 4 trains with matrices.

    python3 solutions/03-backpropagation/gradients.py
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-5


def numeric_grad(f: Callable[[float], float], value: float, eps: float = EPS) -> float:
    """Central finite difference: (f(v+eps) - f(v-eps)) / (2*eps).

    Central beats forward difference ((f(v+eps) - f(v))/eps): its error shrinks
    as eps^2 rather than eps, so it is far more accurate for the same eps.
    """
    return (f(value + eps) - f(value - eps)) / (2 * eps)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


# ======================================================================
# Step 1 + 2 -- one neuron, checked numerically
# ======================================================================
# Graph:  z = w*x + b  ->  y = sigmoid(z)  ->  L = (y - target)^2


def step_one() -> None:
    print("=" * 62)
    print("STEP 1 -- one neuron")
    print("=" * 62)

    x, w, b, target = 2.0, 0.5, -0.3, 1.0

    # Forward pass -- store EVERY intermediate. The backward pass needs them.
    z = w * x + b
    y = sigmoid(z)
    loss = (y - target) ** 2
    print(f"forward:  z = {z:+.6f}   y = {y:.6f}   L = {loss:.6f}")

    # Backward pass, one node at a time, right to left.
    dL_dy = 2 * (y - target)        # local derivative of (y-target)^2
    dy_dz = y * (1 - y)             # local derivative of sigmoid
    dz_dw = x                       # local derivative of w*x + b w.r.t. w
    dz_db = 1.0                     # ... w.r.t. b

    # Chain: multiply along the path from L back to each parameter.
    dL_dz = dL_dy * dy_dz
    dL_dw = dL_dz * dz_dw
    dL_db = dL_dz * dz_db

    print(f"\nlocal derivatives:")
    print(f"  dL/dy = {dL_dy:+.6f}    dy/dz = {dy_dz:+.6f}")
    print(f"  dz/dw = {dz_dw:+.6f}    dz/db = {dz_db:+.6f}")
    print(f"\nchained:  dL/dw = {dL_dw:+.6f}   dL/db = {dL_db:+.6f}")

    # Step 2 -- verify numerically.
    def loss_wrt_w(w_val: float) -> float:
        return (sigmoid(w_val * x + b) - target) ** 2

    def loss_wrt_b(b_val: float) -> float:
        return (sigmoid(w * x + b_val) - target) ** 2

    num_w = numeric_grad(loss_wrt_w, w)
    num_b = numeric_grad(loss_wrt_b, b)
    print(f"numeric:  dL/dw = {num_w:+.6f}   dL/db = {num_b:+.6f}")

    assert np.isclose(dL_dw, num_w, atol=1e-6)
    assert np.isclose(dL_db, num_b, atol=1e-6)
    print("\n  match -- the derivation is correct")


# ======================================================================
# Step 3 -- two layers
# ======================================================================
# Graph:
#   z1 = w1*x + b1  ->  h = tanh(z1)  ->  z2 = w2*h + b2  ->  L = (z2-target)^2


def two_layer_forward(
    x: float, w1: float, b1: float, w2: float, b2: float, target: float
) -> tuple[float, float, float, float]:
    z1 = w1 * x + b1
    h = np.tanh(z1)
    z2 = w2 * h + b2
    loss = (z2 - target) ** 2
    return z1, h, z2, loss


def step_three() -> None:
    print("\n" + "=" * 62)
    print("STEP 3 -- two layers (the shape of Chapter 4)")
    print("=" * 62)

    x, target = 1.5, 0.8
    w1, b1, w2, b2 = 0.4, -0.2, 0.9, 0.1

    z1, h, z2, loss = two_layer_forward(x, w1, b1, w2, b2, target)
    print(f"forward:  z1 = {z1:+.6f}  h = {h:+.6f}  z2 = {z2:+.6f}  L = {loss:.6f}")

    # Backward pass. Each step multiplies the gradient flowing in from above
    # by the local derivative at this node.
    dL_dz2 = 2 * (z2 - target)          # through the squared error
    dL_dw2 = dL_dz2 * h                 # z2 = w2*h + b2, so dz2/dw2 = h
    dL_db2 = dL_dz2 * 1.0

    dL_dh = dL_dz2 * w2                 # gradient continues into the hidden unit
    dL_dz1 = dL_dh * (1 - h**2)         # d/dz tanh(z) = 1 - tanh(z)^2
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1 * 1.0

    print("\nbackward, right to left:")
    print(f"  dL/dz2 = {dL_dz2:+.6f}   ->  dL/dw2 = {dL_dw2:+.6f}   dL/db2 = {dL_db2:+.6f}")
    print(f"  dL/dh  = {dL_dh:+.6f}   (dL/dz2 * w2)")
    print(f"  dL/dz1 = {dL_dz1:+.6f}   (dL/dh * (1 - h^2))")
    print(f"                          ->  dL/dw1 = {dL_dw1:+.6f}   dL/db1 = {dL_db1:+.6f}")

    print(
        "\n  THIS is backpropagation: dL/dw1 was computed by REUSING dL/dz2,\n"
        "  the gradient already calculated for the layer above. Nothing was\n"
        "  recomputed. That reuse is the entire efficiency of the algorithm --\n"
        "  it is what makes training networks with millions of parameters\n"
        "  cost about the same as one extra forward pass."
    )

    # Gradient-check all four parameters.
    params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    analytic = {"w1": dL_dw1, "b1": dL_db1, "w2": dL_dw2, "b2": dL_db2}

    print("\ngradient check:")
    for name in params:
        def loss_fn(value: float, name: str = name) -> float:
            kwargs = dict(params)
            kwargs[name] = value
            return two_layer_forward(x, target=target, **kwargs)[3]

        num = numeric_grad(loss_fn, params[name])
        ok = np.isclose(analytic[name], num, atol=1e-6)
        print(f"  {name}:  analytic {analytic[name]:+.6f}   numeric {num:+.6f}   {'OK' if ok else 'MISMATCH'}")
        assert ok, f"gradient for {name} is wrong"

    print("\n  all four match")


def main() -> None:
    step_one()
    step_three()
    print("\n" + "=" * 62)
    print("Why the backward pass runs right-to-left:")
    print("=" * 62)
    print(
        "  Each node needs the gradient from the node ABOVE it before it can\n"
        "  compute its own. dL/dw1 depends on dL/dz1, which depends on dL/dh,\n"
        "  which depends on dL/dz2. Start at the left and you have nothing to\n"
        "  multiply by.\n\n"
        "  And the forward pass must STORE its intermediates, because the\n"
        "  local derivatives are expressed in terms of them: dy/dz = y*(1-y)\n"
        "  needs y; dh/dz1 = 1 - h^2 needs h. This is exactly why training\n"
        "  uses far more memory than inference -- every intermediate has to\n"
        "  be kept alive until the backward pass consumes it."
    )


if __name__ == "__main__":
    main()
