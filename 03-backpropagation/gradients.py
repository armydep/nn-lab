"""Chapter 3 starter: the chain rule on a computation graph.

Work through the three steps in order. Do not skip the gradient checks --
they are what tell you your derivatives are right.
"""

# --- Step 1: one neuron ------------------------------------------------
# Graph:  z = w*x + b  ->  y = sigmoid(z)  ->  L = (y - target)**2
#
# TODO: pick concrete x, w, b, target
# TODO: forward pass, storing every intermediate (z, y, L)
# TODO: local derivatives: dL/dy, dy/dz, dz/dw, dz/db
# TODO: chain them into dL/dw and dL/db


# --- Step 2: finite-difference gradient check --------------------------
# numeric = (L(w + eps) - L(w - eps)) / (2 * eps),  eps = 1e-5
#
# TODO: write a reusable numeric_grad(f, value) helper
# TODO: confirm analytic and numeric agree to ~6 decimal places


# --- Step 3: two layers ------------------------------------------------
# Graph:  z1 = w1*x + b1 -> h = tanh(z1) -> z2 = w2*h + b2 -> L = (z2 - target)**2
#
# TODO: forward pass, storing every intermediate
# TODO: backward pass node by node, from L back to w1
# TODO: notice dL/dw1 reuses the gradient already computed for w2 and tanh
# TODO: gradient-check all four parameters
