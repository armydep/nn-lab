# Chapter 3 — Backpropagation & Computation Graphs

## Goal

Learn the chain rule as a mechanical procedure on a computation graph, and build up to a
**two-layer** graph — the exact structure you will train in Chapter 4.

## Why this chapter comes before the MLP

Chapter 4 asks you to derive gradients for a network with a hidden layer. That is
backpropagation. If you have not met the chain rule yet, Chapter 4 is not a challenge, it is a
wall. This chapter gives you the one pattern that makes Chapter 4 mechanical:

```text
gradient at a node = (gradient flowing in from above) × (local derivative at this node)
```

Everything else in this repository is that sentence, repeated.

## Concepts

Computation graph, node, local derivative, upstream gradient, chain rule, backward pass,
finite-difference gradient checking.

## Implementation requirements

Work in three steps, and verify each one before moving on.

### Step 1 — One neuron, by hand

For `z = w*x + b`, `y = sigmoid(z)`, `L = (y - target)^2`:

1. Pick concrete numbers for `x`, `w`, `b`, `target`.
2. Compute the forward pass, storing **every** intermediate value (`z`, `y`, `L`).
3. Write down the local derivative at each node: `dL/dy`, `dy/dz`, `dz/dw`, `dz/db`.
4. Multiply them along the graph to get `dL/dw` and `dL/db`.

### Step 2 — Check yourself with finite differences

You never have to wonder whether your derivative is right. Nudge a parameter and measure:

```text
numeric_grad = (L(w + eps) - L(w - eps)) / (2 * eps)     # eps = 1e-5
```

If your analytic gradient and this number agree to ~6 decimal places, your derivative is
correct. Use this in every later chapter when a network will not learn — it is the single most
useful debugging tool in this repository.

### Step 3 — Two layers

Now extend the graph to have a hidden unit:

```text
z1 = w1*x + b1  →  h = tanh(z1)  →  z2 = w2*h + b2  →  L = (z2 - target)^2
```

1. Forward pass, storing every intermediate.
2. Backward pass, one node at a time, from `L` back to `w1`.
3. Observe that `dL/dw1` needs the gradient that already flowed back through `w2` and `tanh`.
   **This is the whole idea of backpropagation:** gradients reuse the work of the layer above.
4. Gradient-check all four parameters.

## The shape rule you will need in Chapter 4

Chapter 4 uses matrices instead of scalars, but nothing conceptually changes. One rule carries
you across:

> **A gradient always has the same shape as the thing it is the gradient of.**

If `W1` is `(2, 4)`, then `dW1` is `(2, 4)`. When you are unsure whether an expression should
be `X.T @ D` or `D @ X.T`, pick the one that produces the right shape — it is almost always
the correct answer.

## Definition of done

- Your analytic gradients match finite differences for all four parameters of the two-layer
  graph.
- You can point at any node and say what its local derivative is and what gradient flows into
  it.
- You can explain why the backward pass runs right-to-left, and why the forward pass has to
  store intermediates for it to work.

## Optional practice

Extra drills in `optional/`. The main exercise is the chapter; these are for more repetitions.

| File | New idea |
|---|---|
| `optional/t2.py` | One parameter used twice — gradient arrives by two routes and must be summed |
| `optional/t3.py` | Find a planted bug from the pattern of failures, then see why `eps` has a sweet spot |

## Reference solutions

`solutions/03-backpropagation/gradients.py` — and `optional/` for the drills. Run them after you
have attempted your own.

## Further reading

Olah, [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/), and Karpathy's [micrograd video](https://www.youtube.com/watch?v=VMj-3vsdGqc)

Full list: [`RESOURCES.md`](../RESOURCES.md)
