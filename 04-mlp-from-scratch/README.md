# Chapter 4 — MLP From Scratch

## Goal

Build a two-layer neural network that learns XOR, deriving every gradient yourself with NumPy.

## What you are carrying in from Chapter 3

This chapter is Chapter 3's two-layer graph, with three changes:

| Chapter 3 | Chapter 4 |
|---|---|
| Scalars `w1`, `b1` | Matrices `W1`, `b1` |
| One example | Four examples at once (a batch) |
| Squared error on one number | Loss averaged over the batch |

The chain rule does not change. Only the shapes do. When a gradient expression looks
ambiguous, apply the shape rule from Chapter 3: **a gradient has the same shape as the thing it
differentiates**.

## Concepts

Hidden layers, matrices, nonlinear activations, forward propagation, batching, representational
power.

## Implementation requirements

1. Use the XOR dataset — 4 examples, 2 inputs, 1 output.
2. Implement a hidden layer with 2–4 neurons.
3. Use a nonlinear activation (`tanh` is easiest to differentiate: `d/dz tanh(z) = 1 - tanh(z)^2`).
4. Implement the output layer.
5. Calculate gradients for all parameters manually with NumPy — no autograd.
6. **Gradient-check every parameter matrix** against finite differences before you trust the
   training loop.
7. Train until XOR is learned.

## Why XOR

XOR is the smallest problem that a linear model provably cannot solve. There is no straight
line that separates `{(0,0), (1,1)}` from `{(0,1), (1,0)}`. If your network learns it, the
hidden layer is genuinely doing something a single neuron cannot — that is the entire point of
depth, demonstrated on four data points you can check by hand.

## Two things that will probably bite you

- **Initialize weights randomly, not to zero.** If every hidden unit starts identical, every
  hidden unit receives an identical gradient and stays identical forever. The network collapses
  to one neuron. This is called symmetry breaking.
- **XOR is sensitive to initialization.** With only 4 examples and 2–4 hidden units, some seeds
  land in a flat region and stall. If loss plateaus around 0.25 and stops, re-run with a
  different seed before assuming your gradients are wrong — then gradient-check to be sure.

## Definition of done

- All four XOR examples are predicted correctly.
- Your analytic gradients match finite differences.
- You can explain why a single linear neuron cannot solve XOR, and what the hidden layer is
  representing that makes it solvable.

## Optional practice

Extra drills in `optional/`. The main exercise is the chapter; these are for more repetitions.

| File | New idea |
|---|---|
| `optional/t2.py` | Zeros, a non-zero constant, and random — the problem is symmetry, not zero |
| `optional/t3.py` | Delete the nonlinearity and fold two layers into one: depth stops mattering |

## Reference solutions

`solutions/04-mlp-from-scratch/train.py` — and `optional/` for the drills. Run them after you
have attempted your own.

## Further reading

Nielsen [ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html) and CS231n [neural-networks-1](https://cs231n.github.io/neural-networks-1/)

Full list: [`RESOURCES.md`](../RESOURCES.md)
