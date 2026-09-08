# Chapter 1 — Single Neuron From Scratch

## Goal

Train the model `y_hat = w*x + b` on a tiny dataset generated from `y = 2*x + 1`.

## Concepts

Weights, bias, forward pass, mean squared error, gradients, learning rate, gradient descent, epoch.

## Implementation requirements

1. Create the dataset manually.
2. Initialize `w` and `b`.
3. Implement the forward pass.
4. Implement MSE.
5. Derive and implement `dw` and `db` manually.
6. Update parameters with gradient descent.
7. Print loss and parameters during training.
8. Do not use PyTorch or autograd.

## Definition of done

Loss decreases substantially and learned parameters approach `w ≈ 2`, `b ≈ 1`. You can explain why each update moves the parameters.

## Optional practice

The main exercise above is the chapter. These are extra, in `optional/` — do them only if you
want more repetitions on the same ideas.

| File | New idea |
|---|---|
| `optional/t2.py` | Two inputs — one gradient per weight, and a weight that must go negative |
| `optional/t3.py` | Learning rate — the three regimes, and exactly where the cliff is |

## Reference solutions

`solutions/01-single-neuron/` — `train.py`, and `optional/` for the drills. Attempt yours first.

## Further reading

Nielsen [ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) and CS231n [optimization-1](https://cs231n.github.io/optimization-1/)

Full list: [`RESOURCES.md`](../RESOURCES.md)
