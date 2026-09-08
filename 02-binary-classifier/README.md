# Chapter 2 — Binary Classifier

## Goal

Build a single artificial neuron with two inputs and sigmoid activation for binary
classification.

## Concepts

Vectors, weighted sum, sigmoid, probability, binary cross-entropy, gradients.

## Implementation requirements

1. Create a tiny linearly separable 2D dataset.
2. Implement `z = w1*x1 + w2*x2 + b`.
3. Implement sigmoid.
4. Implement binary cross-entropy.
5. Derive gradients manually.
6. Train with gradient descent.
7. Inspect predicted probabilities.

## Watch out: `log(0)`

Binary cross-entropy takes `log(p)`. When sigmoid saturates, `p` becomes exactly `0.0` or `1.0`
in floating point, `log(0)` is `-inf`, and your loss turns into `nan` — permanently, since every
subsequent gradient is then `nan` too.

The quick fix is clipping `p` into `[1e-12, 1 - 1e-12]` before the log. See
[`NUMERICS.md`](../NUMERICS.md) for the proper fix and why every framework ships a fused
"sigmoid + cross-entropy" loss.

## Worth deriving yourself

Work out `dL/dz` where `L` is binary cross-entropy and `p = sigmoid(z)`. The chain rule gives
you a messy-looking expression that collapses to:

```text
dL/dz = p - y
```

Prediction minus target. That is the whole gradient. It is a genuinely surprising result the
first time, it makes your code trivial, and the same clean form reappears for softmax +
cross-entropy in Chapter 6 — which is not a coincidence.

## Definition of done

The model correctly separates the tiny dataset, and you can explain the role of sigmoid, what
the decision boundary is, and why `dL/dz` simplifies to `p - y`.

## Optional practice

Extra drills in `optional/`. The main exercise is the chapter; these are for more repetitions.

| File | New idea |
|---|---|
| `optional/t2.py` | Data that does not separate -- why 100% accuracy is the wrong goal |
| `optional/t3.py` | Break it on purpose -- produce the `log(0)` nan, then fix it |

## Reference solutions

`solutions/02-binary-classifier/` -- `train.py`, and `optional/` for the drills. Attempt yours
first.

## Further reading

Nielsen [ch. 3](http://neuralnetworksanddeeplearning.com/chap3.html) — derives why `dL/dz = p - y`

Full list: [`RESOURCES.md`](../RESOURCES.md)
