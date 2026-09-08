# Numerical Stability — the two walls everyone hits

Both of these produce `nan` or `inf` in your loss. Neither is a bug in your gradient
derivation, and both will cost you an evening if you do not know they exist.

Read this when your loss suddenly becomes `nan`.

---

## Wall 1 — `log(0)` in cross-entropy (Chapter 2 onward)

Binary cross-entropy is:

```text
L = -[ y*log(p) + (1-y)*log(1-p) ]
```

If your sigmoid saturates, `p` becomes exactly `0.0` or `1.0` in floating point. Then `log(0)`
is `-inf`, the loss is `inf` or `nan`, and every gradient afterwards is `nan`. Once one `nan`
enters your parameters, everything downstream is `nan` forever.

### Fix: clip before the log

```python
eps = 1e-12
p = np.clip(p, eps, 1.0 - eps)
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
```

### The better fix: never form `p` at all

Combining sigmoid and cross-entropy algebraically gives an expression with no `log` of a
possibly-zero number:

```text
L = max(z, 0) - z*y + log(1 + exp(-|z|))
```

and the gradient collapses to the remarkably clean:

```text
dL/dz = sigmoid(z) - y
```

This is why PyTorch offers `BCEWithLogitsLoss` and warns against `Sigmoid` followed by
`BCELoss`. Derive `dL/dz = sigmoid(z) - y` yourself in Chapter 2 — it takes ten minutes and
explains a design decision you will meet in every framework.

---

## Wall 2 — softmax overflow (Chapters 6–10)

Softmax is:

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

`exp(1000)` is `inf` in float64. Logits reach that range easily once a model trains for a
while. `inf / inf` is `nan`.

### Fix: subtract the max first

```python
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # largest value is now 0
    e = np.exp(x)                                  # so exp() maxes out at 1.0
    return e / np.sum(e, axis=axis, keepdims=True)
```

Subtracting a constant from every logit **does not change the result** — the constant cancels
between numerator and denominator. It only changes what floating point has to represent. This
is a free fix and you should write softmax this way permanently.

### The related trap: log-softmax

For cross-entropy you want `log(softmax(x))`. Do not compute softmax and then take the log —
if a probability underflows to `0.0`, you are back at Wall 1. Use the stable form:

```python
def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
```

Then cross-entropy is just indexing: `loss = -log_softmax(logits)[range(n), targets].mean()`.

---

## Debugging checklist when loss becomes `nan`

1. **Find the first `nan`.** Print loss every step. Did it appear gradually (exploding
   gradients) or instantly (a `log(0)` or `0/0`)?
2. **Check for saturation.** Print `min` and `max` of your probabilities and logits. If
   probabilities hit exactly 0 or 1, that is Wall 1. If logits exceed ~100, that is Wall 2.
3. **Lower the learning rate 10×.** If the `nan` goes away, gradients were exploding rather
   than a stability bug.
4. **Gradient-check** (Chapter 3). If analytic and numeric gradients disagree, the bug is your
   derivation, not numerics.

Rule of thumb: `nan` that appears **instantly** is usually a domain error like `log(0)`. `nan`
that appears **after a few hundred steps** is usually exploding gradients or saturation.
