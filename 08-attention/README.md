# Chapter 8 — Self-Attention From First Principles

## Goal

Implement a minimal causal self-attention calculation separately from a full Transformer.

## Concepts

Query, Key, Value, attention score, scaled dot product, softmax, weighted sum, causal mask.

## Implementation requirements

1. Create a tiny sequence of token vectors.
2. Project them into Q, K, and V.
3. Compute `Q @ K.T`.
4. Scale scores.
5. Apply a causal mask.
6. Apply softmax.
7. Multiply attention weights by V.
8. Inspect the attention matrix.

## Definition of done

You can explain what each row of the attention matrix means and why causal masking prevents looking into the future.

## Watch out

The softmax inside attention is applied to masked scores. Use `-inf` (not a large negative
number, and not zero) for masked positions so they receive exactly zero weight, and keep the
max-subtraction fix from [`NUMERICS.md`](../NUMERICS.md).

Do not forget the `1/sqrt(head_dim)` scaling. Without it, dot products grow with dimension,
softmax saturates into a near-one-hot distribution, and gradients vanish.

## Reference solution

`solutions/08-attention/attention.py`

## Further reading

Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

Full list: [`RESOURCES.md`](../RESOURCES.md)
