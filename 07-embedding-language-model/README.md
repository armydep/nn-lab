# Chapter 7 — Embeddings + Neural Language Model

## Goal

Predict the next character from multiple previous characters using learned embeddings and an MLP.

## Concepts

Embeddings, embedding dimension, context window, sequence examples, batches, learned representations.

## Implementation requirements

1. Reuse the character tokenizer.
2. Build fixed-length context windows.
3. Create an embedding table.
4. Combine context embeddings.
5. Feed them into an MLP.
6. Predict the next token with cross-entropy.
7. Generate autoregressively.

## Definition of done

The model uses more than one previous character, and you understand how token IDs become learned vectors.

## Reuse from Chapter 6

Bring your stable `softmax` / `log_softmax` and the `p - onehot(y)` gradient across unchanged.
If loss becomes `nan`, see [`NUMERICS.md`](../NUMERICS.md).

## Reference solution

`solutions/07-embedding-language-model/train.py`

## Further reading

Karpathy's [makemore](https://github.com/karpathy/makemore) series (parts 2–3)

Full list: [`RESOURCES.md`](../RESOURCES.md)
