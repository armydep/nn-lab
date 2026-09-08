# Chapter 9 — Tiny Transformer

## Goal

Combine embeddings, causal self-attention, a feed-forward network, residual connections, and normalization into a tiny Transformer language model.

## Concepts

Token embeddings, positional information, attention heads, feed-forward network, residual connection, layer normalization, Transformer block.

## Implementation requirements

1. Implement token embeddings.
2. Add positional embeddings.
3. Implement one causal self-attention head.
4. Extend to multi-head attention.
5. Add a feed-forward network.
6. Add residual connections and layer normalization.
7. Stack 1–2 Transformer blocks.
8. Produce vocabulary logits.

## Definition of done

A forward pass accepts a token batch and returns logits of shape `[batch, time, vocab]`, and every component is understandable from earlier chapters.

## Reference solution

`solutions/09-tiny-transformer/model.py`

## Further reading

[Attention Is All You Need](https://arxiv.org/abs/1706.03762); Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Full list: [`RESOURCES.md`](../RESOURCES.md)
