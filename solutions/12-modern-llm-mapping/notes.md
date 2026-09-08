# Tiny GPT → modern LLM mapping

A worked version of `12-modern-llm-mapping/notes.md`. Fill in your own first — the value is in
recognising the components yourself, having built them.

## The mapping

| Your implementation | Production LLM equivalent | What changes at scale? |
|---|---|---|
| Character tokenizer, ~65 tokens | BPE / SentencePiece, 50k–200k tokens | Subword units mean shorter sequences for the same text. The interface is unchanged: `encode(str) -> list[int]`. |
| `nn.Embedding(vocab, 128)` | Same layer, 4k–16k dimensions | Nothing conceptually. Just wider. |
| Learned positional embeddings | RoPE (rotary), ALiBi | Learned absolute positions do not extrapolate past `block_size`. RoPE encodes *relative* position and generalises to longer contexts. A genuine architectural improvement, not just scale. |
| 4 attention heads | 32–128 heads, often grouped-query (GQA) | GQA shares key/value projections across heads to shrink the KV cache at inference. Driven by memory bandwidth, not by modelling. |
| Causal mask via `tril` | Identical, plus FlashAttention | FlashAttention computes the *same* mathematics without materialising the T×T matrix. Pure engineering — identical outputs, far less memory. |
| 3 Transformer blocks | 32–120 blocks | Deeper. Also pre-norm everywhere, and usually RMSNorm rather than LayerNorm (cheaper, works as well). |
| ReLU feed-forward | SwiGLU / GeGLU | Gated activations, empirically better. Same position-wise role. |
| `F.cross_entropy` on next token | Identical | **The objective does not change at all.** GPT-4 is trained on exactly the loss you implemented in Chapter 6. |
| AdamW, lr 3e-4 | AdamW + warmup + cosine decay | Schedules and gradient clipping matter enormously at scale. The optimizer is the same one. |
| 1 MB of Shakespeare | 1–15 trillion tokens, heavily filtered | The largest single difference. Data quality and deduplication now matter more than architecture. |
| 618k parameters | 7B–500B+ | ~5–6 orders of magnitude. |
| One CPU, 3 minutes | Thousands of GPUs, weeks | Requires distributed training: data/tensor/pipeline parallelism, mixed precision, activation checkpointing. All of this exists to fit the same computation into hardware. |
| `model.generate()` sampling loop | Same loop + KV caching, beam/nucleus sampling, batching | KV caching avoids recomputing attention over the whole prefix each step. An optimisation of your loop, not a different one. |

## What stays the same

These you have now genuinely implemented, and they do not change:

1. **The objective.** Predict the next token; minimise cross-entropy. Identical at every scale.
2. **The Transformer block.** Attention to move information between positions, an MLP to
   process within a position, residuals and normalisation to keep it trainable.
3. **Backpropagation.** The chain rule from Chapter 3, applied to a bigger graph.
4. **Gradient descent.** Adam is a refinement of the update rule from Chapter 1.
5. **Autoregressive generation.** Predict, sample, append, repeat.

## What is genuinely new at scale

Not "the same but bigger":

- **RLHF / instruction tuning.** Base models trained as above only continue text. The
  helpful-assistant behaviour comes from a *separate* post-training stage — supervised
  fine-tuning, then reinforcement learning from human or AI feedback. Nothing in Chapters 1–11
  covers this, and it is the difference between a text completer and a chatbot.
- **Emergent capabilities.** In-context learning — solving tasks from examples in the prompt,
  with no weight updates — is not present in a 618k-parameter model at any amount of training.
- **Data curation as the primary lever.** At scale, filtering and deduplication buy more than
  architecture changes.
- **Inference economics.** Quantisation, speculative decoding, KV cache management. Irrelevant
  at your scale, dominant at production scale.

## The honest summary

A production LLM is your Chapter 10 model with: a better tokenizer, rotary positions, gated
activations, ~100× the depth, ~50× the width, ~10⁷× the data, a distributed training stack, and
an entire post-training phase you have not built.

The forward pass would still be recognisable to you. So would the loss, the optimizer, and the
generation loop. That recognition is the point of the whole roadmap — the remaining gap is
engineering and data, plus one genuinely separate idea (RLHF) that you now know to go and learn
about specifically.
