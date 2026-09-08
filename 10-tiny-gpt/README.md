# Chapter 10 — Train a Tiny GPT

## Goal

Turn the components into a complete miniature GPT-style training and text-generation project.

## Concepts

Dataset pipeline, autoregressive objective, batches, validation split, optimizer, checkpointing, generation loop.

## Before you start: get a real corpus

`sample-data/tiny.txt` is four lines. It carried you through Chapters 6–8, and it will actively
mislead you here.

```bash
python3 sample-data/download_corpus.py
```

A Transformer memorises 144 characters in seconds. Loss will drop to near zero and generation
will echo the input back at you — it looks like success and teaches you nothing. You need
roughly 1 MB before validation loss means anything. See `sample-data/README.md`.

## Implementation requirements

1. Organize tokenizer, dataset, model, training, and generation into separate modules.
2. Train on `sample-data/shakespeare.txt` (~1.1 MB), not `tiny.txt`.
3. Hold out ~10% as a validation split and track **both** losses.
4. Save checkpoints.
5. Generate text from a prompt.
6. Keep the model small enough to understand and run locally — a few hundred thousand
   parameters is plenty on a CPU.

## What good output actually looks like

Calibrate your expectations, or you will think a working model is broken. A small
character-level model trained on ~1 MB for a few thousand steps produces text that is
*shaped* like English — plausible word lengths, capital letters after periods, speaker names
followed by colons — while being mostly nonsense words. That is success at this scale.

If you are getting perfectly formed sentences, check that you are not accidentally validating
on training data.

## Definition of done

- You have trained a real miniature GPT-like next-token model on a real corpus.
- You can explain the full path from raw text to generated text.
- You can state your model's parameter count and your final training and validation losses,
  and say whether the gap between them indicates overfitting.

## Reference solution

`solutions/10-tiny-gpt/` -- tokenizer, dataset, model, train, generate.

## Further reading

[nanoGPT](https://github.com/karpathy/nanoGPT) — the grown-up version of this chapter

Full list: [`RESOURCES.md`](../RESOURCES.md)
