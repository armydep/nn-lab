# Chapter 6 — Bigram Language Model

## Goal

Build your first language model: predict the next character using only the current character.
Along the way, meet the two ideas that every remaining chapter depends on — **softmax** and
**multi-class cross-entropy**.

## Concepts

Token, vocabulary, token ID, logits, softmax, cross-entropy, next-token prediction, sampling.

## New here: from 2 classes to N classes

Chapter 2 predicted one probability with sigmoid. Predicting the next character means choosing
among the whole vocabulary — 65 classes for Shakespeare. Two things generalise:

| Binary (Chapter 2) | Multi-class (here) |
|---|---|
| `sigmoid(z)` → one probability | `softmax(z)` → a probability *distribution* over the vocab |
| Binary cross-entropy | Cross-entropy over the true class |
| `dL/dz = p - y` | `dL/dz = p - onehot(y)` |

That last row is the payoff for deriving `p - y` in Chapter 2: the multi-class gradient is the
*same rule*. Softmax probabilities, minus 1 at the correct class. Derive it once here and you
have the gradient used by every model in Chapters 7–12.

### Softmax

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

Turns arbitrary real numbers ("logits") into positive numbers summing to 1. **Always subtract
the max before exponentiating** or you will get `inf` — see [`NUMERICS.md`](../NUMERICS.md).

### Cross-entropy

```text
L = -log(p[correct_class])
```

The model's assigned probability for the right answer, negated-log. Perfect prediction (`p = 1`)
gives loss 0; `p = 0` gives infinite loss. It only ever looks at the correct class.

## Implementation requirements

1. Load a tiny text file (`sample-data/tiny.txt` — small enough to check counts by hand).
2. Build a character vocabulary.
3. Implement encode/decode.
4. Build `(current_char, next_char)` training examples.
5. Implement softmax and cross-entropy yourself, with the max-subtraction fix.
6. Train a bigram model — a `(vocab, vocab)` table of logits, trained by gradient descent.
7. Generate text by repeatedly sampling the next character.
8. **Sample, do not argmax.** Always taking the most likely character produces an immediate
   loop. Draw from the distribution instead.

## A sanity check you can do by hand

A bigram model has a closed-form optimum: the trained probabilities should approach the counts
in your data. Count how often `'l'` is followed by `'e'` in `tiny.txt`, divide by the number of
times `'l'` appears, and compare to your model's learned probability. They should match closely.

This is exactly why the corpus is four lines. Do this check — it confirms your softmax,
cross-entropy, and gradients are all correct, before those same components go into a
Transformer where verification is much harder.

## Definition of done

- The model produces character sequences with patterns learned from the corpus, even if the
  text is poor.
- Your learned probabilities match hand-counted frequencies.
- You can explain what a logit is, what softmax does to it, and why `dL/dlogits` is
  `p - onehot(y)`.

## Reference solution

`solutions/06-bigram-language-model/train.py`

## Further reading

Karpathy's [makemore](https://github.com/karpathy/makemore) series

Full list: [`RESOURCES.md`](../RESOURCES.md)
