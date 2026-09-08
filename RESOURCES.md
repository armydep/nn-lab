# Resources

This repository is the exercise sheet. These are the explanations.

Read them **after** attempting a chapter, not before. An explanation you read while stuck on a
problem lands very differently from one you read cold.

---

## Start here

If you read only three things:

1. **Karpathy — "The spelled-out intro to neural networks and backpropagation: building
   micrograd"** ([video](https://www.youtube.com/watch?v=VMj-3vsdGqc), ~2.5 hours)
   This is Chapters 3–5 of this roadmap, almost exactly: derivatives by hand, then an autograd
   engine built from nothing. Code at [karpathy/micrograd](https://github.com/karpathy/micrograd).

2. **Chris Olah — "Calculus on Computational Graphs: Backpropagation"**
   ([colah.github.io](https://colah.github.io/posts/2015-08-Backprop/))
   Short, and the clearest written account of *why* the backward pass reuses the work of the
   layer above. Chapter 3 in essay form.

3. **3Blue1Brown — Neural Networks**
   ([playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), 4 short videos)
   Visual intuition for gradient descent and backpropagation. Best used alongside the maths,
   not instead of it.

---

## By chapter

| Chapter | Read |
|---|---|
| **01–02** Single neuron, classifier | Nielsen, [ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html); CS231n [optimization-1](https://cs231n.github.io/optimization-1/) |
| **03** Backpropagation | Olah's [backprop post](https://colah.github.io/posts/2015-08-Backprop/); CS231n [optimization-2](https://cs231n.github.io/optimization-2/); Nielsen [ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html) |
| **04** MLP from scratch | Nielsen ch. 2; CS231n [neural-networks-1](https://cs231n.github.io/neural-networks-1/) |
| **05** Autograd + PyTorch | Karpathy's micrograd video; PyTorch [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) and [autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) |
| **06–07** Bigram, embeddings | Karpathy's **makemore** series ([repo](https://github.com/karpathy/makemore)) — the same bigram → MLP → embeddings progression |
| **08–09** Attention, Transformer | Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/); then [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) |
| **10** Tiny GPT | Karpathy, ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY); [nanoGPT](https://github.com/karpathy/nanoGPT) — the grown-up version of Chapter 10 |
| **11** Experiments | Karpathy, ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/); Distill, [Momentum](https://distill.pub/2017/momentum/) |
| **12** Modern LLMs | Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) |

---

## On specific things this repo makes you confront

### Why `dL/dz = p - y` (Chapters 2 and 6)

Nielsen, [ch. 3](http://neuralnetworksanddeeplearning.com/chap3.html) — the section on
cross-entropy versus quadratic cost derives exactly this cancellation, and explains why it makes
training faster. The single best explanation of the result you derive in Chapter 2.

### Why the loss value is separate from the gradient

CS231n [optimization-1](https://cs231n.github.io/optimization-1/) covers the analytic-vs-numeric
gradient distinction and why you would compute both. If you have wondered why `mse()` never
feeds into the parameter update, this is the section.

### Numerical stability — `log(0)` and softmax overflow

*Deep Learning* (Goodfellow, Bengio, Courville),
[ch. 4](https://www.deeplearningbook.org/contents/numerical.html) — the textbook treatment of
overflow, underflow, and the log-sum-exp trick. This is the theory behind
[`NUMERICS.md`](NUMERICS.md).

### Learning rate and divergence

CS231n [neural-networks-3](https://cs231n.github.io/neural-networks-3/) covers learning-rate
schedules, gradient checking, and sanity checks — much of Chapter 11, written down.

### Why you should not skip the manual derivation

Karpathy, ["Yes you should understand
backprop"](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — makes
the same argument Chapter 3 does: treating `backward()` as a black box will eventually cost you.

---

## Longer form

- **Michael Nielsen, *Neural Networks and Deep Learning*** —
  [free online](http://neuralnetworksanddeeplearning.com/). The best theory-first companion to
  Chapters 1–5. Chapters 1–3 cover everything up to Chapter 5.
- **CS231n course notes** — [cs231n.github.io](https://cs231n.github.io/). Sits between Nielsen
  and the Goodfellow textbook in depth. The optimization and neural-network notes are the
  relevant ones here.
- **Karpathy, "Neural Networks: Zero to Hero"** —
  [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero). This
  roadmap and that course are close to the same curriculum: this repo is the exercise sheet, his
  is the lecture series. Running them in parallel works well — attempt the chapter, then watch the
  matching video to see another route to the same place.
- **Goodfellow, Bengio & Courville, *Deep Learning*** —
  [free online](https://www.deeplearningbook.org/). Reference, not a tutorial. Reach for a
  specific chapter; do not read it front to back.

---

## Two caveats

**These links go stale.** The Annotated Transformer (Harvard NLP's line-by-line implementation of
the original paper) has moved at least once — search for it by name rather than trusting a URL.
The same applies to anything here that 404s.

**This list leans on Karpathy.** That is deliberate: his material is unusually well matched to
implementation-first learning, which is what this repository committed to. If you would rather
have theory first, Nielsen and Goodfellow are the better spine, with CS231n between them.
