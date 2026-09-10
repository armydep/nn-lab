# Neural Networks From Scratch — The Exercise Lab

Twelve chapters of exercises that build a neural network from nothing. You write every line: the
forward pass, the loss, the gradients derived by hand, then your own autograd engine, and finally
a small GPT you train yourself. NumPy only until Chapter 5, PyTorch after that.

No frameworks until you have earned them. No pretrained models. No hidden steps.

## Run the tests

Every chapter ships with a worked reference solution, and those solutions check themselves.

```bash
pip install -r requirements.txt

python3 solutions/verify_all.py --quick   # 20 checks, nothing to download

python3 sample-data/download_corpus.py    # ~1 MB, needed by the last 3 checks
python3 solutions/verify_all.py           # all 23 checks
```

A zero exit code means every chapter actually met its goal, not merely that it ran without
crashing. Chapters 1–11 are covered by those 23 checks; Chapter 12 is a written exercise with a
worked answer rather than a runnable one.

## The book is separate, and optional

This repository is the free companion to the book *The Practical Introduction to AI* by Arkady
Mishiev. **The book is not in this repository, and you do not need it to use the lab** — every
chapter brief states what to build and how to know you are done, and every solution is here.

That split is deliberate, and it extends to the licensing: **this code is MIT licensed (see
[`LICENSE`](LICENSE)); the book is a separate work, sold separately, under its own licence.**

## Part 1 — Neural network fundamentals (Chapters 1–5)

**This is the core.** Five chapters, realistically a weekend or two.

```text
01 Single neuron from scratch
        ↓
02 Binary classifier
        ↓
03 Backpropagation & computation graphs
        ↓
04 MLP from scratch
        ↓
05 Mini-autograd, then rebuild with PyTorch
        ↓
     you are done
```

By the end of Chapter 5 you will have derived gradients by hand, trained a network that learns
something a linear model provably cannot, and built the autograd engine that PyTorch's
`loss.backward()` replaces. That is a complete answer to *"how are neural networks trained?"*

**Finishing Chapter 5 is finishing.** It is not five twelfths of anything.

### What you need

High-school algebra, and enough Python to write a loop. Derivatives are explained where they are
used — you are not expected to arrive knowing calculus, only to be willing to follow it.

## Part 2 — The language model arc (Chapters 6–12)

Optional continuation, and a much bigger commitment — several weeks. It builds directly on
Part 1 and ends at a small GPT you train yourself.

```text
06 Bigram language model  →  07 Embeddings  →  08 Self-attention  →  09 Tiny Transformer
        →  10 Tiny GPT training project  →  11 Training experiments  →  12 Map to modern LLMs
```

Do this part if you want to understand LLMs specifically. Skip it and you have still learned the
thing Part 1 promised.

## How each chapter is laid out

```text
03-backpropagation/
├── README.md        what to build, and the definition of done
├── gradients.py     the exercise: a scaffold with the answers left out
└── optional/        extra drills, for more practice on the same ideas
    ├── t2.py
    └── t3.py
```

**Everything in `optional/` is optional.** The main exercise is the chapter. The drills are
there if you want more repetitions, or if something did not click the first time.

## Learning rule

Implement each chapter yourself before moving to the next. Prefer explicit code over frameworks
until Chapter 5.

## Ground rules

- Chapters 1–4: use Python/NumPy only; no autograd.
- Chapter 5+: use PyTorch where requested.
- Do not use Hugging Face, LangChain, pretrained models, agents, or RAG here.
- Keep datasets tiny enough that you can inspect them manually.
- Track loss and inspect model output instead of treating training as a black box.

## Core training loop

```text
Data → Forward pass → Prediction → Loss → Backpropagation → Gradients → Parameter update → Repeat
```

Every chapter in Part 1 is one piece of that cycle.

## The one debugging tool to learn early

From Chapter 3 onward, whenever a network will not learn, check your gradients numerically:

```text
numeric_grad = (L(w + eps) - L(w - eps)) / (2 * eps)     # eps = 1e-5
```

If this disagrees with your analytic gradient, the bug is in your derivation, not your learning
rate. This will save you more time than anything else in this repository.

See [`NUMERICS.md`](NUMERICS.md) for the floating-point traps that produce silently wrong numbers.

## Reference solutions

Every chapter has a worked solution under `solutions/`. **Attempt the chapter first** — the value is
in the struggle, and reading the answer first spends the learning without buying anything. They
are there for when you are genuinely stuck, and to diff against once you have something working.

Each one also runs standalone:

```bash
python3 solutions/01-single-neuron/train.py
```

## Environment

Python 3.11+. Chapters 1–4 need only NumPy. Chapter 5 onward needs PyTorch — if the pinned build
in `requirements.txt` does not suit your machine, install PyTorch using the command from the
[official PyTorch installation page](https://pytorch.org/get-started/locally/) instead.

## Corpus for Part 2 (Chapters 6–12)

`sample-data/tiny.txt` is four lines. That is deliberate: in Chapters 6–7 you want a corpus small
enough to verify counts by hand.

It is far too small to train a Transformer. Before Chapter 10, fetch a real one:

```bash
python3 sample-data/download_corpus.py
```

See [`sample-data/README.md`](sample-data/README.md) for why corpus size matters and what to
expect.

## Further reading

Curated tutorials, videos and books, mapped chapter by chapter:
[`RESOURCES.md`](RESOURCES.md). Read them *after* attempting a chapter — an explanation you meet
while stuck lands very differently from one read cold.

## Licence

MIT — see [`LICENSE`](LICENSE). Use it, fork it, teach from it.
