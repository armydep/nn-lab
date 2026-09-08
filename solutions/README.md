# Reference solutions

## Use these correctly

**Attempt the chapter first.** Reading a solution before struggling with the problem spends the
learning without buying anything — you will recognise every line and retain none of it.

These exist for two moments:

1. You are genuinely stuck and want to see one way through.
2. You have something working and want to diff your approach against another.

## Running them

```bash
pip install -r requirements.txt
python3 solutions/01-single-neuron/train.py
```

Chapters 1–4 need only NumPy. Chapter 5 onward needs PyTorch. Chapters 10–11 need the downloaded
corpus:

```bash
python3 sample-data/download_corpus.py
```

Verify everything runs:

```bash
python3 solutions/verify_all.py
```

## Layout

Each chapter's main exercise sits at the top of its folder. Anything in `optional/` is an extra
drill — worth doing, never required.

```text
solutions/03-backpropagation/
├── gradients.py        the chapter's exercise
└── optional/
    ├── t2.py
    └── t3.py
```

## What each solution demonstrates

| Chapter | File | Verifies |
|---|---|---|
| 01 | `train.py` | Converges to `w≈2, b≈1`; gradient-checked |
| 01 | `optional/t2.py` | One gradient per weight; a weight that must go negative |
| 01 | `optional/t3.py` | The three learning-rate regimes, and where the cliff is |
| 02 | `train.py` | 8/8 accuracy; confirms `dL/dz = p - y` |
| 02 | `optional/t2.py` | Non-separable data; 80% is the correct ceiling |
| 02 | `optional/t3.py` | Naive cross-entropy produces `nan`; the safe form never does |
| 03 | `gradients.py` | All four gradients match finite differences |
| 03 | `optional/t2.py` | Two routes into one parameter; the one-path version fails the check |
| 03 | `optional/t3.py` | Locates a planted bug from which gradients fail; U-shaped `eps` curve |
| 04 | `train.py` | XOR 4/4; every parameter matrix gradient-checked |
| 04 | `optional/t2.py` | Zeros and constants both collapse to 1 distinct hidden unit; random does not |
| 04 | `optional/t3.py` | Linear network cannot pass 2/4; folds exactly into one layer |
| 05 | `micrograd.py` | Engine reproduces Chapter 3's hand-derived gradients exactly |
| 05 | `train.py` | PyTorch `.grad` matches the same numbers; XOR 4/4 |
| 05 | `optional/t2.py` | Engine composes `exp`/`log` into `dL/dz = p - y` unaided |
| 05 | `optional/t3.py` | Two backwards double `.grad`; omitting `zero_grad()` loses XOR |
| 06 | `train.py` | Learned probabilities match hand-counted bigram frequencies |
| 07 | `train.py` | Wider context lowers loss — and demonstrates memorisation |
| 08 | `attention.py` | Rows sum to 1; causal mask leaks nothing |
| 09 | `model.py` | Logits are `[batch, time, vocab]`; init loss ≈ `log(vocab)` |
| 10 | `train.py`, `generate.py` | Trains on ~1 MB; tracks train/val; checkpoints; generates |
| 11 | `run_experiments.py` | One-variable-at-a-time sweeps with a fixed seed |
| 12 | `notes.md` | Worked tiny-GPT → production-LLM mapping |

## The thread running through them

Chapter 3 derives four gradients by hand. Chapter 5's `micrograd.py` reproduces those exact numbers
automatically, and Chapter 5's `train.py` shows PyTorch producing them a third time. The same four
numbers, three ways — that is the point where `loss.backward()` stops being magic.

## Expectations for Chapter 10

Trained on ~1.1 MB of Shakespeare for 1500 steps (~3 minutes on a CPU), a 618k-parameter model
reaches roughly train 1.93 / val 2.00 and generates text like:

```text
Cisin:
A would the, reven you heery therf, ade mors,
Thincais the waty beling bekingt your to a partse
```

Speaker names, colons, line breaks, plausible word shapes — and mostly nonsense words. **That is
success at this scale.** If you expected sentences, recalibrate rather than assuming a bug.
