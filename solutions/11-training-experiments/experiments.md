# Worked experiment log

A filled-in example of `11-training-experiments/experiments.md`, from an actual run of
`run_experiments.py --steps 120 --sweep lr`. Yours will differ — the point is the shape of the
record, and writing the hypothesis down *before* seeing the result.

---

## Experiment 1 — learning rate

- **Hypothesis:** Higher learning rate trains faster, so 1e-2 should give the lowest loss in a
  fixed 120-step budget.
- **Variable changed:** `learning_rate`
- **Baseline:** 3e-4
- **Values tried:** 1e-5, 1e-4, 3e-4, 1e-3, 1e-2

| lr | train | val | gap |
|---|---|---|---|
| 1e-5 | 3.8716 | 3.8913 | +0.0197 |
| 1e-4 | 3.0179 | 3.0605 | +0.0425 |
| 3e-4 | 2.6315 | 2.6402 | +0.0087 |
| **1e-3** | **2.4587** | **2.4637** | +0.0050 |
| 1e-2 | 2.5058 | 2.5167 | +0.0108 |

- **Runtime:** ~12s per run, flat across all values.
- **Interpretation:** **Hypothesis wrong.** Loss falls from 1e-5 to 1e-3, then *rises* again at
  1e-2. Learning rate has an optimum, not a direction.

  At 1e-5 the model is not stuck — it is simply taking steps too small to get anywhere in 120
  steps. At 1e-2 the steps overshoot the minimum and it settles worse. Runtime is identical
  everywhere, which confirms this is about step *size*, not amount of computation.

  Note that 1e-3 beat the 3e-4 baseline here. That is a short-run result: with more steps,
  larger learning rates often lose to smaller ones that converge more precisely. Worth re-running
  at 2000 steps before concluding anything about the best long-run value.

---

## Experiment 2 — model width

- **Hypothesis:** Bigger `n_embed` gives lower training loss, and the train/val gap widens as
  capacity grows.
- **Variable changed:** `n_embed` (32, 64, 128, 256)
- **What to watch:** the **gap** column, not the training loss. Training loss falling while
  validation stalls is the signature of memorisation rather than learning.
- **Interpretation:** _(run `--sweep embed` and fill in)_

---

## Experiment 3 — context length

- **Hypothesis:** Longer `block_size` gives the model more to condition on, so loss should fall.
- **Variable changed:** `block_size` (16, 32, 64, 128)
- **Also record:** runtime. Attention cost grows with the *square* of context length, so this
  sweep should show a clear time penalty the others did not.
- **Interpretation:** _(run `--sweep block` and fill in)_

---

## Method notes

- **One variable at a time.** Change two and you cannot attribute the result to either.
- **Fix the seed.** Otherwise you are measuring initialisation noise. `run_experiments.py`
  re-seeds identically before every run.
- **Write the hypothesis first.** A prediction you got wrong teaches far more than a table you
  read after the fact — Experiment 1 above is the example.
- **Record runtime.** "Better loss" that costs 4× the time is a trade-off, not a win.
