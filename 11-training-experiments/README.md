# Chapter 11 — Training Experiments

## Goal

Change one training variable at a time and observe its effect instead of treating hyperparameters as magic constants.

## Concepts

Learning rate, batch size, context length, embedding size, depth, heads, overfitting, underfitting, generalization.

## Implementation requirements

1. Run controlled experiments on learning rate.
2. Run controlled experiments on batch size.
3. Change model size and context length.
4. Record parameter count, training loss, validation loss, runtime, and generated samples.

## Definition of done

You can explain at least three experiments where changing one variable produced a measurable effect and why you think it happened.

## Reference solution

`solutions/11-training-experiments/run_experiments.py` -- a harness that sweeps one variable
at a time and writes results into the experiment table.

## Further reading

Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

Full list: [`RESOURCES.md`](../RESOURCES.md)
