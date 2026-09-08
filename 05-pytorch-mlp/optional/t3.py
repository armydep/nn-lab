"""Chapter 5, Task 3 -- what optimizer.zero_grad() is actually for.

Your Part A engine accumulates with `+=`, because a value used twice must
receive gradient from both paths. PyTorch made the same decision, and it
has a consequence that catches everyone exactly once: gradients also
accumulate ACROSS training steps unless you clear them.

Nothing crashes. Nothing warns. The model just trains worse, for a reason
that is invisible in the training loop.

    python3 05-pytorch-mlp/optional/t3.py
"""

import torch
import torch.nn as nn

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 8
LEARNING_RATE = 0.1
EPOCHS = 3000
SEED = 0


def make_model():
    """Seeded, so both runs below start from identical weights."""
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(2, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))


# ----------------------------------------------------------------------
# YOUR TURN (1 of 3) -- prove that backward() adds rather than sets.
#
# Build a model and an MSELoss. Then:
#
#   1. forward, loss, backward       -> record next(model.parameters()).grad
#   2. forward, loss, backward AGAIN -> record it again, no clearing
#   3. model.zero_grad(), then forward/loss/backward once more
#
# Predict the second number before you print it. Then assert it: the
# second reading should be exactly twice the first, and the third should
# be back to the first.
# ----------------------------------------------------------------------
def part_one():
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (2 of 3) -- the same training loop, one line apart.
#
# Write ONE train() that takes a `clear_grads` flag and calls
# optimizer.zero_grad() only when it is True. Everything else -- seed,
# model, optimizer, epochs -- identical.
#
# Return (final_loss, correct_out_of_4, gradient_norm). The gradient norm
# is the diagnostic that makes the cause obvious:
#
#   torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
#
# Run it both ways and print them side by side.
# ----------------------------------------------------------------------
def train(clear_grads):
    pass  # <- replace


def part_two():
    pass  # <- replace


# ----------------------------------------------------------------------
# YOUR TURN (3 of 3) -- accumulation on purpose.
#
# The behaviour that bit you in part two is a feature when you ask for it.
# Show that these two produce the SAME gradient:
#
#   a) one backward over all 4 examples
#   b) four backwards, one example at a time, each loss divided by 4,
#      with no zeroing in between
#
# Compare with max absolute difference; expect < 1e-6. This is gradient
# accumulation -- how models are trained with an effective batch size
# larger than the memory available.
# ----------------------------------------------------------------------
def part_three():
    pass  # <- replace


def main():
    part_one()
    part_two()
    part_three()


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Definition of done
#
#   Two backwards double .grad; the run with zero_grad() solves XOR 4/4
#   while the run without it does not; and chunked accumulation reproduces
#   the whole-batch gradient to within 1e-6.
#
# Then think about:
#
#   1. Why does PyTorch accumulate instead of overwriting? Trace it back
#      to a decision you made yourself in Part A of this chapter.
#
#   2. Without clearing, what exactly does step k apply -- not "too much",
#      but precisely which quantity? Say it as a formula over steps 1..k,
#      then look at the two gradient norms again.
#
#   3. Nothing raises an exception in the broken run. Compare that with
#      Chapter 3's one-path gradient. What do these two bugs have in
#      common, and why is that category the most expensive kind?
#
#   4. In part three you relied on the same accumulation deliberately. If
#      you split a batch of 64 into 4 chunks of 16, what must you divide
#      each chunk's loss by, and what happens to the result if you forget?
# ----------------------------------------------------------------------
