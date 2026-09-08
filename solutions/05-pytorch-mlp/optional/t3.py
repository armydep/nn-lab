"""Chapter 5, Task 3 solution -- what optimizer.zero_grad() is actually for.

Part A's engine accumulates with `+=`, because a value used twice must
receive gradient from both paths. PyTorch made the same decision, and it
has a consequence that catches everyone once: gradients also accumulate
ACROSS training steps unless you clear them.

Nothing crashes. The loss just gets worse, for a reason invisible in the
training loop.

    python3 solutions/05-pytorch-mlp/optional/t3.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 8
LEARNING_RATE = 0.1
EPOCHS = 3000
SEED = 0


def make_model() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(2, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))


def part_one() -> None:
    """Two backward passes, no clearing in between."""
    print("=" * 62)
    print("PART 1 -- backward() adds to .grad, it does not set it")
    print("=" * 62)

    model = make_model()
    criterion = nn.MSELoss()
    first_param = next(model.parameters())

    loss = criterion(model(X), Y)
    loss.backward()
    once = first_param.grad.clone()

    # A second backward on a freshly built graph, with no zeroing.
    loss = criterion(model(X), Y)
    loss.backward()
    twice = first_param.grad.clone()

    print(f"after 1 backward   grad[0][0] = {once[0][0]:+.9f}")
    print(f"after 2 backwards  grad[0][0] = {twice[0][0]:+.9f}")
    print(f"ratio = {(twice[0][0] / once[0][0]):.6f}")

    assert torch.allclose(twice, 2 * once, atol=1e-7), "second backward should double .grad"

    model.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    after_clear = first_param.grad.clone()
    print(f"after zero_grad()  grad[0][0] = {after_clear[0][0]:+.9f}   (back to the single value)")
    assert torch.allclose(after_clear, once, atol=1e-7), "zero_grad should restore the single value"


def train(clear_grads: bool) -> tuple[float, int, float]:
    """Train XOR. The ONLY difference between the two runs is zero_grad()."""
    model = make_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for _ in range(EPOCHS):
        if clear_grads:
            optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(X)
        final_loss = criterion(out, Y).item()
        correct = int(((out >= 0.5) == (Y == 1)).sum())
        grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
    return final_loss, correct, grad_norm


def part_two() -> None:
    print("\n" + "=" * 62)
    print("PART 2 -- the same loop, run with and without zero_grad()")
    print("=" * 62)

    loss_ok, correct_ok, norm_ok = train(clear_grads=True)
    loss_bad, correct_bad, norm_bad = train(clear_grads=False)

    print(f"{'run':<24}{'final loss':>14}{'correct':>10}{'grad norm':>16}")
    print(f"{'with zero_grad()':<24}{loss_ok:>14.6f}{correct_ok:>10}{norm_ok:>16.3e}")
    print(f"{'without zero_grad()':<24}{loss_bad:>14.6f}{correct_bad:>10}{norm_bad:>16.3e}")

    assert correct_ok == 4, "the correct loop should solve XOR"
    assert loss_ok < 0.01, "the correct loop should reach a low loss"
    assert loss_bad > loss_ok, "omitting zero_grad should make the result worse"
    assert norm_bad > norm_ok, "omitting zero_grad should inflate the gradient norm"

    print("\nNo exception, no warning. One run learned XOR and the other did")
    print("not, and the training loops differ by a single line.")


def part_three() -> None:
    """Accumulation is a feature when you ask for it on purpose."""
    print("\n" + "=" * 62)
    print("PART 3 -- accumulation used deliberately")
    print("=" * 62)

    # One backward over the whole batch of 4.
    model = make_model()
    criterion = nn.MSELoss()
    model.zero_grad()
    criterion(model(X), Y).backward()
    whole_batch = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

    # Four backwards, one example at a time, each scaled by 1/4, accumulated.
    model = make_model()
    model.zero_grad()
    for i in range(4):
        loss = criterion(model(X[i : i + 1]), Y[i : i + 1]) / 4
        loss.backward()
    accumulated = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

    diff = (whole_batch - accumulated).abs().max().item()
    print(f"one backward over 4 examples vs 4 accumulated backwards")
    print(f"max difference = {diff:.3e}")
    assert diff < 1e-6, "accumulation should reproduce the whole-batch gradient"

    print("\nIdentical. This is gradient accumulation -- how you train with an")
    print("effective batch size larger than your memory allows. The same")
    print("behaviour that bites you in Part 2 is the feature here.")


def main() -> None:
    part_one()
    part_two()
    part_three()

    print(
        "\nAnswers:\n"
        "  1. `.grad` is accumulated with += because a parameter can receive\n"
        "     gradient from several paths in one backward pass -- the exact\n"
        "     decision you made in Part A of this chapter. PyTorch cannot\n"
        "     tell 'another path in this step' from 'a new step', so it\n"
        "     leaves the clearing to you.\n\n"
        "  2. Without clearing, step k applies the SUM of the gradients from\n"
        "     steps 1..k. The effective learning rate grows without bound,\n"
        "     which is why the gradient norm above is orders of magnitude\n"
        "     larger and the loss is worse.\n\n"
        "  3. Nothing raises. The loop runs, the loss changes, the model\n"
        "     trains -- badly. This is the same class of failure as the\n"
        "     one-path gradient in Chapter 3's Task 2: silently wrong beats\n"
        "     loudly broken, and not in a good way.\n\n"
        "  4. Part 3 is the legitimate use: split a batch too large for\n"
        "     memory into chunks, backward each, and step once. Same\n"
        "     gradient, less memory -- standard practice when training\n"
        "     large models."
    )


if __name__ == "__main__":
    main()
