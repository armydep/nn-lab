"""Chapter 5 Part B solution: rebuild the XOR MLP with PyTorch.

Every line here replaces something you wrote by hand in Chapter 4.

    python3 solutions/05-pytorch-mlp/train.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(1)

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

HIDDEN = 4
LEARNING_RATE = 0.5
EPOCHS = 2000


class XORNet(nn.Module):
    """Identical architecture to Chapter 4: 2 -> HIDDEN -> tanh -> 1."""

    def __init__(self) -> None:
        super().__init__()
        # nn.Linear holds the W and b you allocated by hand, and initialises
        # them randomly for you -- the symmetry breaking from Chapter 4.
        self.layer1 = nn.Linear(2, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.layer1(x))
        return self.layer2(h)


def show_manual_autograd_equivalence() -> None:
    """Confirm PyTorch's .grad matches a hand-derived gradient."""
    print("=" * 66)
    print("Does autograd agree with your Chapter 3 derivation?")
    print("=" * 66)

    # Chapter 3 step 3, in PyTorch. Same numbers.
    x, target = 1.5, 0.8
    w1 = torch.tensor(0.4, requires_grad=True)
    b1 = torch.tensor(-0.2, requires_grad=True)
    w2 = torch.tensor(0.9, requires_grad=True)
    b2 = torch.tensor(0.1, requires_grad=True)

    z1 = w1 * x + b1
    h = torch.tanh(z1)
    z2 = w2 * h + b2
    loss = (z2 - target) ** 2
    loss.backward()

    print(f"  PyTorch:        dL/dw1 = {w1.grad:+.6f}   dL/db1 = {b1.grad:+.6f}")
    print(f"                  dL/dw2 = {w2.grad:+.6f}   dL/db2 = {b2.grad:+.6f}")
    print("  Chapter 3 by hand: dL/dw1 = -0.827167   dL/db1 = -0.551444")
    print("                   dL/dw2 = -0.272078   dL/db2 = -0.716092")
    assert abs(w1.grad.item() - (-0.827167)) < 1e-5
    assert abs(b1.grad.item() - (-0.551444)) < 1e-5
    assert abs(w2.grad.item() - (-0.272078)) < 1e-5
    assert abs(b2.grad.item() - (-0.716092)) < 1e-5
    print("\n  Identical. loss.backward() is your Chapter 3 backward pass,")
    print("  discovered automatically from the recorded graph.\n")


def main() -> None:
    show_manual_autograd_equivalence()

    model = XORNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 66)
    print(f"Training XOR ({n_params} parameters)")
    print("=" * 66)

    for epoch in range(EPOCHS + 1):
        predictions = model(X)                 # forward  (Chapter 4: forward())
        loss = loss_fn(predictions, Y)         # loss     (Chapter 4: loss_fn())

        optimizer.zero_grad()                  # clear    (Chapter 4: implicit)
        loss.backward()                        # backward (Chapter 4: backward())
        optimizer.step()                       # update   (Chapter 4: p -= lr*g)

        if epoch % 200 == 0:
            print(f"  epoch {epoch:5d}   loss {loss.item():.8f}")

    with torch.no_grad():
        predictions = model(X)
        final_loss = loss_fn(predictions, Y).item()

    print(f"\nFinal loss: {final_loss:.8f}\n")
    print("XOR truth table:")
    correct = 0
    for x_i, y_i, pred in zip(X, Y, predictions):
        rounded = 1 if pred.item() >= 0.5 else 0
        correct += rounded == int(y_i.item())
        print(
            f"  {int(x_i[0])} XOR {int(x_i[1])} = {int(y_i.item())}   "
            f"predicted {pred.item():+.4f} -> {rounded}"
        )
    print(f"\nCorrect: {correct}/4")
    assert correct == 4, "XOR was not learned -- try a different seed"

    print("\n" + "=" * 66)
    print("What replaced what")
    print("=" * 66)
    print(
        "  Chapter 4, by hand                  PyTorch\n"
        "  ------------------------------    ----------------------------\n"
        "  cache = {z1, h, z2}               the autograd graph\n"
        "  backward(): dW2, db2, dW1, db1    loss.backward()\n"
        "  for name in p: p -= lr * g        optimizer.step()\n"
        "  (gradients rebuilt every step)    optimizer.zero_grad()\n"
        "  rng.normal(...) init              nn.Linear's default init\n\n"
        "  Nothing here is new mathematics. PyTorch records the graph you\n"
        "  used to trace by hand, and walks it in the order your Part A\n"
        "  topological sort works out. That is the whole of it."
    )


if __name__ == "__main__":
    main()
