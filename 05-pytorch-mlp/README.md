# Chapter 5 — Build a Mini-Autograd, Then Rebuild the MLP With PyTorch

## Goal

First write a ~50-line autograd engine yourself. Then rebuild Chapter 4's network with PyTorch
and recognise every piece of it.

## Why the engine comes first

The usual way to learn PyTorch is to be handed `loss.backward()` and told it computes
gradients. You then spend months with a hole where that understanding should be.

You have already done backpropagation by hand twice — Chapter 3 on scalars, Chapter 4 on matrices.
Both times you wrote the backward pass manually, in the right order, by knowing the graph
ahead of time. An autograd engine does exactly one new thing: it **records** the graph as the
forward pass runs, so the backward order can be worked out automatically.

Build that, and `loss.backward()` stops being magic — it becomes code you have already written.

## Part A — Mini-autograd engine

### Concepts

Dynamic computation graph, node, parent references, topological ordering, gradient accumulation.

### Implementation requirements

1. Write a `Value` class wrapping a single number, with fields `data`, `grad`, `_parents`,
   and `_backward`.
2. Implement `__add__` and `__mul__`. Each returns a new `Value` that remembers its parents
   and knows how to push gradient to them.
3. Add `tanh` (or `sigmoid`).
4. Implement `backward()`: topologically sort the graph, seed `self.grad = 1.0`, then walk
   backwards calling each node's `_backward`.
5. **Accumulate** gradients with `+=`, never `=`. A value used twice receives gradient from
   both paths — this is the bug everyone writes once.
6. Verify: rebuild Chapter 3's two-layer graph with `Value` objects and confirm the gradients
   match what you derived by hand.

### Definition of done for Part A

Your engine reproduces your Chapter 3 gradients, and you can explain why the topological sort is
necessary — what breaks if you process nodes in the wrong order.

## Part B — PyTorch

### Concepts

Tensor, `requires_grad`, computation graph, `nn.Module`, `loss.backward()`, optimizer, SGD/Adam,
batching.

### Implementation requirements

1. Rebuild XOR using PyTorch tensors.
2. Define the model with `nn.Module`.
3. Use a PyTorch loss function.
4. Call `loss.backward()`.
5. Inspect `.grad` values — confirm they match your Chapter 4 hand-derived gradients.
6. Use an optimizer.
7. Compare each PyTorch operation with the corresponding Chapter 4 code.

### The mapping to internalize

| You wrote | PyTorch does it with | Which is your |
|---|---|---|
| Storing intermediates in the forward pass | The autograd graph | Part A `_parents` |
| Manual `dW2`, `db2`, `dW1`, `db1` | `loss.backward()` | Part A `backward()` |
| `W1 -= lr * dW1` for every parameter | `optimizer.step()` | Chapter 4 update loop |
| Clearing gradients between steps | `optimizer.zero_grad()` | The `+=` in Part A |

That last row is worth dwelling on. `zero_grad()` exists **because** gradients accumulate — the
same design decision you made in Part A step 5. Forget it and your gradients silently sum
across steps.

## Definition of done

The PyTorch model learns XOR, and you can explain exactly what autograd and `optimizer.step()`
replace from your manual implementation — because you built a small version of both.

## Optional practice

Extra drills in `optional/`, one per part of this chapter.

| File | New idea |
|---|---|
| `optional/t2.py` | Add `exp` and `log` to your engine, then watch it re-derive `dL/dz = p - y` |
| `optional/t3.py` | Omit `zero_grad()` and measure the damage — then use accumulation on purpose |

## Reference solutions

- `solutions/05-pytorch-mlp/micrograd.py` — the mini-autograd engine
- `solutions/05-pytorch-mlp/train.py` — the PyTorch rebuild
- `solutions/05-pytorch-mlp/optional/` — the drills

## Further reading

Karpathy's [micrograd video](https://www.youtube.com/watch?v=VMj-3vsdGqc); PyTorch [autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

Full list: [`RESOURCES.md`](../RESOURCES.md)
