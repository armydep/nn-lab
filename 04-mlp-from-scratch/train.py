"""Chapter 4 starter: a tiny MLP that learns XOR.

This is Chapter 3's two-layer graph with matrices instead of scalars.
Remember the shape rule: a gradient has the same shape as what it differentiates.
"""

# TODO: XOR dataset -- X shape (4, 2), Y shape (4, 1)
# TODO: initialize W1, b1, W2, b2  (random, NOT zeros -- see README)
# TODO: forward pass: Z1 = X @ W1 + b1 -> H = tanh(Z1) -> Z2 = H @ W2 + b2
# TODO: loss (mean squared error over the batch)
# TODO: backpropagate all gradients manually
#       work backwards: dZ2 -> dW2, db2 -> dH -> dZ1 -> dW1, db1
# TODO: gradient-check every parameter before trusting the training loop
# TODO: training loop, printing loss and predictions
