"""Chapter 1 starter: implement a single trainable linear neuron."""

# TODO: create x and y training data
# TODO: initialize w and b
# TODO: implement forward pass
# TODO: implement MSE loss
# TODO: derive and calculate dw and db manually
# TODO: update w and b
# TODO: repeat for multiple epochs and print progress

import numpy as np

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([1.0, 3.0, 5.0, 7.0])

w = 0.0
b = 0.0

LEARNING_RATE = 0.001
EPOCHS = 100000

def forward(x, w, b):
    return x * w + b

def mse(predictions, y):
    return float(np.mean((predictions - y) ** 2))


def main():
    global w, b

    print("\nChapter 1 starter")
    for epoch in range(EPOCHS):
        # 1. Predict
        predictions = forward(X, w, b)
        # 2. Measure error
        loss = mse(predictions, Y)

        # 3. Calculate gradients
        error = predictions - Y
        dw = 2 * np.mean(error * X)
        db = 2 * np.mean(error)

        # Print progress for the current parameters
        if epoch % 1000 == 0:
            print(f"epoch={epoch}, loss={loss:.6f}, w={w:.4f}, b={b:.4f}")

        # 4. Update parameters for the next epoch
        w = w - LEARNING_RATE * dw
        b = b - LEARNING_RATE * db


if __name__ == "__main__":
    main()
