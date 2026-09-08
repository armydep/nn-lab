"""Chapter 2: train one neuron to classify examples into class 0 or class 1."""

import numpy as np


# Step 1: Create the training data.
# Each row of X will be one example with two features: [x1, x2].
# Each value in Y will be that example's correct class: either 0 or 1.

# TODO: replace with a 2D NumPy array
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

# TODO: replace with a 1D NumPy array
Y = np.array([0.0, 0.0, 0.0, 1.0])


# Step 2: Start the neuron with values it can learn and improve.
w = np.array([0.0, 0.0])
b = 0.0

LEARNING_RATE = 0.1
EPOCHS = 1000

# Step 3: Calculate the neuron's raw score: z = w1*x1 + w2*x2 + b.
# TODO: implement the weighted sum
def linear(X, w, b):
    return np.dot(X, w) + b



# Step 4: Convert each raw score into a probability between 0 and 1.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Step 5: Measure how well the predicted probabilities match Y.
# TODO: clip probabilities, then calculate the loss
def binary_cross_entropy(probabilities, y):
    safe_probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    losses = -(y * np.log(safe_probabilities) + (1 - y) * np.log(1 - safe_probabilities))
    return float(np.mean(losses))

def main():
    global w, b

    # Step 6: Repeat prediction, loss, gradient, and update.
    for epoch in range(EPOCHS):
        # TODO: calculate z
        z = linear(X, w, b)
        # TODO: calculate probabilities
        probabilities = sigmoid(z)
        # TODO: calculate loss
        loss = binary_cross_entropy(probabilities, Y)
        error = probabilities - Y
        # TODO: calculate dw and db manually
        dw = X.T.dot(error) / len(X)
        db = np.mean(error)
        # TODO: update w and b
        w = w - LEARNING_RATE * dw
        b = b - LEARNING_RATE * db
        # TODO: print occasional progress
        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss={loss:.6f}, w={w}, b={b:.4f}")

    # Step 7: Inspect final probabilities and predicted classes.
    # TODO: print each example, probability, predicted class, and true class
    probabilities = sigmoid(linear(X, w, b))
    predicted_classes = (probabilities >= 0.5).astype(int)
    for x, probability, predicted_class, true_class in zip(
        X, probabilities, predicted_classes, Y
    ):
        print(
            f"x={x}, probability={probability:.4f}, "
            f"predicted={predicted_class}, actual={int(true_class)}"
        )

if __name__ == "__main__":
    main()
