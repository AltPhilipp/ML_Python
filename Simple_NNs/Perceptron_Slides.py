import numpy as np


# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)


# Training Data
training_input = np.array([[1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 1, 1],
                           [0, 0, 1],
                           [0, 1, 0]])

training_output = np.array([[0], [1], [1], [1], [0], [0]])

# Test Data
test_input = np.array([[0, 1, 1],
                       [0, 0, 0]])

test_output = np.array([[1], [0]])

# Initialize parameters
bias = 0.4
learning_rate = 0.1

np.random.seed(1)  # For reproducibility
weights = 2 * np.random.random((3, 1)) - 1  # Random weights from -1 to 1

epochs = 1

# Training loop
for epoch in range(epochs):
    for i in range(len(training_input)):
        # Forward pass
        prediction = step_function(np.dot(training_input[i], weights) + bias)

        # Compute error
        error = training_output[i] - prediction

        # Update weights and bias
        weights += learning_rate * error * training_input[i].reshape(-1, 1)
        bias += learning_rate * error

    print(f"Weights after epoch {epoch + 1}:\n{weights}")
    print(f"Bias after epoch {epoch + 1}: {bias}")

# Testing loop
print("\nTesting the model:")
for i in range(len(test_input)):
    prediction = step_function(np.dot(test_input[i], weights) + bias)
    print(f"Test example {i + 1}: Predicted = {prediction[0]}, Actual = {test_output[i][0]}")
