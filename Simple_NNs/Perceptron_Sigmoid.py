import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(output):
    return output * (1 - output)

# Training data
training_input = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]])  # 4x3 training input matrix

training_output = np.array([[0], [1], [1], [0]])  # 4x1 training output matrix

# Initialize weights randomly
np.random.seed(1)  # For reproducibility
weights = 2 * np.random.random((3, 1)) - 1  # Random weights from -1 to 1

print(f"Random starting synaptic weights: \n{weights}")

# Training parameters
epochs = 10000

for i in range(epochs):
    # Forward pass
    f_input = np.dot(training_input, weights)  # Matrix multiplication
    output = sigmoid(f_input)

    # Compute error
    output_error = training_output - output
    print(f"Output error: \n{output_error}")

    # Weight adjustments
    adjustment = np.dot(training_input.T, output_error * sigmoid_derivative(output))
    weights += adjustment  # Update weights

    print(f"New weights: \n{weights}")

# Final output after forward pass
f_input = np.dot(training_input, weights)
output = sigmoid(f_input)
print(f"Output after forward pass: \n{output}")
