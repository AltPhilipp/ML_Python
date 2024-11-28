import numpy as np

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

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
epochs = 10  # Increase epochs for more training
learning_rate = 0.1  # Introduce a learning rate for weight updates

for i in range(epochs):
    # Forward pass
    f_input = np.dot(training_input, weights)  # Matrix multiplication
    output = step_function(f_input)  # Binary activation function

    # Compute error
    output_error = training_output - output
    print(f"Output error after epoch {i+1}: \n{output_error}")

    # Weight adjustments
    weights += learning_rate * np.dot(training_input.T, output_error)  # Update weights

    print(f"New weights after epoch {i+1}: \n{weights}")

# Final output after forward pass
f_input = np.dot(training_input, weights)
output = step_function(f_input)
print(f"Output after training: \n{output}")
