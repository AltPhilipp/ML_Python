import numpy as np
import pandas as pd

from Simple_NNs.Perceptron_Slides import step_function

# Load the dataset
data = pd.read_csv("training_data.csv")
training_input = data.iloc[:, :-1].values  # All columns except the last are inputs
training_output = data.iloc[:, -1].values.reshape(-1, 1)  # The last column is output

# Test Data (manually define or split from the dataset)
test_input = np.array([[0, 1, 1],
                       [0, 0, 0]])
test_output = np.array([[1], [0]])

# Initialize parameters
bias = 0.4
learning_rate = 0.1
np.random.seed(1)
weights = 2 * np.random.random((training_input.shape[1], 1)) - 1  # Random weights for the number of features

# Training loop with convergence check
epochs = 1000
for epoch in range(epochs):
    all_correct = True
    for i in range(len(training_input)):
        # Forward pass
        prediction = step_function(np.dot(training_input[i], weights) + bias)

        # Compute error
        error = training_output[i] - prediction

        # Update weights and bias if there's an error
        if error != 0:
            all_correct = False
            weights += learning_rate * error * training_input[i].reshape(-1, 1)
            bias += learning_rate * error

    # Stop training if all predictions are correct
    if all_correct:
        print(f"Training converged after epoch {epoch + 1}")
        break

# Test the model
print("\nTesting the model:")
for i in range(len(test_input)):
    prediction = step_function(np.dot(test_input[i], weights) + bias)
    print(f"Test example {i + 1}: Predicted = {prediction[0]}, Actual = {test_output[i][0]}")
