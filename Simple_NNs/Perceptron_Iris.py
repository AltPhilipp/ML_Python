import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features for simplicity
y = (iris.target != 0).astype(int)  # Convert to binary classification: 1 for not Setosa, 0 for Setosa

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (scale to mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize weights and bias
np.random.seed(7474)
weights = 2 * np.random.random((2, 1)) - 1  # Two features, one output
bias = np.random.random(1) - 0.5  # Initialize bias

# Training parameters
epochs = 2
learning_rate = 0.01

# Reshape y_train to match dimensions
y_train = y_train.reshape(-1, 1)

# Training loop
for epoch in range(epochs):
    # Forward pass
    f_input = np.dot(X_train, weights) + bias
    output = step_function(f_input)

    # Compute error
    output_error = y_train - output

    # Update weights and bias
    weights += learning_rate * np.dot(X_train.T, output_error)
    bias += learning_rate * np.sum(output_error)

# Test the model
f_input_test = np.dot(X_test, weights) + bias
y_pred = step_function(f_input_test)

# Accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# Print test results
print("\nTest Data Evaluation:")
for i in range(len(y_test)):
    predicted = y_pred[i][0]
    actual = y_test[i]
    status = "Correct" if predicted == actual else "Wrong"
    print(f"Test example {i+1}: Predicted = {predicted}, Actual = {actual} -> {status}")

# Visualization
def plot_decision_boundary(X, y, weights, bias):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Compute decision boundary
    Z = step_function(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot decision boundary with training data
plot_decision_boundary(X_train, y_train.flatten(), weights, bias)
