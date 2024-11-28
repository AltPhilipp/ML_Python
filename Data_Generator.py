import numpy as np
import pandas as pd

# Generate random inputs (features)
num_samples = 100
num_features = 3
np.random.seed(42)
inputs = np.random.randint(0, 2, size=(num_samples, num_features))  # Binary features (0 or 1)

# Generate corresponding outputs
# Rule: output is 1 if the sum of inputs > 1, otherwise 0
outputs = np.where(np.sum(inputs, axis=1) > 1, 1, 0).reshape(-1, 1)

# Combine inputs and outputs into a single dataset
data = np.hstack((inputs, outputs))

# Save dataset to CSV
dataset_file = "training_data.csv"
pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(num_features)] + ["Output"]).to_csv(dataset_file, index=False)
print(f"Dataset saved to {dataset_file}")
