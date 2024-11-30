# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # ELM Prototype
# Testing out banded solver for large ELM models.

# %% 
# Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# %% 
# Generate simple training data
np.random.seed(42)  # For reproducibility
num_samples = 1000

# Generate features
X1 = np.random.rand(num_samples)
X2 = np.random.rand(num_samples)

# Generate target variable with some noise
y = 3 * X1 + 2 * X2 + np.random.randn(num_samples) * 0.1

# Create a DataFrame
data = pd.DataFrame({
    'Feature1': X1,
    'Feature2': X2,
    'Target': y
})

# %%

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target variable for training and validation sets
X_train = train_data[['Feature1', 'Feature2']].values
y_train = train_data[['Target']].values
X_val = val_data[['Feature1', 'Feature2']].values
y_val = val_data[['Target']].values

# %%
# Basic data setup and normalization

# Compute normalization parameters on training set
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# Apply normalization to training and validation sets
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

print(X_train)
print(X_val)

# %% 
# Basic ELM setup

n_inputs = 2
n_neurons = 3

W = np.random.randn(n_inputs, n_neurons)
bias = np.random.randn(1, n_neurons)


H_train = np.tanh(X_train @ W + bias)
H_train = np.hstack([np.ones((H_train.shape[0], 1)), H_train])

print(H_train)

# %%

res = np.linalg.lstsq(H_train.T @ H_train, H_train.T @ y_train, rcond=None)
B = res[0]
for r in res:
    print(r)

# %%

H_val = np.hstack([
    np.ones((X_val.shape[0], 1)),
    np.tanh(X_val @ W + bias)
])

yh_val = H_val @ B

# compute MSE from sklearn
mse = sklearn.metrics.mean_squared_error(y_val, yh_val)
print(f"MSE: {mse}")

# %%

