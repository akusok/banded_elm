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
from time import time

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

%matplotlib inline

# %%
# load MNIST dataset

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
X = X.to_numpy(dtype=float)

# One-hot encode the target labels
y = y.to_numpy(dtype=int)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

print(X.shape)
print(y.shape)

# %%

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# %%
# Basic data setup and normalization using RobustScaler

scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and validation sets
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(X_train.mean(1))
print(X_train.std(1))
print(X_val[:2])

# %% 
# Basic ELM setup

n_inputs = 784
n_neurons = 300

W = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs)
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
# solve

H_val = np.hstack([
    np.ones((X_val.shape[0], 1)),
    np.tanh(X_val @ W + bias)
])

yh_val = H_val @ B

# compute MSE from sklearn
mse = sklearn.metrics.mean_squared_error(y_val, yh_val)
print(mse)

# %%
# print ROC curve

# Convert one-hot encoded labels back to single class labels
y_val_single = np.argmax(y_val, axis=1)
yh_val_single = np.argmax(yh_val, axis=1)

fpr, tpr, thresholds = roc_curve(y_val_single == 1, yh_val_single == 1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
