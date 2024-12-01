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
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

import scipy as sp

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

# %% 
# Basic ELM setup

n_inputs = 784
n_neurons = 10000

W = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs)
bias = np.random.randn(1, n_neurons)

H_train = np.tanh(X_train @ W + bias)
H_train = np.hstack([np.ones((H_train.shape[0], 1)), H_train])

H_val = np.tanh(X_val @ W + bias)
H_val = np.hstack([np.ones((H_val.shape[0], 1)), H_val])

HtH = H_train.T @ H_train
HtY = H_train.T @ y_train


# %%
# pre-heat solvers

B = np.linalg.solve(HtH, HtY)

# %%
# Solvers and timing

t = time()
B = np.linalg.solve(HtH, HtY)
t = time() - t
acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
print(f"Basic solve. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

t = time()
B = sp.linalg.solve(HtH, HtY)
t = time() - t
acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
print(f"Scipy solve. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

t = time()
B = sp.linalg.solve(HtH, HtY, assume_a="her")
t = time() - t
acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
print(f"Hermitian solve. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

# solve with scipy and cholesky decomposition
t = time()
L = sp.linalg.cho_factor(HtH)
B = sp.linalg.cho_solve(L, HtY)
t = time() - t
acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
print(f"Cholesky solve. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

# solve with scipy and LU decomposition
t = time()
lu, piv = sp.linalg.lu_factor(HtH)
B = sp.linalg.lu_solve((lu, piv), HtY)
t = time() - t
acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
print(f"LU solve. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

# %%

def make_banded_matrix(HtH, n_diag):
    HtH_ab = np.zeros((2 * n_diag + 1, HtH.shape[1]))
    for i in range(-n_diag, n_diag + 1):
        if i < 0:
            HtH_ab[i + n_diag, -i:] = np.diag(HtH, i)
        else:
            HtH_ab[i + n_diag, :HtH.shape[0] - i] = np.diag(HtH, -i)
    return HtH_ab

# %%
# general banded solve

# take 1%, 5%, 10%, 20%, 50% and 100% of n_neurons
for n_percent in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 1]:
    n_diag = int(n_neurons * n_percent)
    
    HtH_ab = make_banded_matrix(HtH, n_diag)
    t = time()
    B = sp.linalg.solve_banded((n_diag, n_diag), HtH_ab, HtY)
    t = time() - t

    acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
    print(f"Banded solve for {100*n_percent:.0f}% neurons. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%")

# %%
# general banded solve with high norm

# take 1%, 5%, 10%, 20%, 50% and 100% of n_neurons
norm = 15
for n_percent in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 1]:
    n_diag = int(n_neurons * n_percent)
    
    HtH_ab = make_banded_matrix(HtH + (2**norm)*np.eye(HtH.shape[0]), n_diag)
    t = time()
    B = sp.linalg.solve_banded((n_diag, n_diag), HtH_ab, HtY)
    t = time() - t

    acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
    print(f"Banded solve for {100*n_percent:.0f}% neurons. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%, Norm: 2^{norm}")

# %% 
# general Hermitian banded solve

# take 1%, 5%, 10%, 20%, 50% and 100% of n_neurons
for n_percent in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 1]:
    n_diag = int(n_neurons * n_percent)
    for norm in range(-20, 30):
        try:
            HtH_ab_lower = make_banded_matrix(HtH + (2**norm) * np.eye(HtH.shape[0]), n_diag)[n_diag:]
            t = time()
            B = sp.linalg.solveh_banded(HtH_ab_lower, HtY, lower=True)
            t = time() - t
            break
        except np.linalg.LinAlgError:
            continue

    acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
    print(f"Banded Hermitian solve for {100*n_percent:.0f}% neurons. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%, Norm: 2^{norm}")


# %% 
# cholesky banded solve

# take 1%, 5%, 10%, 20%, 50% and 100% of n_neurons
for n_percent in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 1]:
    n_diag = int(n_neurons * n_percent)
    is_lower = True
    for norm in range(-20, 30):
        try:
            HtH_ab_lower = make_banded_matrix(HtH + (2**norm) * np.eye(HtH.shape[0]), n_diag)[n_diag:]
            t = time()
            C = sp.linalg.cholesky_banded(HtH_ab_lower, lower=is_lower)
            B = sp.linalg.cho_solve_banded((C, is_lower), HtY)
            t = time() - t
            break
        except np.linalg.LinAlgError:
            continue

    acc = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(H_val @ B, axis=1))
    print(f"Banded Cholesky solve for {100*n_percent:.0f}% neurons. Time: {1000*t:.3f}ms, Accuracy: {100*acc:.1f}%, Norm: 2^{norm}")

# %%


