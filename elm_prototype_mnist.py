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

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

import scipy as sp
from scipy.linalg import solveh_banded, solve_banded

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
n_neurons = 10000

W = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs)
bias = np.random.randn(1, n_neurons)

H_train = np.tanh(X_train @ W + bias)
H_train = np.hstack([np.ones((H_train.shape[0], 1)), H_train])

# print(H_train)

# %%

res = sp.linalg.solve(H_train.T @ H_train, H_train.T @ y_train, assume_a="her")
B = res

# %%
# solve

H_val = np.hstack([
    np.ones((X_val.shape[0], 1)),
    np.tanh(X_val @ W + bias)
])

yh_val = H_val @ B

# compute MSE from sklearn
mse = sklearn.metrics.mean_squared_error(y_val, yh_val)
print(f"MSE: {mse}")

acc_basic = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(yh_val, axis=1))
print(f"Accuracy: {acc_basic}")


# %% [markdown]
# ## Banded Solver section
#
# Accuracy idea:
# 5 neurons: 30%
# 5 neurons 4 bands: 25%
# 100 neurons: 78%
# 100 neurons 4 bands: 40%
# 100 neurons 10 bands: 50%


# %%

HtH = H_train.T @ H_train
HtY = H_train.T @ y_train

# %%
# plot of accuracy vs number of diagonal bands

n_diag = 20
n_diag_values = np.arange(2, n_diag)
accuracies = []
norms = []

# convert HtH to an 'ab' diagonal matrix for banded solver
HtH_ab = np.zeros((n_diag + 1, HtH.shape[1]))
for i in range(0, n_diag + 1):
    HtH_ab[i, :HtH.shape[0] - i] = np.diag(HtH, -i)

for i in n_diag_values:
    # solve using banded solver
    print("n_diag:", i)
    B_banded = None
    for e in range(1, 30)[::-1]:
        HtH_ab_copy = HtH_ab.copy()
        HtH_ab_copy[0, :] += 2**e
        try:
            B_banded = solveh_banded(HtH_ab_copy[:i], HtY, lower=True)
        except:
            break

    # compute predictions
    yh_val_banded = H_val @ B_banded

    # compute accuracy
    acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(yh_val_banded, axis=1))
    accuracies.append(acc)
    norms.append(e)

    print(f"n_diag: {n_diag}, accuracy: {acc}, {HtH_ab_copy.shape}")

# plot accuracy vs n_diag
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('n_diag')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(n_diag_values, accuracies, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Norms', color=color)  # we already handled the x-label with ax1
ax2.plot(n_diag_values, norms, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Accuracy and Norms vs n_diag')
plt.grid(True)
plt.show()



# %%
# compare accuracy of regular solver with the same regularization strength


res = sp.linalg.solve(
    H_train.T @ H_train + np.eye(H_train.shape[1]) * 1e5, 
    H_train.T @ y_train,
    assume_a="her"
)
B_reg = res


# %%
H_val = np.hstack([
    np.ones((X_val.shape[0], 1)),
    np.tanh(X_val @ W + bias)
])

yh_reg_val = H_val @ B_reg

acc_reg = sklearn.metrics.accuracy_score(np.argmax(y_val, axis=1), np.argmax(yh_reg_val, axis=1))
print(f"Accuracy: {acc_reg}")

# %%
# try general banded solver

HtH = H_train.T @ H_train + np.eye(H_train.shape[1]) * 1e+5
HtY = H_train.T @ y_train


n_diag = 10000
n_diag_values = np.arange(200, n_diag, 200)
accuracies = []

# convert HtH to an 'ab' diagonal matrix for banded solver
HtH_ab = np.zeros((n_diag*2 + 1, HtH.shape[1]))
for i in range(-n_diag, n_diag + 1):
    if i < 0:
        HtH_ab[i + n_diag, -i:] = np.diag(HtH, i)
    else:
        HtH_ab[i + n_diag, :HtH.shape[0] - i] = np.diag(HtH, -i)

for i in n_diag_values:
    # solve using banded solver
    if i%20 == 0:
        print(".", end="")
    B_banded = solve_banded((i, i), HtH_ab[n_diag-i : n_diag + 1 + i], HtY)

    # compute predictions
    yh_val_banded = H_val @ B_banded

    # compute accuracy
    acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(yh_val_banded, axis=1))
    accuracies.append(acc)

# plot accuracy vs n_diag
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('n_diag')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(n_diag_values, accuracies, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax1.plot(n_diag_values, [acc_basic]*len(n_diag_values), color=color)

color = 'tab:orange'
ax1.plot(n_diag_values, [acc_reg]*len(n_diag_values), color=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Accuracy vs n_diag')
plt.ylim([0, acc_basic + 0.1])
plt.grid(True)
plt.show()

# %%
# TODO: try Cholesky banded solver

