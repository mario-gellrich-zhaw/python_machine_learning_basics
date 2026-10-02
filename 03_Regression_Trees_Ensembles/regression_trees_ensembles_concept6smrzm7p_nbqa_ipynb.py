# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# Simulate data with a box structure and fit a tree with three splits
rng = np.random.default_rng(0)
X = rng.uniform(0, 10, (300, 2))
y = np.where(
    X[:, 0] <= 4,
    10,
    np.where(X[:, 1] <= 6, 25, np.where(X[:, 0] <= 7.5, 40, 55)),
) + rng.normal(0, 3, 300)

# Regression tree with four leaves (three splits)
reg_tree = DecisionTreeRegressor(max_leaf_nodes=4, random_state=0).fit(X, y)
print(tree.export_text(reg_tree, feature_names=['x1', 'x2'], decimals=2))

# Predictions of the tree on a grid of (x1, x2)
xx, yy = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 10, 300))
zz = reg_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# Figure showing the predicted values and the boxes of the tree
plt.figure(figsize=(5.5, 5))
plt.contourf(xx, yy, zz, levels=20, cmap='Blues', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], s=6, color='black', alpha=0.5)
for v in np.unique(zz):  # label each box at its centre
    box = zz == v
    plt.text(
        xx[box].mean(),
        yy[box].mean(),
        f'ŷ = {v:.1f}',
        ha='center',
        bbox={'facecolor': 'white', 'alpha': 0.8},
    )
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Regression tree: one constant per box')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Worked example: SSE of all possible splits
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 12, 14, 30, 34])


def sse(v):
    """Sum of squared errors around the mean."""
    return np.sum((v - v.mean()) ** 2)


# SSE of all possible splits
rows = []
for s in (x[:-1] + x[1:]) / 2:
    left, right = y[x <= s], y[x > s]
    rows.append(
        {
            'split': f'x <= {s}',
            'left node': list(left),
            'mean left': left.mean(),
            'right node': list(right),
            'mean right': right.mean(),
            'SSE(R1) + SSE(R2)': sse(left) + sse(right),
        }
    )
print('Root: mean =', y.mean(), ' SSE =', sse(y))
print(pd.DataFrame(rows).to_string(index=False))


# %%NBQA-CELL-SEP42d9ce
# Pruning: training and CV error along the cost-complexity path
rng = np.random.default_rng(1)
x1 = rng.uniform(0, 10, 300)
y1 = np.sin(x1) * 3 + 0.3 * x1 + rng.normal(0, 1, 300)
X1 = x1.reshape(-1, 1)

# Alphas of the pruning path; training and CV error for each alpha
alphas = (
    DecisionTreeRegressor(random_state=0)
    .cost_complexity_pruning_path(X1, y1)
    .ccp_alphas
)
alphas = alphas[:: max(1, len(alphas) // 40)]
cv = KFold(5, shuffle=True, random_state=0)
leaves, train_mse, cv_mse = [], [], []
for a in alphas:
    m = DecisionTreeRegressor(ccp_alpha=a, random_state=0).fit(X1, y1)
    leaves.append(m.get_n_leaves())
    train_mse.append(np.mean((y1 - m.predict(X1)) ** 2))
    cv_mse.append(
        -cross_val_score(
            DecisionTreeRegressor(ccp_alpha=a, random_state=0),
            X1,
            y1,
            cv=cv,
            scoring='neg_mean_squared_error',
        ).mean()
    )

# Figure showing the training and CV error by number of leaves
plt.figure(figsize=(7, 4))
plt.plot(leaves, train_mse, marker='.', label='training error')
plt.plot(leaves, cv_mse, marker='.', label='5-fold CV error')
best = leaves[int(np.argmin(cv_mse))]
plt.axvline(
    best, color='black', linestyle=':', label=f'optimal size ≈ {best} leaves'
)
plt.xscale('log')
plt.xlabel('number of leaves |T| (tree complexity, log scale)')
plt.ylabel('MSE')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Variance of an average of B correlated models
B = np.arange(1, 201)
# Figure showing the variance of the average by B for different rho
plt.figure(figsize=(7, 4))
for rho in [0.0, 0.2, 0.5, 0.8]:
    plt.plot(B, rho + (1 - rho) / B, label=f'rho = {rho}')
plt.xscale('log')
plt.xlabel('number of models B (log scale)')
plt.ylabel('variance of the average (sigma² = 1)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print('sigma2 = 1, rho = 0.2, B = 100:', round(0.2 + 0.8 / 100, 3))


# %%NBQA-CELL-SEP42d9ce
# Probability that an observation is not in a bootstrap sample
for n in [10, 100, 1000, 10000]:
    print(f'n = {n:5d}: (1 - 1/n)^n = {(1 - 1 / n)**n:.4f}')
print(f'limit e^-1 = {np.exp(-1):.4f}')

# Simulation: one bootstrap sample of n = 1000 observations
rng = np.random.default_rng(0)
n = 1000
sample = rng.integers(0, n, n)
print(
    'Simulation: share of observations not in one bootstrap sample = '
    f'{1 - len(np.unique(sample)) / n:.3f}'
)


# %%NBQA-CELL-SEP42d9ce
# Gradient boosting: one step by hand and the fit after more and more trees
print('One boosting step: 7 + 0.1 * (10 - 7) =', 7 + 0.1 * (10 - 7))

# Gradient boosting with 200 trees; predictions after each tree
x_grid = np.linspace(0, 10, 400).reshape(-1, 1)
gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0
).fit(X1, y1)
staged = list(gb.staged_predict(x_grid))

# Figure showing the fit after 1, 5, 20 and 200 trees
plt.figure(figsize=(8, 4))
plt.scatter(x1, y1, s=6, color='grey', alpha=0.6)
for m_ in [1, 5, 20, 200]:
    plt.plot(x_grid, staged[m_ - 1], linewidth=2, label=f'after {m_} trees')
plt.xlabel('x')
plt.ylabel('y')
plt.title(
    'Gradient boosting: the fit improves step by step '
    '(learning rate 0.1, depth 2)'
)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
