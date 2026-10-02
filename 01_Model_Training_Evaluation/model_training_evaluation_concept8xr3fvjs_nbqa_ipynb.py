# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# Plot regression and classification loss functions
e = np.linspace(-3, 3, 300)  # error y - y_hat (regression)
m = np.linspace(-2, 3, 300)  # margin y * score (classification)

# Figure showing loss functions for regression and classification
# Left: regression losses as a function of the error
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(e, e**2, label='squared loss $(y-\\hat y)^2$')
axes[0].plot(e, np.abs(e), label='absolute loss $|y-\\hat y|$')
axes[0].set_xlabel('error $y - \\hat y$')
axes[0].set_ylabel('loss')
axes[0].set_title('Regression losses')
axes[0].legend()

# Right: classification losses as a function of the margin
axes[1].plot(m, (m <= 0).astype(float), label='0-1 loss')
axes[1].plot(m, np.log2(1 + np.exp(-m)), label='log-loss (base 2)')
axes[1].plot(m, np.maximum(0, 1 - m), label='hinge loss')
axes[1].axvline(0, color='grey', linewidth=0.8)
axes[1].set_xlabel('margin $m = y \\cdot s$  (m > 0: correct side)')
axes[1].set_title('Classification losses')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Simulate rents and plot the MSE as a function of the slope
rng = np.random.default_rng(1)
area = rng.uniform(30, 140, 40)
rent = 160 + 30 * area + rng.normal(0, 250, 40)


def mse_for_slope(slope):
    """MSE of the line with this slope and the best intercept."""
    intercept = np.mean(rent - slope * area)  # optimal intercept for this slope
    return np.mean((rent - (intercept + slope * area)) ** 2)


# MSE for a range of slopes; the minimum is the OLS solution
slopes = np.linspace(5, 55, 200)
mse = [mse_for_slope(s) for s in slopes]
best = slopes[np.argmin(mse)]

# Figure showing three candidate lines (left) and the MSE by slope (right)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(area, rent, s=15, color='black')
for s, label in [(15, 'line A'), (best, 'line B (OLS)'), (45, 'line C')]:
    intercept = np.mean(rent - s * area)
    x = np.array([25, 145])
    axes[0].plot(
        x,
        intercept + s * x,
        label=f'{label}: slope {s:.1f}, MSE = {mse_for_slope(s) / 1e4:.1f}e4',
    )
axes[0].set_xlabel('living area [m²]')
axes[0].set_ylabel('rent [CHF/month]')
axes[0].set_title('Three candidate models')
axes[0].legend(fontsize=8)
axes[1].plot(slopes, np.array(mse) / 1e4)
axes[1].axvline(
    best,
    color='darkred',
    linestyle='--',
    label=f'minimum = OLS solution (slope {best:.1f})',
)
axes[1].set_xlabel('slope $\\beta_1$ (intercept set optimally)')
axes[1].set_ylabel('average loss: MSE [$10^4$ CHF²]')
axes[1].set_title('Loss as a function of the parameter')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Visualise the folds of a 5-fold CV for 20 observations
kf = KFold(n_splits=5, shuffle=True, random_state=0)
grid = np.zeros((5, 20))
for run, (train_idx, val_idx) in enumerate(kf.split(np.arange(20))):
    grid[run, val_idx] = 1

# Figure showing which observations are validated in each run
plt.figure(figsize=(9, 2.5))
plt.imshow(grid, cmap='coolwarm', aspect='auto', vmin=-0.3, vmax=1.3)
plt.yticks(range(5), [f'run {i + 1}' for i in range(5)])
plt.xticks(range(20), range(1, 21), fontsize=7)
plt.xlabel('observation (training set)')
plt.title('5-fold cross-validation: blue = train, red = validate')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Fit polynomials of degree 1, 4 and 15 to noisy points of a sine curve
rng = np.random.default_rng(3)
x = np.sort(rng.uniform(0, 1, 18))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 18)
x_test = rng.uniform(0, 1, 200)
y_test = np.sin(2 * np.pi * x_test) + rng.normal(0, 0.25, 200)
x_grid = np.linspace(0, 1, 400)


def poly_model(degree):
    """Polynomial regression model of the given degree."""
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


# Figure showing the fits of degree 1, 4 and 15 with train and test MSE
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (d, title) in zip(
    axes, [(1, 'Underfitting'), (4, 'Good fit'), (15, 'Overfitting')]
):
    m = poly_model(d).fit(x.reshape(-1, 1), y)
    train_mse = np.mean((y - m.predict(x.reshape(-1, 1))) ** 2)
    test_mse = np.mean((y_test - m.predict(x_test.reshape(-1, 1))) ** 2)
    ax.plot(
        x_grid,
        np.sin(2 * np.pi * x_grid),
        'k--',
        linewidth=1,
        label='true function',
    )
    ax.plot(
        x_grid,
        m.predict(x_grid.reshape(-1, 1)),
        linewidth=2,
        label=f'degree {d}',
    )
    ax.scatter(x, y, color='black', s=20)
    ax.set_ylim(-2, 2)
    ax.set_title(
        f'{title} (degree {d})\n'
        f'train MSE = {train_mse:.3f}, test MSE = {test_mse:.3f}'
    )
    ax.legend(loc='lower left', fontsize=8)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Simulate bias² and variance of polynomial models of increasing degree
rng = np.random.default_rng(0)
x_eval = np.linspace(0.1, 0.9, 50)
f_true = np.sin(2 * np.pi * x_eval)
degrees = list(range(1, 13))
bias2, var = [], []
for d in degrees:
    preds = []
    for _ in range(300):  # 300 different training samples of 40 points
        xs = rng.uniform(0, 1, 40)
        ys = np.sin(2 * np.pi * xs) + rng.normal(0, 0.25, 40)
        preds.append(
            poly_model(d)
            .fit(xs.reshape(-1, 1), ys)
            .predict(x_eval.reshape(-1, 1))
        )
    preds = np.array(preds)
    bias2.append(np.mean((preds.mean(axis=0) - f_true) ** 2))
    var.append(np.mean(preds.var(axis=0)))

# Figure showing bias², variance and their sum by polynomial degree
noise = 0.25**2
plt.figure(figsize=(7, 4))
plt.plot(degrees, bias2, marker='o', label='bias²')
plt.plot(degrees, var, marker='o', label='variance')
plt.plot(
    degrees,
    np.array(bias2) + np.array(var),
    marker='o',
    linewidth=2.5,
    label='bias² + variance',
)
plt.yscale('log')
plt.xlabel('polynomial degree (model complexity)')
plt.ylabel('expected squared error (log scale)')
plt.title(
    'Bias–variance trade-off (simulation, n = 40; '
    f'noise σ² = {noise:.4f} is added to all models)'
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
