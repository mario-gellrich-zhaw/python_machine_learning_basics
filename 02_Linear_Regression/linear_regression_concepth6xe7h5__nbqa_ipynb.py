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
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# Simulate rents and compute the OLS line by hand
rng = np.random.default_rng(7)
x = rng.uniform(30, 140, 25)
y = 170 + 31 * x + rng.normal(0, 250, 25)

# OLS estimates by hand
beta_1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
beta_0 = y.mean() - beta_1 * x.mean()
print(f'beta_0 = {beta_0:.2f}, beta_1 = {beta_1:.2f}')

# Figure showing the data, the OLS line and the residuals
plt.figure(figsize=(7, 4))
plt.scatter(x, y, color='black', s=15, zorder=3)
xx = np.array([25, 145])
plt.plot(xx, beta_0 + beta_1 * xx, color='tab:blue', label='OLS line')
plt.vlines(
    x,
    y,
    beta_0 + beta_1 * x,
    color='tab:orange',
    linewidth=1,
    label='residuals',
)
plt.scatter(
    [x.mean()],
    [y.mean()],
    color='darkred',
    s=60,
    zorder=4,
    label='point of means',
)
plt.xlabel('living area [m²]')
plt.ylabel('rent [CHF/month]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Prediction of the OLS line for a 95 m² flat
beta_0, beta_1 = 166.79, 30.84
print(f'Predicted rent for 95 m2: {beta_0 + beta_1 * 95:.2f} CHF')


# %%NBQA-CELL-SEP42d9ce
# R² and RMSE of the small examples
print('R2 =', 1 - 20 / 100)
res = np.array([10, 5, 2, 3, 4])
print('RMSE =', round(np.sqrt(np.mean(res**2)), 2))


# %%NBQA-CELL-SEP42d9ce
# Confidence and prediction intervals with statsmodels
df = pd.DataFrame({'area': x, 'rent': y})
model = smf.ols('rent ~ area', data=df).fit()
grid = pd.DataFrame({'area': np.linspace(25, 145, 100)})
bands = model.get_prediction(grid).summary_frame(alpha=0.05)

# Figure showing the confidence and the prediction interval
plt.figure(figsize=(7, 4.5))
plt.fill_between(
    grid['area'],
    bands['obs_ci_lower'],
    bands['obs_ci_upper'],
    color='tab:orange',
    alpha=0.3,
    label='95% prediction interval',
)
plt.fill_between(
    grid['area'],
    bands['mean_ci_lower'],
    bands['mean_ci_upper'],
    color='tab:blue',
    alpha=0.4,
    label='95% confidence interval',
)
plt.plot(grid['area'], bands['mean'], color='tab:blue')
plt.scatter(df['area'], df['rent'], color='black', s=15)
plt.xlabel('living area [m²]')
plt.ylabel('rent [CHF/month]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Both intervals for a flat with 95 m²
print(
    model.get_prediction(pd.DataFrame({'area': [95]}))
    .summary_frame(alpha=0.05)
    .round(0)
)


# %%NBQA-CELL-SEP42d9ce
# Residual plots: assumptions met vs. violated
rng = np.random.default_rng(2)
fitted = rng.uniform(0, 10, 300)
res_ok = rng.normal(0, 1, 300)
res_het = rng.normal(0, 0.2 + 0.3 * fitted)

# Figure showing two residual plots and a normal Q-Q plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(fitted, res_ok, s=8)
axes[0].set_title('Residuals vs. fitted: OK')
axes[1].scatter(fitted, res_het, s=8, color='tab:orange')
axes[1].set_title('Heteroscedastic: violated')
for ax in axes[:2]:
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('fitted values')
    ax.set_ylabel('residuals')
sm.qqplot(res_ok, line='45', fit=True, ax=axes[2])
axes[2].set_title('Normal Q-Q plot: OK')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Prediction of the multiple regression model
print(
    'Prediction for 4.5 rooms and 110 m2: '
    f'{793.843 + 36.477 * 4.5 + 9.795 * 110:.0f} CHF'
)


# %%NBQA-CELL-SEP42d9ce
# Likelihood of 7 heads in 10 coin tosses
p = np.linspace(0, 1, 501)
L = p**7 * (1 - p) ** 3
print(
    'L(0.5) =',
    round(0.5**7 * 0.5**3, 5),
    ' L(0.7) =',
    round(0.7**7 * 0.3**3, 5),
)
# Figure showing the likelihood as a function of p
plt.figure(figsize=(6, 3.5))
plt.plot(p, L)
plt.axvline(
    p[np.argmax(L)],
    color='darkred',
    linestyle='--',
    label=f'maximum at p = {p[np.argmax(L)]:.2f}',
)
plt.xlabel('p (probability of heads)')
plt.ylabel('L(p)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Gradient descent on standardised x (makes the loss surface well-conditioned)
z = (x - x.mean()) / x.std()


def run_gd(eta, steps=50):
    """Gradient descent with learning rate eta; returns b0, b1, MSEs."""
    b0, b1, path = 0.0, 0.0, []
    for _ in range(steps):
        res = y - (b0 + b1 * z)
        path.append(np.mean(res**2))
        b0 += eta * 2 * np.mean(res)
        b1 += eta * 2 * np.mean(res * z)
    return b0, b1, path


# Figure showing the MSE by iteration for different learning rates
plt.figure(figsize=(7, 4))
for eta in [0.01, 0.1, 0.5, 0.99]:
    b0, b1, path = run_gd(eta)
    plt.plot(path, label=f'learning rate {eta}')
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('MSE (log scale)')
plt.title('Gradient descent for simple linear regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Compare the slope from gradient descent with the OLS slope
b0, b1, _ = run_gd(0.1, steps=500)
ols_slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum(
    (x - x.mean()) ** 2
)
print(
    'Slope from gradient descent (back-transformed to m2): '
    f'{b1 / x.std():.2f}   OLS slope: {ols_slope:.2f}'
)


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
