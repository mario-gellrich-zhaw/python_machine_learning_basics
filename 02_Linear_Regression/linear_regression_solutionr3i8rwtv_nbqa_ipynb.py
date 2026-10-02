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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Exercise 1: OLS by hand
x = np.array([40, 60, 80, 100, 120])
y = np.array([1200, 1800, 2100, 2700, 3200])

# OLS estimates, fitted values and residuals
beta_1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
beta_0 = y.mean() - beta_1 * x.mean()
y_hat = beta_0 + beta_1 * x
res = y - y_hat
sse = np.sum(res**2)
sst = np.sum((y - y.mean()) ** 2)

# Print the intermediate results
print('x_mean =', x.mean(), ' y_mean =', y.mean())
print('beta_1 =', round(beta_1, 3), ' beta_0 =', round(beta_0, 3))
print('fitted values:', y_hat.round(1))
print('residuals:    ', res.round(1))
print('SSE =', round(sse, 1), ' SST =', round(sst, 1))
print(
    'R2 =', round(1 - sse / sst, 4), ' RMSE =', round(np.sqrt(sse / len(y)), 1)
)
print('Prediction for 95 m2:', round(beta_0 + beta_1 * 95, 1))

# Check with numpy's polyfit
print('np.polyfit:', np.polyfit(x, y, deg=1).round(3))


# %%NBQA-CELL-SEP42d9ce
# a) Import and clean the data
apt = pd.read_csv('../00_Data/apartments_data_enriched_cleaned.csv', sep=';')
apt = apt[
    [
        'bfs_name',
        'rooms',
        'area',
        'price',
        'luxurious',
        'lat',
        'lon',
        'pop',
        'pop_dens',
        'emp',
        'frg_pct',
        'mean_taxable_income',
        'dist_supermarket',
    ]
]
apt = apt.drop_duplicates(
    subset=['bfs_name', 'rooms', 'area', 'price']
).dropna()
apt = apt[(apt['price'] >= 1000) & (apt['price'] <= 5000)].reset_index(
    drop=True
)
print(apt.shape)
print(apt[['rooms', 'area', 'price']].describe().round(1))

# b) Scatter plot price vs. area
plt.figure(figsize=(7, 4))
plt.scatter(apt['area'], apt['price'], s=10, alpha=0.5)
plt.xlabel('Living area [m²]')
plt.ylabel('Rent [CHF/month]')
plt.title('Rental apartments in the canton of Zurich')
plt.grid(alpha=0.3)
plt.show()

# c) Train/test split of the data frame
apt_train, apt_test = train_test_split(apt, test_size=0.20, random_state=42)
print('Training set:', apt_train.shape, ' Test set:', apt_test.shape)


# %%NBQA-CELL-SEP42d9ce
# a) Fit price ~ area on the training set and print the summary
apt_model_1 = smf.ols('price ~ area', data=apt_train).fit()
print(apt_model_1.summary())

# c) Prediction for 95 m2 and test RMSE
flat_95 = pd.DataFrame({'area': [95]})
print(
    'Predicted rent for 95 m2:',
    round(apt_model_1.predict(flat_95).iloc[0], 2),
    'CHF',
)
print(
    'Test RMSE:',
    round(
        root_mean_squared_error(
            apt_test['price'], apt_model_1.predict(apt_test)
        )
    ),
    'CHF',
)


# %%NBQA-CELL-SEP42d9ce
# a) Intervals for 95 m2
pred_95 = apt_model_1.get_prediction(
    pd.DataFrame({'area': [95]})
).summary_frame(alpha=0.05)
print(pred_95.round(0).to_string())

# b) Plot with confidence and prediction intervals
area_grid = pd.DataFrame({'area': np.linspace(20, 200, 200)})
bands = apt_model_1.get_prediction(area_grid).summary_frame(alpha=0.05)

# Figure showing the training data, the regression line and both intervals
plt.figure(figsize=(8, 5))
plt.scatter(
    apt_train['area'],
    apt_train['price'],
    s=8,
    alpha=0.4,
    color='grey',
    label='training data',
)
plt.fill_between(
    area_grid['area'],
    bands['obs_ci_lower'],
    bands['obs_ci_upper'],
    color='tab:orange',
    alpha=0.25,
    label='95% prediction interval',
)
plt.fill_between(
    area_grid['area'],
    bands['mean_ci_lower'],
    bands['mean_ci_upper'],
    color='tab:blue',
    alpha=0.4,
    label='95% confidence interval',
)
plt.plot(
    area_grid['area'],
    bands['mean'],
    color='tab:blue',
    linewidth=2,
    label='OLS line',
)
plt.xlabel('Living area [m²]')
plt.ylabel('Rent [CHF/month]')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) Models with rooms
apt_model_r = smf.ols('price ~ rooms', data=apt_train).fit()
apt_model_ra = smf.ols('price ~ rooms + area', data=apt_train).fit()
print('price ~ rooms:        ', apt_model_r.params.round(1).to_dict())
print(apt_model_ra.summary().tables[1])
print(
    'R2 (rooms + area):',
    round(apt_model_ra.rsquared, 3),
    '  R2 (area only):',
    round(apt_model_1.rsquared, 3),
)

# b) Correlation between rooms and area
print(
    '\nCorrelation rooms - area:',
    round(apt_train[['rooms', 'area']].corr().iloc[0, 1], 2),
)

# c) VIF
X_vif = sm.add_constant(apt_train[['rooms', 'area']])
vif = pd.DataFrame(
    {
        'feature': X_vif.columns,
        'VIF': [
            variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])
        ],
    }
)
print(vif[vif['feature'] != 'const'].round(2).to_string(index=False))


# %%NBQA-CELL-SEP42d9ce
# Exercise 6: residual diagnostics
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# a) Residuals vs. fitted values
axes[0].scatter(apt_model_1.fittedvalues, apt_model_1.resid, s=10, alpha=0.5)
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_xlabel('Fitted values [CHF]')
axes[0].set_ylabel('Residuals [CHF]')
axes[0].set_title('Residuals vs. fitted (price ~ area)')

# b) Q-Q plot
sm.qqplot(apt_model_1.resid, line='45', fit=True, ax=axes[1])
axes[1].set_title('Q-Q plot of residuals')
plt.show()

# Spread of the residuals for small and large fitted values
print(
    apt_model_1.resid.groupby(
        pd.cut(apt_model_1.fittedvalues, [0, 2000, 2500, 3000, 6000])
    )
    .std()
    .round(0)
)


# %%NBQA-CELL-SEP42d9ce
# a) Dummy variable city
for d in [apt_train, apt_test]:
    d['city'] = (d['bfs_name'] == 'Zürich').astype(int)
print(
    'Share of flats in the city of Zurich:', round(apt_train['city'].mean(), 3)
)

# b) Models with city
apt_model_c = smf.ols('price ~ area + city', data=apt_train).fit()
apt_model_ci = smf.ols('price ~ area * city', data=apt_train).fit()
print(apt_model_ci.summary().tables[1])

# c) Price per additional m2 in the city and outside
p = apt_model_ci.params
print('Additional m2 outside the city:', round(p['area'], 2), 'CHF')
print(
    'Additional m2 in the city:     ',
    round(p['area'] + p['area:city'], 2),
    'CHF',
)
print(
    'R2:  area',
    round(apt_model_1.rsquared, 3),
    '| area + city',
    round(apt_model_c.rsquared, 3),
    '| area * city',
    round(apt_model_ci.rsquared, 3),
)

# d) Scatter plot with two lines
area_grid = pd.DataFrame({'area': np.linspace(20, 200, 100)})
# Figure showing the regression lines for the city and the rest of the canton
plt.figure(figsize=(8, 5))
for c, label, color in [
    (0, 'rest of the canton', 'tab:blue'),
    (1, 'city of Zurich', 'tab:red'),
]:
    sub = apt_train[apt_train['city'] == c]
    plt.scatter(
        sub['area'], sub['price'], s=10, alpha=0.4, color=color, label=label
    )
    plt.plot(
        area_grid['area'],
        apt_model_ci.predict(area_grid.assign(city=c)),
        color=color,
        linewidth=2,
    )
plt.xlabel('Living area [m²]')
plt.ylabel('Rent [CHF/month]')
plt.title('price ~ area * city')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Exercise 8: test RMSE and R² of the four models in a table
results = []
for name, m in [
    ('price ~ area', apt_model_1),
    ('price ~ rooms + area', apt_model_ra),
    ('price ~ area + city', apt_model_c),
    ('price ~ area * city', apt_model_ci),
]:
    y_pred = m.predict(apt_test)
    results.append(
        {
            'model': name,
            'R2 (train)': round(m.rsquared, 3),
            'R2 (test)': round(r2_score(apt_test['price'], y_pred), 3),
            'RMSE (test)': round(
                root_mean_squared_error(apt_test['price'], y_pred)
            ),
        }
    )
print(pd.DataFrame(results).to_string(index=False))


# %%NBQA-CELL-SEP42d9ce
# Exercise 9: more covariates
covariates = [
    'area',
    'rooms',
    'luxurious',
    'mean_taxable_income',
    'frg_pct',
    'dist_supermarket',
    'pop_dens',
    'pop',
    'emp',
    'lat',
    'lon',
]

# a) Full model
model_full = smf.ols(
    'price ~ area * city + rooms + luxurious + mean_taxable_income'
    ' + frg_pct + dist_supermarket'
    ' + pop_dens + pop + emp + lat + lon',
    data=apt_train,
).fit()
print(model_full.summary().tables[1])

# b) VIF of the numerical covariates
X_vif = sm.add_constant(apt_train[covariates])
vif = pd.Series(
    [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    index=X_vif.columns,
)
print(vif.drop('const').round(1).sort_values(ascending=False))

# c) Reduced model
model_reduced = smf.ols(
    'price ~ area * city + rooms + luxurious + mean_taxable_income'
    ' + frg_pct + dist_supermarket'
    ' + lat + lon',
    data=apt_train,
).fit()
print(model_reduced.summary().tables[1])

# d) Comparison table
rows = []
for name, m in [
    ('price ~ area', apt_model_1),
    ('price ~ area * city', apt_model_ci),
    ('full model', model_full),
    ('reduced model', model_reduced),
]:
    y_pred = m.predict(apt_test)
    rows.append(
        {
            'model': name,
            'parameters': len(m.params),
            'adj. R2 (train)': round(m.rsquared_adj, 3),
            'R2 (test)': round(r2_score(apt_test['price'], y_pred), 3),
            'RMSE (test)': round(
                root_mean_squared_error(apt_test['price'], y_pred)
            ),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))


# %%NBQA-CELL-SEP42d9ce
# Exercise 10: maximum likelihood for k = 3 successes in n = 10 trials
n, k = 10, 3


def likelihood(p):
    """Likelihood of k successes in n trials."""
    return p**k * (1 - p) ** (n - k)


# Likelihood for three values of p
for p in [0.1, 0.3, 0.5]:
    print(f'L({p}) = {likelihood(p):.5f}')

# Maximum likelihood estimate on a grid of p values
p_grid = np.linspace(0, 1, 501)
p_hat = p_grid[np.argmax(likelihood(p_grid))]

# Figure showing the likelihood and its maximum
plt.figure(figsize=(6, 3.5))
plt.plot(p_grid, likelihood(p_grid))
plt.axvline(
    p_hat, color='darkred', linestyle='--', label=f'maximum at p = {p_hat:.2f}'
)
plt.xlabel('p')
plt.ylabel('L(p)')
plt.title('Likelihood of 3 city flats in 10 flats')
plt.legend()
plt.show()

# Compare with the share of city flats in the full data set
print(
    'Share of city flats in the full data set:',
    round((apt['bfs_name'] == 'Zürich').mean(), 3),
)


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
