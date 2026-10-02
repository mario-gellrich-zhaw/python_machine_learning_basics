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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Exercise 1: residuals, MSE, RMSE and MAE of two models
y = np.array([1200, 2500, 3100, 4000, 4600])
y_hat_a = np.array([1500, 2300, 3300, 3700, 4500])
y_hat_b = np.array([1250, 2550, 3050, 4050, 3800])

# Residuals and loss measures of both models
for name, y_hat in [('A', y_hat_a), ('B', y_hat_b)]:
    res = y - y_hat
    mse = np.mean(res**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(res))
    print(
        f'Model {name}: residuals = {res}, MSE = {mse:,.0f}, '
        f'RMSE = {rmse:.1f}, MAE = {mae:.1f}'
    )


# %%NBQA-CELL-SEP42d9ce
# a) Import the data and print its shape
bikes = pd.read_csv('../00_Data/zurich_bike_counts_2024_2025.csv', sep=',')
print(bikes.shape)

# Keep working days only and add the day of the year
df = bikes[(bikes['weekend'] == 0) & (bikes['holiday'] == 0)].copy()
df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
df = df[
    [
        'date',
        'day_of_year',
        'temp_mean',
        'rain_duration_min',
        'solar_radiation',
        'school_holiday',
        'bikes_total',
    ]
].reset_index(drop=True)
print(df.shape)
print(df.head())

# b) Scatter plot bikes_total vs. day_of_year
plt.figure(figsize=(8, 4))
plt.scatter(df['day_of_year'], df['bikes_total'], s=10, alpha=0.6)
plt.xlabel('Day of the year')
plt.ylabel('Bikes per day (16 stations)')
plt.title('Bikes on working days in Zurich 2024–2025')
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Helper function: polynomial regression pipeline
def make_poly_model(degree):
    """Polynomial regression model of the given degree."""
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


# %%NBQA-CELL-SEP42d9ce
# Exercise 3: feature and target
X = df[['day_of_year']]
y = df['bikes_total']

# a) Train/test split (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print('Training set:', X_train.shape, ' Test set:', X_test.shape)

# b) Fit a polynomial of degree 1 and compute train and test RMSE
model_1 = make_poly_model(1).fit(X_train, y_train)
print(
    'Train RMSE:',
    round(root_mean_squared_error(y_train, model_1.predict(X_train))),
)
print(
    'Test RMSE: ',
    round(root_mean_squared_error(y_test, model_1.predict(X_test))),
)


# %%NBQA-CELL-SEP42d9ce
# a) Sample of 30 training days
X_small = X_train.sample(30, random_state=13)
y_small = y_train.loc[X_small.index]

# b) Train and test RMSE for degree 1..15
degrees = range(1, 16)
rmse_train, rmse_test = [], []
for d in degrees:
    m = make_poly_model(d).fit(X_small, y_small)
    rmse_train.append(root_mean_squared_error(y_small, m.predict(X_small)))
    rmse_test.append(root_mean_squared_error(y_test, m.predict(X_test)))

# Table of the train and test RMSE by degree
print(
    pd.DataFrame(
        {
            'degree': degrees,
            'train RMSE': np.round(rmse_train),
            'test RMSE': np.round(rmse_test),
        }
    ).to_string(index=False)
)

# Figure showing the training and test error by polynomial degree
plt.figure(figsize=(7, 4))
plt.plot(degrees, rmse_train, marker='o', label='training error (30 days)')
plt.plot(degrees, rmse_test, marker='o', label='test error')
plt.yscale('log')
plt.xlabel('Polynomial degree d')
plt.ylabel('RMSE (log scale)')
plt.title('Training vs. test error')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# c) Fitted curves for degree 1, 3 and 12
x_grid = pd.DataFrame(
    {
        'day_of_year': np.linspace(
            X_small['day_of_year'].min(), X_small['day_of_year'].max(), 300
        )
    }
)

# Figure showing the 30 training days and the fitted curves
plt.figure(figsize=(8, 4))
plt.scatter(
    X_small['day_of_year'],
    y_small,
    color='black',
    s=20,
    label='30 training days',
)
for d in [1, 3, 12]:
    m = make_poly_model(d).fit(X_small, y_small)
    plt.plot(
        x_grid['day_of_year'],
        m.predict(x_grid),
        linewidth=2,
        label=f'degree {d}',
    )
plt.ylim(0, 70000)
plt.xlabel('Day of the year')
plt.ylabel('Bikes per day')
plt.title('Polynomial fits on 30 training days')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Exercise 5: 5-fold cross-validation for polynomial degrees
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def cv_rmse(degree, X, y):
    """Mean 5-fold CV RMSE of a polynomial model."""
    scores = cross_val_score(
        make_poly_model(degree),
        X,
        y,
        cv=cv,
        scoring='neg_root_mean_squared_error',
    )
    return -scores.mean()


# CV RMSE by degree for 30 days and for the full training set
degrees_cv = range(1, 11)
cv_small = [cv_rmse(d, X_small, y_small) for d in degrees_cv]
cv_full = [cv_rmse(d, X_train, y_train) for d in degrees_cv]

# Table of the CV RMSE and the selected degrees
print(
    pd.DataFrame(
        {
            'degree': degrees_cv,
            'CV RMSE (30 days)': np.round(cv_small),
            'CV RMSE (full training set)': np.round(cv_full),
        }
    ).to_string(index=False)
)
print(
    'Selected degree (30 days):          ', degrees_cv[int(np.argmin(cv_small))]
)
print(
    'Selected degree (full training set): ', degrees_cv[int(np.argmin(cv_full))]
)

# Figure showing the CV RMSE by degree for both training sets
plt.figure(figsize=(7, 4))
plt.plot(degrees_cv, cv_small, marker='o', label='30 training days')
plt.plot(degrees_cv, cv_full, marker='o', label=f'{len(X_train)} training days')
plt.yscale('log')
plt.xlabel('Polynomial degree d')
plt.ylabel('5-fold CV RMSE (log scale)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# c) Fit the selected model on the full training set,
#    evaluate once on the test set
best_degree = degrees_cv[int(np.argmin(cv_full))]
final_model = make_poly_model(best_degree).fit(X_train, y_train)
print(
    f'Degree {best_degree}: test RMSE =',
    round(root_mean_squared_error(y_test, final_model.predict(X_test))),
)


# %%NBQA-CELL-SEP42d9ce
# Exercise 6: 50 polynomial fits on random samples of 30 days
x_grid = pd.DataFrame({'day_of_year': np.linspace(1, 366, 300)})
x_180 = pd.DataFrame({'day_of_year': [180]})

# Figure with one panel per degree: 50 fits on random samples
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
pred_180 = {1: [], 3: [], 12: []}

# Fit the model on 50 random samples of 30 days per degree
for ax, d in zip(axes, [1, 3, 12]):
    for seed in range(50):
        X_s = X_train.sample(30, random_state=seed)
        y_s = y_train.loc[X_s.index]
        m = make_poly_model(d).fit(X_s, y_s)
        ax.plot(
            x_grid['day_of_year'],
            m.predict(x_grid),
            color='steelblue',
            alpha=0.25,
            linewidth=1,
        )
        pred_180[d].append(m.predict(x_180)[0])
    ax.scatter(X_train['day_of_year'], y_train, color='black', s=4, alpha=0.3)
    ax.set_title(f'degree {d}: 50 fits on 30 random days')
    ax.set_xlabel('Day of the year')
    ax.set_ylim(0, 70000)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Bikes per day')
plt.show()

# b) Mean and standard deviation of the predictions for day 180
print(
    pd.DataFrame(
        {d: [np.mean(p), np.std(p)] for d, p in pred_180.items()},
        index=['mean prediction for day 180', 'std of predictions for day 180'],
    ).round(0)
)


# %%NBQA-CELL-SEP42d9ce
# Exercise 7: date features plus weather and holiday covariates
covariates = [
    'temp_mean',
    'rain_duration_min',
    'solar_radiation',
    'school_holiday',
]


def date_and_weather_features(rows):
    """Polynomial of degree 3 in day_of_year plus weather and holidays."""
    X = df.loc[rows, ['day_of_year'] + covariates].copy()
    X['day_of_year_2'] = X['day_of_year'] ** 2
    X['day_of_year_3'] = X['day_of_year'] ** 3
    return X


# Features of the training and test set; model with covariates
Xc_train, Xc_test = date_and_weather_features(
    X_train.index
), date_and_weather_features(X_test.index)
model_cov = make_pipeline(StandardScaler(), LinearRegression())

# a) and b) 5-fold CV RMSE and test RMSE of both models
rows = []
for name, model, Xtr, Xte in [
    ('degree 3 (date only)', make_poly_model(3), X_train, X_test),
    ('degree 3 + weather + holidays', model_cov, Xc_train, Xc_test),
]:
    cv_score = -cross_val_score(
        model, Xtr, y_train, cv=cv, scoring='neg_root_mean_squared_error'
    ).mean()
    test_score = root_mean_squared_error(
        y_test, model.fit(Xtr, y_train).predict(Xte)
    )
    rows.append(
        {
            'model': name,
            'CV RMSE': round(cv_score),
            'test RMSE': round(test_score),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))

# Coefficients of the model with covariates (standardised features)
print(pd.Series(model_cov[-1].coef_, index=Xc_train.columns).round(0))


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
