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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# a) Best split
x = np.array([-2, 3, 8, 14, 20, 26])
y = np.array([15, 22, 30, 38, 42, 44])


def sse(v):
    """Sum of squared errors around the mean."""
    return np.sum((v - v.mean()) ** 2)


# SSE of the root and of all possible splits
print(f'Root: mean = {y.mean():.2f}, SSE = {sse(y):.2f}')
rows = []
for s in (x[:-1] + x[1:]) / 2:
    left, right = y[x <= s], y[x > s]
    rows.append(
        {
            'split': f'x <= {s}',
            'mean left': left.mean(),
            'mean right': right.mean(),
            'SSE left': sse(left),
            'SSE right': sse(right),
            'SSE total': sse(left) + sse(right),
        }
    )
print(pd.DataFrame(rows).round(2).to_string(index=False))

# Check with scikit-learn: a tree with one split (a "stump")
stump = DecisionTreeRegressor(max_depth=1).fit(x.reshape(-1, 1), y)
print(tree.export_text(stump, feature_names=['temperature']))

# b) Variance of an average
sigma2, B = 4, 100
for rho in [0.6, 0.2]:
    print(
        f'rho = {rho}: Var = {rho * sigma2 + (1 - rho) / B * sigma2:.3f}  '
        f'(limit for B -> inf: {rho * sigma2})'
    )

# c) Boosting steps
y_obs, pred, nu, steps = 40, 30, 0.1, 0
while y_obs - pred >= 5:
    pred = pred + nu * (y_obs - pred)
    steps += 1
    if steps == 1:
        print('Prediction after 1 step:', pred)
print(f'Steps until residual < 5: {steps} (prediction {pred:.2f})')


# %%NBQA-CELL-SEP42d9ce
# a) Import the data
bikes = pd.read_csv('../00_Data/zurich_bike_counts_2024_2025.csv', sep=',')

# Features (weekday as dummy variables) and target
features = [
    'temp_mean',
    'temp_max',
    'rain_duration_min',
    'solar_radiation',
    'month',
    'holiday',
    'school_holiday',
    'weekday',
]
X = pd.get_dummies(bikes[features], columns=['weekday'], dtype=int)
y = bikes['mythenquai']
print(X.shape)
print(y.describe().round(0))

# b) Scatter plot by working day / weekend
plt.figure(figsize=(8, 4.5))
for w, label, color in [
    (0, 'working day', 'tab:blue'),
    (1, 'weekend', 'tab:red'),
]:
    m = bikes['weekend'] == w
    plt.scatter(
        bikes.loc[m, 'temp_max'],
        y[m],
        s=12,
        alpha=0.6,
        color=color,
        label=label,
    )
plt.xlabel('Maximum temperature [°C]')
plt.ylabel('Bikes per day at the Mythenquai')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# c) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print('Training set:', X_train.shape, ' Test set:', X_test.shape)


# %%NBQA-CELL-SEP42d9ce
# Exercise 3: linear regression with all features; test RMSE and R²
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print(
    'Linear regression - test RMSE:',
    round(root_mean_squared_error(y_test, y_pred_lin)),
    ' test R2:',
    round(r2_score(y_test, y_pred_lin), 3),
)


# %%NBQA-CELL-SEP42d9ce
# a) Fit the tree
reg_tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(
    X_train, y_train
)

# b) Plot the tree
plt.figure(figsize=(20, 7))
tree.plot_tree(
    reg_tree,
    feature_names=list(X_train.columns),
    filled=True,
    rounded=True,
    fontsize=9,
    precision=1,
)
plt.show()

# c) Test RMSE and R2
y_pred_tree = reg_tree.predict(X_test)
print(
    'Tree (depth 3) - test RMSE:',
    round(root_mean_squared_error(y_test, y_pred_tree)),
    ' test R2:',
    round(r2_score(y_test, y_pred_tree), 3),
)


# %%NBQA-CELL-SEP42d9ce
# Exercise 5: tree complexity and pruning
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# a) Training vs. CV error for different depths
depths = range(1, 16)
rmse_train, rmse_cv = [], []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    rmse_cv.append(
        -cross_val_score(
            m, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error'
        ).mean()
    )
    rmse_train.append(
        root_mean_squared_error(
            y_train, m.fit(X_train, y_train).predict(X_train)
        )
    )

# Figure showing the training and CV error by tree depth
plt.figure(figsize=(7, 4))
plt.plot(depths, rmse_train, marker='o', label='training error')
plt.plot(depths, rmse_cv, marker='o', label='5-fold CV error')
plt.xlabel('max_depth')
plt.ylabel('RMSE [bikes per day]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print(
    'Best depth by CV:',
    depths[int(np.argmin(rmse_cv))],
    ' CV RMSE:',
    round(min(rmse_cv)),
)

# b) Cost-complexity pruning
alphas = (
    DecisionTreeRegressor(random_state=42)
    .cost_complexity_pruning_path(X_train, y_train)
    .ccp_alphas
)
grid_tree = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    {'ccp_alpha': alphas},
    cv=cv,
    scoring='neg_root_mean_squared_error',
)
grid_tree.fit(X_train, y_train)
pruned_tree = grid_tree.best_estimator_
print('Number of candidate alphas:', len(alphas))
print(
    'Best ccp_alpha:',
    round(grid_tree.best_params_['ccp_alpha'], 1),
    ' CV RMSE:',
    round(-grid_tree.best_score_),
)
print(
    'Leaves: full tree =',
    DecisionTreeRegressor(random_state=42).fit(X_train, y_train).get_n_leaves(),
    ' pruned tree =',
    pruned_tree.get_n_leaves(),
)
print(
    'Pruned tree - test RMSE:',
    round(root_mean_squared_error(y_test, pruned_tree.predict(X_test))),
)


# %%NBQA-CELL-SEP42d9ce
# a) Number of trees vs. OOB error
n_trees = [1, 5, 10, 25, 50, 100, 200, 400]
oob_rmse = []
for n in n_trees:
    rf = RandomForestRegressor(
        n_estimators=n, oob_score=True, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_oob = rf.oob_prediction_
    ok = ~np.isnan(
        pred_oob
    )  # with very few trees, some days are never out-of-bag
    oob_rmse.append(root_mean_squared_error(y_train[ok], pred_oob[ok]))

# Figure showing the OOB RMSE by number of trees
plt.figure(figsize=(7, 4))
plt.plot(n_trees, oob_rmse, marker='o')
plt.xscale('log')
plt.xlabel('n_estimators (log scale)')
plt.ylabel('OOB RMSE [bikes per day]')
plt.title('Random forest: OOB error vs. number of trees')
plt.grid(alpha=0.3)
plt.show()
print(
    pd.DataFrame(
        {'n_estimators': n_trees, 'OOB RMSE': np.round(oob_rmse)}
    ).to_string(index=False)
)

# b) max_features
rf_cv = {}
for mf in [1.0, 'sqrt', 0.5]:
    rf = RandomForestRegressor(
        n_estimators=300, max_features=mf, random_state=42, n_jobs=-1
    )
    rf_cv[mf] = -cross_val_score(
        rf, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error'
    ).mean()
    print(f'max_features = {mf}: CV RMSE = {rf_cv[mf]:.0f}')


# %%NBQA-CELL-SEP42d9ce
# c) Test RMSE and feature importance (use the best max_features from b)
best_mf = min(rf_cv, key=rf_cv.get)
rf_best = RandomForestRegressor(
    n_estimators=300, max_features=best_mf, random_state=42, n_jobs=-1
)
rf_best.fit(X_train, y_train)
print(
    f'Random forest (max_features = {best_mf}) - test RMSE:',
    round(root_mean_squared_error(y_test, rf_best.predict(X_test))),
    ' test R2:',
    round(r2_score(y_test, rf_best.predict(X_test)), 3),
)

# Permutation importance on the test set vs. impurity-based importance
perm = permutation_importance(
    rf_best, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
imp = pd.DataFrame(
    {
        'impurity-based (MDI)': rf_best.feature_importances_,
        'permutation (test set)': perm.importances_mean,
    },
    index=X_train.columns,
).sort_values('permutation (test set)')
print(imp.round(3))

# Figure comparing both importance measures
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
imp['impurity-based (MDI)'].plot(
    kind='barh', ax=axes[0], color='grey', title='Impurity-based importance'
)
imp['permutation (test set)'].plot(
    kind='barh',
    ax=axes[1],
    color='darkred',
    title='Permutation importance (decrease in R²)',
)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) Test error per iteration for different learning rates
plt.figure(figsize=(8, 4))
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(
        n_estimators=1000, max_depth=3, learning_rate=lr, random_state=42
    )
    gb.fit(X_train, y_train)
    rmse_staged = [
        root_mean_squared_error(y_test, p) for p in gb.staged_predict(X_test)
    ]
    plt.plot(range(1, 1001), rmse_staged, label=f'learning rate {lr}')
    print(
        f'learning rate {lr}: min test RMSE = {min(rmse_staged):.0f} '
        f'after {int(np.argmin(rmse_staged)) + 1} trees,'
        f' after 1000 trees = {rmse_staged[-1]:.0f}'
    )
plt.ylim(0, 1000)
plt.xlabel('Number of trees M')
plt.ylabel('Test RMSE [bikes per day]')
plt.title('Gradient boosting: test error vs. number of trees')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Exercise 8: compare all models with the same cross-validation
models = {
    'Linear regression': LinearRegression(),
    'Pruned tree': DecisionTreeRegressor(
        ccp_alpha=grid_tree.best_params_['ccp_alpha'], random_state=42
    ),
    'Random forest': RandomForestRegressor(
        n_estimators=300, max_features=best_mf, random_state=42, n_jobs=-1
    ),
    'Gradient boosting': GradientBoostingRegressor(
        learning_rate=0.05, n_estimators=300, max_depth=3, random_state=42
    ),
}

# 5-fold CV RMSE of all models
rows = []
for name, m in models.items():
    scores = -cross_val_score(
        m, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error'
    )
    rows.append(
        {
            'model': name,
            'CV RMSE (mean)': scores.mean(),
            'CV RMSE (std)': scores.std(),
        }
    )
comparison = pd.DataFrame(rows).sort_values('CV RMSE (mean)')
print(comparison.round(0).to_string(index=False))

# Fit the best model on the training set, evaluate on the test set
best_name = comparison.iloc[0]['model']
best_model = models[best_name].fit(X_train, y_train)
print(f'\nBest model by CV: {best_name}')
print(
    'Test RMSE:',
    round(root_mean_squared_error(y_test, best_model.predict(X_test))),
    ' test R2:',
    round(r2_score(y_test, best_model.predict(X_test)), 3),
)


# %%NBQA-CELL-SEP42d9ce
# a) Extended feature set
bikes['day_of_year'] = pd.to_datetime(bikes['date']).dt.dayofyear
features_more = features + ['pressure', 'day_of_year', 'year']
X_more = pd.get_dummies(bikes[features_more], columns=['weekday'], dtype=int)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_more, y, test_size=0.20, random_state=42
)
print(
    X_more.shape,
    ' same test days as before:',
    (X2_test.index == X_test.index).all(),
)

# b) CV RMSE with the extended feature set
rows = []
for name, m in models.items():
    if name == 'Pruned tree':
        continue
    scores = -cross_val_score(
        m, X2_train, y2_train, cv=cv, scoring='neg_root_mean_squared_error'
    )
    before = comparison.loc[comparison['model'] == name, 'CV RMSE (mean)'].iloc[
        0
    ]
    rows.append(
        {
            'model': name,
            'CV RMSE (Exercise 8)': round(before),
            'CV RMSE (more covariates)': round(scores.mean()),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))

# c) Test RMSE and permutation importance of the best model
gb_more = GradientBoostingRegressor(
    learning_rate=0.05, n_estimators=300, max_depth=3, random_state=42
)
gb_more.fit(X2_train, y2_train)
print(
    'Gradient boosting (more covariates) - test RMSE:',
    round(root_mean_squared_error(y2_test, gb_more.predict(X2_test))),
    ' test R2:',
    round(r2_score(y2_test, gb_more.predict(X2_test)), 3),
)
perm = permutation_importance(
    gb_more, X2_test, y2_test, n_repeats=10, random_state=42, n_jobs=-1
)
imp = pd.Series(perm.importances_mean, index=X_more.columns).sort_values()
imp.tail(10).plot(
    kind='barh',
    color='darkred',
    figsize=(7, 4),
    title='Permutation importance (test set)',
)
plt.show()
print(imp.sort_values(ascending=False).head(8).round(3))


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
