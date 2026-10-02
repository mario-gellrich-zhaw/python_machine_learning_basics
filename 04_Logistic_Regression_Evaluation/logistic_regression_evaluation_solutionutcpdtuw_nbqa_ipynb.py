# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# a) Probabilities
beta_0, beta_1 = -3.5, 2.5
buried = np.array([0, 1, 2])
score = beta_0 + beta_1 * buried
p = 1 / (1 + np.exp(-score))
for n, s, pi in zip(buried, score, p):
    print(
        f'fully buried = {n}: score = {s:5.2f}, p = {pi:.3f}, '
        f'fatal (tau=0.5): {pi >= 0.5}, fatal (tau=0.2): {pi >= 0.2}'
    )

# b) Metrics
TP, FP, FN, TN = 100, 60, 30, 810
n = TP + FP + FN + TN
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print(f'\nAccuracy  = {(TP + TN) / n:.3f}')
print(f'Precision = {precision:.3f}')
print(f'Recall    = {recall:.3f}')
print(f'FPR       = {FP / (FP + TN):.3f}')
print(f'F1        = {2 * precision * recall / (precision + recall):.3f}')

# c) Baseline: always predict 'not fatal'
print(f'\nBaseline accuracy = {(FP + TN) / n:.3f}, baseline recall = 0')


# %%NBQA-CELL-SEP42d9ce
# a) Import the data, keep the years 2000/01 to 2023/24 and print the
#    number of accidents and the share of fatal accidents
aval = pd.read_csv('../00_Data/avalanche_accidents_switzerland.csv', sep=',')
aval = aval[aval['hydrological_year'] >= '2000/01'].reset_index(drop=True)
print('Accidents 2000/01-2023/24:', len(aval))
print('Share of fatal accidents:', round(aval['fatal'].mean(), 3))
print('Missing values (share):')
print(
    aval[
        [
            'danger_level',
            'elevation',
            'aspect',
            'activity',
            'caught',
            'fully_buried',
        ]
    ]
    .isna()
    .mean()
    .round(2)
)

# b) Share of fatal accidents by danger level, activity and number of
#    buried persons
aval['buried_group'] = pd.cut(
    aval['fully_buried'], [-1, 0, 1, 2, 100], labels=['0', '1', '2', '3+']
)
# Figure showing the share of fatal accidents by group
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, col in zip(axes, ['danger_level', 'activity', 'buried_group']):
    rate = aval.groupby(col, observed=True)['fatal'].agg(['mean', 'size'])
    ax.bar(rate.index.astype(str), rate['mean'], color='steelblue')
    for i, (m, s) in enumerate(zip(rate['mean'], rate['size'])):
        ax.text(i, m + 0.01, f'{m:.0%}\n(n={s})', ha='center', fontsize=8)
    ax.set_title(f'Share of fatal accidents by {col}')
    ax.tick_params(axis='x', rotation=20)
axes[0].set_ylabel('Share fatal')
axes[0].set_ylim(0, 1.05)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) Fit the logistic regression fatal ~ fully_buried on all accidents
#    and print beta_0 and beta_1
log_reg_1 = LogisticRegression()
log_reg_1.fit(aval[['fully_buried']], aval['fatal'])
print(
    'beta_0 =',
    round(log_reg_1.intercept_[0], 3),
    ' beta_1 =',
    round(log_reg_1.coef_[0][0], 3),
)

# b) Plot predicted probability and observed share
grid = pd.DataFrame({'fully_buried': np.linspace(0, 8, 200)})
observed = (
    aval[aval['fully_buried'] <= 8]
    .groupby('fully_buried')['fatal']
    .agg(['mean', 'size'])
)

# Figure showing the observed shares and the predicted probabilities
plt.figure(figsize=(7, 4))
plt.scatter(
    observed.index,
    observed['mean'],
    s=np.sqrt(observed['size']) * 8,
    color='grey',
    label='observed share (dot size ~ number of accidents)',
)
plt.plot(
    grid['fully_buried'],
    log_reg_1.predict_proba(grid)[:, 1],
    color='darkred',
    linewidth=2,
    label='logistic regression',
)
plt.axhline(
    0.5, color='black', linestyle=':', linewidth=1, label='threshold 0.5'
)
plt.xlabel('Number of completely buried persons')
plt.ylabel('P(fatal)')
plt.ylim(0, 1.05)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) Complete cases, dummy variables and stratified split
features_a = [
    'danger_level',
    'elevation',
    'aspect',
    'activity',
    'month',
    'caught',
]
features_b = features_a + ['fully_buried']
data = aval.dropna(subset=features_b).reset_index(drop=True)
X = pd.get_dummies(
    data[features_b], columns=['aspect', 'activity'], drop_first=True, dtype=int
)
y = data['fatal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print('Complete cases:', len(data), ' share fatal:', round(y.mean(), 3))
print('Training set:', X_train.shape, ' Test set:', X_test.shape)

# Feature sets: model A without and model B with fully_buried
cols_b = list(X.columns)
cols_a = [c for c in cols_b if c != 'fully_buried']

# b) Logistic regression for both feature sets
model_a = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
).fit(X_train[cols_a], y_train)
model_b = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
).fit(X_train[cols_b], y_train)

# c) Coefficients of model B
coefs = pd.Series(model_b[-1].coef_[0], index=cols_b).sort_values()
# Figure showing the coefficients of model B
plt.figure(figsize=(7, 5))
coefs.plot(kind='barh', color=np.where(coefs > 0, 'darkred', 'steelblue'))
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Coefficient (standardised features)')
plt.title('Model B: coefficients')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) Confusion matrix and metrics
y_pred = model_b.predict(X_test[cols_b])
cm = confusion_matrix(
    y_test, y_pred
)  # rows = actual, columns = predicted (0 = not fatal, 1 = fatal)
TN, FP, FN, TP = cm.ravel()

# Figure showing the confusion matrix as a heatmap
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['not fatal', 'fatal'],
    yticklabels=['not fatal', 'fatal'],
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Model B, tau = 0.5')
plt.show()

# Metrics at the threshold 0.5
print(f'Accuracy  = {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision = {precision_score(y_test, y_pred):.3f}')
print(f'Recall    = {recall_score(y_test, y_pred):.3f}')
print(f'F1        = {f1_score(y_test, y_pred):.3f}')
print(f'FPR       = {FP / (FP + TN):.3f}')

# b) Baseline accuracy (never fatal)
print(f'\nBaseline accuracy (never fatal) = {1 - y_test.mean():.3f}')


# %%NBQA-CELL-SEP42d9ce
# Exercise 6: expected costs for different thresholds
y_prob_b = model_b.predict_proba(X_test[cols_b])[:, 1]
cost_fn, cost_fp = 10, 1

# Confusion matrix and costs for each threshold
rows = []
for tau in np.arange(0.05, 0.91, 0.05):
    y_pred_tau = (y_prob_b >= tau).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_tau).ravel()
    rows.append(
        {
            'tau': round(tau, 2),
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision_score(y_test, y_pred_tau, zero_division=0),
            'recall': TP / (TP + FN),
            'FPR': FP / (FP + TN),
            'costs': cost_fn * FN + cost_fp * FP,
        }
    )
thresholds = pd.DataFrame(rows)
print(thresholds.round(3).to_string(index=False))

# Cost-optimal threshold compared with tau = 0.5
best = thresholds.loc[thresholds['costs'].idxmin()]
print(
    f"\nCost-optimal threshold: tau = {best['tau']}, "
    f"costs = {best['costs']:.0f}"
)
costs_05 = thresholds.loc[np.isclose(thresholds['tau'], 0.5), 'costs']
print(f'Costs at tau = 0.5:      {costs_05.iloc[0]:.0f}')

# Figure showing the total costs by threshold
plt.figure(figsize=(7, 4))
plt.plot(thresholds['tau'], thresholds['costs'], marker='o')
plt.axvline(
    best['tau'],
    color='darkred',
    linestyle='--',
    label=f"minimum at tau = {best['tau']}",
)
plt.xlabel('Threshold tau')
plt.ylabel('Total costs on the test set')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) ROC curves
y_prob_a = model_a.predict_proba(X_test[cols_a])[:, 1]
fpr_a, tpr_a, _ = roc_curve(y_test, y_prob_a)
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)
auc_a, auc_b = roc_auc_score(y_test, y_prob_a), roc_auc_score(y_test, y_prob_b)

# Figure showing the ROC curves of both models
plt.figure(figsize=(6, 6))
plt.plot(
    fpr_b,
    tpr_b,
    linewidth=2,
    label=f'B: with burial information (AUC = {auc_b:.3f})',
)
plt.plot(
    fpr_a,
    tpr_a,
    linewidth=2,
    label=f'A: terrain and bulletin only (AUC = {auc_a:.3f})',
)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='random ranking')
for tau, marker in [(0.5, 'o'), (best['tau'], 's')]:
    row = thresholds.loc[np.isclose(thresholds['tau'], tau)].iloc[0]
    plt.scatter(
        row['FPR'],
        row['recall'],
        s=80,
        marker=marker,
        color='black',
        zorder=3,
        label=(
            f'B, tau = {tau}: TPR = {row["recall"]:.2f}, '
            f'FPR = {row["FPR"]:.2f}'
        ),
    )
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (recall)')
plt.title('ROC curves (test set)')
plt.legend(loc='lower right', fontsize=8)
plt.grid(alpha=0.3)
plt.show()

# b) AUC as probability of a correct ranking
rng = np.random.default_rng(42)
p_pos = y_prob_b[y_test.values == 1]
p_neg = y_prob_b[y_test.values == 0]
a = rng.choice(p_pos, 100_000)
b_ = rng.choice(p_neg, 100_000)
share = np.mean(a > b_) + 0.5 * np.mean(a == b_)
print(f'Share of correctly ranked pairs: {share:.3f}  vs. AUC = {auc_b:.3f}')


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
