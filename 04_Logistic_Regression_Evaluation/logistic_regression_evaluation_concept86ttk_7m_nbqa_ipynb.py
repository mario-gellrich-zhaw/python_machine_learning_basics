# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# Sigmoid function and a simulated logistic regression
def sigmoid(s):
    """Sigmoid function: maps a score to a probability."""
    return 1 / (1 + np.exp(-s))


# Example: beta_0 = -5, beta_1 = 1 and x = 6
print('beta_0 = -5, beta_1 = 1, x = 6 -> p =', round(sigmoid(-5 + 1 * 6), 3))

# Simulate data and fit a logistic regression
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 200)
y = (rng.uniform(size=200) < sigmoid(-5 + 1 * x)).astype(int)
model = LogisticRegression().fit(x.reshape(-1, 1), y)
grid = np.linspace(0, 10, 300).reshape(-1, 1)

# Figure showing the data and the fitted probability curve
plt.figure(figsize=(7, 4))
plt.scatter(x, y + rng.normal(0, 0.02, 200), s=12, alpha=0.6, color='black')
plt.plot(
    grid,
    model.predict_proba(grid)[:, 1],
    linewidth=2,
    label='$\\hat p(x) = \\sigma(\\hat\\beta_0 + \\hat\\beta_1 x)$',
)
plt.axhline(0.5, color='tab:orange', linestyle='--', label='threshold τ = 0.5')
plt.xlabel('x')
plt.ylabel('P(y = 1 | x)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Estimated coefficients
print(
    'Estimated coefficients:',
    round(model.intercept_[0], 2),
    round(model.coef_[0][0], 2),
)


# %%NBQA-CELL-SEP42d9ce
# Log-loss for different predicted probabilities
for p_hat in [0.9, 0.6, 0.5, 0.1, 0.01]:
    print(f'y = 1, p_hat = {p_hat:4}: loss = {-np.log(p_hat):.3f}')


# %%NBQA-CELL-SEP42d9ce
# Metrics of the example confusion matrix
TP, FP, FN, TN = 50, 22, 24, 75
n = TP + FP + FN + TN
precision, recall = TP / (TP + FP), TP / (TP + FN)
print(f'n = {n}')
print(f'accuracy  = {(TP + TN) / n:.3f}')
print(f'precision = {precision:.3f}')
print(f'recall    = {recall:.3f}')
print(f'FPR       = {FP / (FP + TN):.3f}')
print(f'F1        = {2 * precision * recall / (precision + recall):.3f}')

# Why accuracy misleads with imbalanced classes: 1000 cases, 20 positives,
# model predicts always "negative"
print(
    '\nAlways "negative" with 2 % positives: accuracy =',
    980 / 1000,
    ', recall = 0',
)


# %%NBQA-CELL-SEP42d9ce
# Simulated scores: errors and metrics for different thresholds
rng = np.random.default_rng(1)
y_true = np.r_[np.zeros(600), np.ones(400)]
p_hat = np.r_[rng.beta(3, 5, 600), rng.beta(6, 3, 400)]

# Figure showing the score distributions (left) and the metrics by threshold
# (right)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(
    p_hat[y_true == 0], bins=30, alpha=0.7, label='actual negatives (y = 0)'
)
axes[0].hist(
    p_hat[y_true == 1], bins=30, alpha=0.7, label='actual positives (y = 1)'
)
axes[0].axvline(0.5, color='black', linewidth=2, label='threshold τ = 0.5')
axes[0].set_xlabel('predicted score $\\hat p(x)$')
axes[0].set_ylabel('count')
axes[0].set_title('FN: positives left of τ, FP: negatives right of τ')
axes[0].legend()

# Right: recall, FPR and precision for different thresholds
taus = np.linspace(0.05, 0.95, 19)
rec = [np.mean(p_hat[y_true == 1] >= t) for t in taus]
fpr = [np.mean(p_hat[y_true == 0] >= t) for t in taus]
prec = [
    np.sum((p_hat >= t) & (y_true == 1)) / max(np.sum(p_hat >= t), 1)
    for t in taus
]
axes[1].plot(taus, rec, marker='o', label='recall')
axes[1].plot(taus, prec, marker='o', label='precision')
axes[1].plot(taus, fpr, marker='o', label='FPR')
axes[1].set_xlabel('threshold τ')
axes[1].set_title('Metrics as a function of the threshold')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# ROC curve and AUC of the simulated scores
fpr_c, tpr_c, thr = roc_curve(y_true, p_hat)
auc = roc_auc_score(y_true, p_hat)

# Figure showing the ROC curve and some thresholds
plt.figure(figsize=(5.5, 5.5))
plt.plot(fpr_c, tpr_c, linewidth=2, label=f'model (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='random (AUC = 0.5)')
for t in [0.3, 0.5, 0.7]:
    plt.scatter(
        np.mean(p_hat[y_true == 0] >= t),
        np.mean(p_hat[y_true == 1] >= t),
        s=60,
        zorder=3,
    )
    plt.annotate(
        f'τ = {t}',
        (np.mean(p_hat[y_true == 0] >= t), np.mean(p_hat[y_true == 1] >= t)),
        xytext=(8, -12),
        textcoords='offset points',
    )
plt.xlabel('false positive rate (FPR)')
plt.ylabel('true positive rate (TPR)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# AUC as the probability of a correct ranking (all positive-negative pairs)
pos, neg = p_hat[y_true == 1], p_hat[y_true == 0]
pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (
    pos[:, None] == neg[None, :]
).mean()
print(f'AUC = {auc:.4f}, share of correctly ranked pairs = {pairs:.4f}')


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
