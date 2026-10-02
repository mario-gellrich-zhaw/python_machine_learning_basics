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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# Gini impurity and entropy for two classes
def gini(p):
    """Gini impurity of a node with share p of class 'yes'."""
    return 1 - (p**2 + (1 - p) ** 2)


def entropy(p):
    """Entropy of a node with share p of class 'yes'."""
    if 0 < p < 1:
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return 0.0


# Table of Gini and entropy for different class shares
print(
    pd.DataFrame(
        {
            'yes : no': [f'{k} : {10 - k}' for k in range(0, 11)],
            'Gini': [round(gini(k / 10), 2) for k in range(0, 11)],
            'entropy': [round(entropy(k / 10), 2) for k in range(0, 11)],
        }
    ).to_string(index=False)
)

# Figure showing Gini and entropy as a function of the class share
p = np.linspace(0.001, 0.999, 300)
plt.figure(figsize=(6, 3.5))
plt.plot(p, gini(p), label='Gini')
plt.plot(p, [entropy(v) / 2 for v in p], label='entropy / 2 (scaled)')
plt.xlabel('share of class "yes" in the node')
plt.ylabel('impurity')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Worked example: weighted Gini of two candidate splits
def gini_counts(a, b):
    """Gini impurity of a node with class counts a and b."""
    n = a + b
    return 1 - ((a / n) ** 2 + (b / n) ** 2)


# Weighted Gini and impurity decrease of both splits
g_parent = gini_counts(40, 60)
splits = [
    ('male?', (12, 48), (28, 12)),
    ('1st class?', (15, 10), (25, 50)),
]
for name, left, right in splits:
    n_l, n_r = sum(left), sum(right)
    g_l, g_r = gini_counts(*left), gini_counts(*right)
    g_w = n_l / 100 * g_l + n_r / 100 * g_r
    print(
        f'{name:11s}: G_L = {g_l:.3f}, G_R = {g_r:.3f}, '
        f'weighted G = {g_w:.3f}, Delta G = {g_parent - g_w:.3f}'
    )


# %%NBQA-CELL-SEP42d9ce
# Class probability in a leaf
print('Leaf 145 : 125 -> P(survived) =', round(145 / 270, 3))


# %%NBQA-CELL-SEP42d9ce
# Random forest: average the probabilities of the trees
p_trees = np.array([0.8, 0.6, 0.3])
print(
    'mean probability =',
    round(p_trees.mean(), 2),
    '-> survived' if p_trees.mean() >= 0.5 else '-> not survived',
)


# %%NBQA-CELL-SEP42d9ce
# Grid search for a random forest on simulated data
X, y = make_classification(
    n_samples=800, n_features=8, n_informative=4, random_state=0
)

# Train/test split and grid search with 5-fold CV
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 5, 20]},
    cv=5,
    scoring="roc_auc",
)
grid.fit(X_tr, y_tr)
p = grid.predict_proba(X_te)[:, 1]
print(grid.best_params_)
print(roc_auc_score(y_te, p))
print(confusion_matrix(y_te, p >= 0.5))


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
