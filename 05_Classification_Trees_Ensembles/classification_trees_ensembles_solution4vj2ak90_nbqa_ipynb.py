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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Exercise 1: Gini impurity by hand
def gini(n_survived, n_died):
    """Gini impurity of a node with two classes."""
    n = n_survived + n_died
    return 1 - ((n_survived / n) ** 2 + (n_died / n) ** 2)


# a) Gini impurity
g_parent = gini(76, 124)
print(f'Parent: G = {g_parent:.4f}')
splits = [
    ('male?', (25, 105), (51, 19)),
    ('1st class?', (30, 20), (46, 104)),
]
for name, left, right in splits:
    n_l, n_r = sum(left), sum(right)
    g_l, g_r = gini(*left), gini(*right)
    g_w = n_l / (n_l + n_r) * g_l + n_r / (n_l + n_r) * g_r
    print(
        f'{name:11s} G_L = {g_l:.4f}, G_R = {g_r:.4f}, '
        f'weighted G = {g_w:.4f}, Delta G = {g_parent - g_w:.4f}'
    )

# b) Leaf probability
print(f'\nLeaf 145 : 125 -> P(survived) = {145 / 270:.3f}')

# c) Random forest: mean probability vs. majority vote
p_trees = np.array([0.9, 0.7, 0.4, 0.6, 0.3])
print(
    f'Mean probability = {p_trees.mean():.2f} -> '
    f'predict survived: {p_trees.mean() >= 0.5}'
)
print(f'Majority vote: {np.sum(p_trees >= 0.5)} of 5 trees vote survived')


# %%NBQA-CELL-SEP42d9ce
# a) Import the data; survivors and missing values
titanic = pd.read_csv('../00_Data/titanic.csv', sep=',')
print(titanic.shape)
print(
    'Survived:',
    titanic['Survived'].sum(),
    f"({titanic['Survived'].mean():.1%})",
)
print(
    'Missing values:', titanic.isna().sum()[titanic.isna().sum() > 0].to_dict()
)

# b) Survival rates
print(titanic.groupby('Sex')['Survived'].mean().round(3))
print(titanic.groupby('Pclass')['Survived'].mean().round(3))
print(
    titanic.pivot_table(
        index='Sex', columns='Pclass', values='Survived', aggfunc='mean'
    ).round(3)
)

# Survival rate by age group (10 years) and sex
titanic['age_group'] = pd.cut(
    titanic['Age'],
    bins=range(0, 90, 10),
    right=False,
    labels=[f'{a}-{a + 9}' for a in range(0, 80, 10)],
)
rate = titanic.pivot_table(
    index='age_group',
    columns='Sex',
    values='Survived',
    aggfunc='mean',
    observed=True,
)
rate.plot(kind='bar', figsize=(8, 3.5), color=['tab:red', 'tab:blue'])
plt.ylabel('Survival rate')
plt.xlabel('Age group [years]')
plt.title('Survival rate by age group and sex')
plt.xticks(rotation=0)
plt.show()

# c) Features X, target y and stratified train/test split
df = pd.DataFrame(
    {
        'male': (titanic['Sex'] == 'male').astype(int),
        'Age': titanic['Age'].fillna(titanic['Age'].median()),
        'Pclass': titanic['Pclass'],
        'SibSp': titanic['SibSp'],
        'Parch': titanic['Parch'],
        'Fare': titanic['Fare'],
        'Embarked': titanic['Embarked'].fillna('S'),
    }
)
X = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print('Training set:', X_train.shape, ' Test set:', X_test.shape)


# %%NBQA-CELL-SEP42d9ce
# a) Fit and plot the tree
X_example = pd.DataFrame(
    {
        'male': X['male'],
        'Age': X['Age'],
        'third_class': (X['Pclass'] == 3).astype(int),
        'Fare': X['Fare'],
    }
)
example_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(
    X_example, y
)
# Figure showing the fitted tree
plt.figure(figsize=(20, 7))
tree.plot_tree(
    example_tree,
    feature_names=list(X_example.columns),
    class_names=['died', 'survived'],
    filled=True,
    rounded=True,
    fontsize=9,
    precision=2,
)
plt.show()
print(
    tree.export_text(
        example_tree, feature_names=list(X_example.columns), show_weights=True
    )
)

# b) Depth, size and number of leaves
print(
    'Depth:',
    example_tree.get_depth(),
    ' size (nodes):',
    example_tree.tree_.node_count,
    ' leaves:',
    example_tree.get_n_leaves(),
)

# c) Class counts of the root and its children
t = example_tree.tree_
for node, name in [
    (0, 'root'),
    (t.children_left[0], 'left child'),
    (t.children_right[0], 'right child'),
]:
    counts = np.round(t.value[node][0] * t.n_node_samples[node]).astype(int)
    print(
        f'{name:12s}: n = {t.n_node_samples[node]}, died = {counts[0]}, '
        f'survived = {counts[1]}, Gini = {t.impurity[node]:.3f}'
    )
n0, nl, nr = (
    t.n_node_samples[0],
    t.n_node_samples[t.children_left[0]],
    t.n_node_samples[t.children_right[0]],
)
delta = t.impurity[0] - (
    nl / n0 * t.impurity[t.children_left[0]]
    + nr / n0 * t.impurity[t.children_right[0]]
)
print(f'Delta G of the root split = {delta:.3f}')

# d) Predictions for two passengers
new_passengers = pd.DataFrame(
    {'male': [0, 1], 'Age': [30, 5], 'third_class': [1, 0], 'Fare': [10, 30]}
)
print(
    pd.DataFrame(
        example_tree.predict_proba(new_passengers),
        columns=['P(died)', 'P(survived)'],
    ).round(3)
)
print('Leaf ids:', example_tree.apply(new_passengers))


# %%NBQA-CELL-SEP42d9ce
# Exercise 4: overfitting and pruning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# a) Depth-3 tree vs. full tree
tree_3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(
    X_train, y_train
)
full_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
for name, m in [('depth 3', tree_3), ('full tree', full_tree)]:
    print(
        f'{name:10s}: leaves = {m.get_n_leaves():3d}, '
        f'train accuracy = {m.score(X_train, y_train):.3f}, '
        f'test accuracy = {m.score(X_test, y_test):.3f}, '
        f'test AUC = {roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f}'
    )

# b) Pruning with GridSearchCV
alphas = full_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
grid_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {'ccp_alpha': alphas},
    cv=cv,
    scoring='roc_auc',
)
grid_tree.fit(X_train, y_train)
pruned_tree = grid_tree.best_estimator_
print(
    f"\nBest ccp_alpha = {grid_tree.best_params_['ccp_alpha']:.4f}, "
    f"CV AUC = {grid_tree.best_score_:.3f}"
)
print(
    f'Pruned tree: leaves = {pruned_tree.get_n_leaves()}, '
    'test AUC = '
    f'{roc_auc_score(y_test, pruned_tree.predict_proba(X_test)[:, 1]):.3f}'
)


# %%NBQA-CELL-SEP42d9ce
# a) Random forest with OOB score
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print(f'OOB accuracy  = {rf.oob_score_:.3f}')
print(f'Test accuracy = {rf.score(X_test, y_test):.3f}')

# b) Probabilities of single trees for one passenger
passenger = X_test.iloc[[0]]
print(passenger.to_string())
p_trees = np.array(
    [est.predict_proba(passenger.values)[0, 1] for est in rf.estimators_]
)
print('First 10 trees:', p_trees[:10].round(2))
print(f'Mean over all 500 trees = {p_trees.mean():.3f}')
print(f'rf.predict_proba        = {rf.predict_proba(passenger)[0, 1]:.3f}')
print(f'Share of trees voting "survived" = {(p_trees >= 0.5).mean():.3f}')


# %%NBQA-CELL-SEP42d9ce
# a) Random forest as in the concept notebook
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 5, 20]},
    cv=5,
    scoring='roc_auc',
)
grid.fit(X_train, y_train)
p = grid.predict_proba(X_test)[:, 1]
print(
    'Best parameters:',
    grid.best_params_,
    ' CV AUC:',
    round(grid.best_score_, 3),
)
print('Test AUC:', round(roc_auc_score(y_test, p), 3))
print(confusion_matrix(y_test, p >= 0.5))

# b) Gradient boosting
grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 300],
        'max_depth': [2, 3],
    },
    cv=5,
    scoring='roc_auc',
)
grid_gb.fit(X_train, y_train)
p_gb = grid_gb.predict_proba(X_test)[:, 1]
print(
    '\nBest parameters:',
    grid_gb.best_params_,
    ' CV AUC:',
    round(grid_gb.best_score_, 3),
)
print('Test AUC:', round(roc_auc_score(y_test, p_gb), 3))
print(confusion_matrix(y_test, p_gb >= 0.5))


# %%NBQA-CELL-SEP42d9ce
# Exercise 7: compare the models on the test set
models = {
    'Tree (depth 3)': tree_3,
    'Pruned tree': pruned_tree,
    'Random forest': grid.best_estimator_,
    'Gradient boosting': grid_gb.best_estimator_,
}

# a) ROC curves and AUC
plt.figure(figsize=(6.5, 6.5))
for name, m in models.items():
    p = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(
        fpr,
        tpr,
        linewidth=1.8,
        label=f'{name} (AUC = {roc_auc_score(y_test, p):.3f})',
    )
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate (recall)')
plt.title('ROC curves on the test set')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.show()

# b) Permutation importance
perm = permutation_importance(
    grid.best_estimator_,
    X_test,
    y_test,
    scoring='roc_auc',
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
# Figure showing the permutation importance
plt.figure(figsize=(7, 4))
imp.plot(kind='barh', color='darkred')
plt.xlabel('Decrease in test AUC when the feature is shuffled')
plt.title('Permutation importance (random forest)')
plt.show()
print(imp.sort_values(ascending=False).round(3))


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
