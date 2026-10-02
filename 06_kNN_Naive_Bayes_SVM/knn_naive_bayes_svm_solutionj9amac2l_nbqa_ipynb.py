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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import ConfusionMatrixDisplay

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# a) k-NN with and without scaling
new = np.array([30, 40])
X_known = np.array([[58, 41], [29, 70], [33, 75], [62, 38], [31, 8]])
survived = np.array(['no', 'yes', 'yes', 'no', 'no'])
sd = np.array([13, 50])

# Distances with raw and standardised features
d_raw = np.sqrt(((X_known - new) ** 2).sum(axis=1))
d_std = np.sqrt((((X_known - new) / sd) ** 2).sum(axis=1))
print(
    pd.DataFrame(
        {
            'survived': survived,
            'distance (raw)': d_raw.round(1),
            'distance (standardised)': d_std.round(3),
        },
        index=['P1', 'P2', 'P3', 'P4', 'P5'],
    )
)
for name, d in [('raw', d_raw), ('standardised', d_std)]:
    print(f'{name:13s}: 3 nearest = {survived[np.argsort(d)[:3]].tolist()}')

# b) Naive Bayes
score_surv = 342 / 891 * 233 / 342 * 136 / 342
score_died = 549 / 891 * 81 / 549 * 80 / 549
print(f'\nscore(survived) = {score_surv:.4f}, score(died) = {score_died:.4f}')
print(
    'P(survived | woman, 1st class) = '
    f'{score_surv / (score_surv + score_died):.3f}'
)
print(f'Observed survival rate of women in 1st class = {91/94:.3f}')


# %%NBQA-CELL-SEP42d9ce
# a) Import and prepare the data
titanic = pd.read_csv('../00_Data/titanic.csv', sep=',')
titanic['male'] = (titanic['Sex'] == 'male').astype(int)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
features = ['male', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']
X, y = titanic[features], titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print('Training set:', X_train.shape, ' Test set:', X_test.shape)
print(X_train.describe().round(1))

# b) Scatter plot Age vs. Fare
plt.figure(figsize=(8, 4.5))
sns.scatterplot(
    data=titanic,
    x='Age',
    y=titanic['Fare'] + 1,
    hue='Survived',
    s=18,
    alpha=0.7,
)
plt.yscale('log')
plt.ylabel('Fare + 1 [£] (log scale)')
plt.title('Titanic passengers')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Exercise 3: k-NN - scaling and choice of k
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# a) Unscaled vs. scaled
for name, m in [
    ('unscaled', KNeighborsClassifier(n_neighbors=5)),
    (
        'scaled',
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    ),
]:
    acc = cross_val_score(m, X_train, y_train, cv=cv, scoring='accuracy')
    print(
        f'k-NN ({name:8s}): CV accuracy = {acc.mean():.3f} '
        f'(+/- {acc.std():.3f})'
    )

# b) Choice of k
ks = range(1, 41)
cv_acc = [
    cross_val_score(
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
        X_train,
        y_train,
        cv=cv,
    ).mean()
    for k in ks
]
# Figure showing the CV accuracy by k
plt.figure(figsize=(7, 3.5))
plt.plot(ks, cv_acc, marker='o', markersize=4)
plt.xlabel('k (number of neighbours)')
plt.ylabel('5-fold CV accuracy')
plt.grid(alpha=0.3)
plt.show()

# Fit k-NN with the best k and evaluate it on the test set
best_k = ks[int(np.argmax(cv_acc))]
knn = make_pipeline(
    StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)
).fit(X_train, y_train)
print(
    f'Chosen k = {best_k}, CV accuracy = {max(cv_acc):.3f}, '
    f'test accuracy = {knn.score(X_test, y_test):.3f}'
)
ConfusionMatrixDisplay.from_estimator(
    knn,
    X_test,
    y_test,
    display_labels=['died', 'survived'],
    cmap='Blues',
    colorbar=False,
)
plt.title(f'k-NN (k = {best_k}) on the test set')
plt.show()

# c) Two new passengers
new_passengers = pd.DataFrame(
    {
        'male': [1, 0],
        'Age': [30, 30],
        'Pclass': [3, 1],
        'SibSp': [0, 0],
        'Parch': [0, 0],
        'Fare': [8, 80],
    }
)
print(
    pd.DataFrame(
        knn.predict_proba(new_passengers),
        columns=['P(died)', 'P(survived)'],
        index=['man, 3rd class', 'woman, 1st class'],
    ).round(2)
)


# %%NBQA-CELL-SEP42d9ce
# a) CV accuracy and number of support vectors (all features)
rows = []
for kernel in ['linear', 'rbf']:
    for C in [0.01, 1, 100]:
        svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C))
        acc = cross_val_score(svm, X_train, y_train, cv=cv).mean()
        n_sv = len(svm.fit(X_train, y_train)[-1].support_)
        rows.append(
            {
                'kernel': kernel,
                'C': C,
                'support vectors': n_sv,
                'CV accuracy': round(acc, 3),
            }
        )
print(pd.DataFrame(rows).to_string(index=False))

# b) Decision regions with two features
X2_train = pd.DataFrame(
    {'Age': X_train['Age'], 'log_fare': np.log1p(X_train['Fare'])}
)
xx, yy = np.meshgrid(np.linspace(0, 82, 300), np.linspace(0, 6.5, 300))
grid = pd.DataFrame({'Age': xx.ravel(), 'log_fare': yy.ravel()})

# Figure showing the decision regions of the SVM for each kernel and C
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
for row, kernel in enumerate(['linear', 'rbf']):
    for col, C in enumerate([0.01, 1, 100]):
        ax = axes[row, col]
        svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C)).fit(
            X2_train, y_train
        )
        zz = svm.predict(grid).reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[-0.5, 0.5, 1.5],
            colors=['tab:red', 'tab:blue'],
            alpha=0.15,
        )
        for cls, color, label in [
            (0, 'tab:red', 'died'),
            (1, 'tab:blue', 'survived'),
        ]:
            m = y_train.values == cls
            ax.scatter(
                X2_train['Age'][m],
                X2_train['log_fare'][m],
                s=8,
                color=color,
                label=label,
            )
        sv = X2_train.iloc[svm[-1].support_]
        ax.scatter(
            sv['Age'],
            sv['log_fare'],
            s=40,
            facecolors='none',
            edgecolors='black',
            linewidths=0.5,
            label='support vectors',
        )
        ax.set_title(f'kernel = {kernel}, C = {C}: {len(sv)} support vectors')
        ax.set_xlabel('Age [years]' if row == 1 else '')
        ax.set_ylabel('log(1 + fare)' if col == 0 else '')
axes[0, 0].legend(loc='upper right', fontsize=8)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# a) CategoricalNB on male and Pclass (all passengers)
X_cat = pd.DataFrame(
    {'male': titanic['male'], 'Pclass': titanic['Pclass'] - 1}
)  # categories must start at 0
nb_cat = CategoricalNB(alpha=1e-10).fit(X_cat, titanic['Survived'])
woman_1st = pd.DataFrame({'male': [0], 'Pclass': [0]})
print(
    'P(survived | woman, 1st class) =',
    round(nb_cat.predict_proba(woman_1st)[0, 1], 3),
)


# b) CategoricalNB with age group
def categorical_features(X):
    """Categorical features for CategoricalNB (codes start at 0)."""
    return pd.DataFrame(
        {
            'male': X['male'],
            'Pclass': X['Pclass'] - 1,
            'age_group': pd.cut(X['Age'], [-1, 9.99, 59.99, 100], labels=False),
        }
    )


# CV accuracy of CategoricalNB with age group
acc_nb = cross_val_score(
    CategoricalNB(), categorical_features(X_train), y_train, cv=cv
).mean()
print(f'CategoricalNB (male, Pclass, age_group): CV accuracy = {acc_nb:.3f}')


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
