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
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# k-NN decision boundaries for k = 1 and k = 15
X, y = make_moons(n_samples=250, noise=0.35, random_state=0)
xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))

# Figure showing the decision regions for k = 1 and k = 15
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, k, title in [
    (axes[0], 1, 'k = 1 (high variance)'),
    (axes[1], 15, 'k = 15 (smoother)'),
]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    zz = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(
        xx,
        yy,
        zz,
        levels=[-0.5, 0.5, 1.5],
        colors=['tab:orange', 'tab:blue'],
        alpha=0.2,
    )
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, color='tab:orange')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color='tab:blue')
    ax.set_title(title)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Distance with and without standardisation
diff = np.array([30, 300])  # difference in age [years] and income [CHF]
sd = np.array([15, 2000])
print(
    'distance unscaled   :',
    round(np.sqrt(np.sum(diff**2)), 1),
    ' (dominated by income)',
)
print(
    'distance standardised:',
    round(np.sqrt(np.sum((diff / sd) ** 2)), 2),
    ' (dominated by age)',
)


# %%NBQA-CELL-SEP42d9ce
# Naive Bayes by hand: play golf?
golf = pd.DataFrame(
    {
        'outlook': [
            'sunny',
            'sunny',
            'overcast',
            'rainy',
            'rainy',
            'rainy',
            'overcast',
            'sunny',
            'sunny',
            'rainy',
            'sunny',
            'overcast',
            'overcast',
            'rainy',
        ],
        'temp': [
            'hot',
            'hot',
            'hot',
            'mild',
            'cool',
            'cool',
            'cool',
            'mild',
            'cool',
            'mild',
            'mild',
            'mild',
            'hot',
            'mild',
        ],
        'humidity': [
            'high',
            'high',
            'high',
            'high',
            'normal',
            'normal',
            'normal',
            'high',
            'normal',
            'normal',
            'normal',
            'high',
            'normal',
            'high',
        ],
        'windy': [
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            True,
        ],
        'play': [
            'no',
            'no',
            'yes',
            'yes',
            'yes',
            'no',
            'yes',
            'no',
            'yes',
            'yes',
            'yes',
            'yes',
            'yes',
            'no',
        ],
    }
)
print(golf.to_string())

# Naive Bayes scores for a new day
new_day = {
    'outlook': 'sunny',
    'temp': 'cool',
    'humidity': 'high',
    'windy': True,
}
scores = {}
for c in ['yes', 'no']:
    sub = golf[golf['play'] == c]
    score = len(sub) / len(golf)  # base rate P(y)
    factors = [f'{len(sub)}/{len(golf)}']
    for feature, value in new_day.items():
        k = (sub[feature] == value).sum()
        score *= k / len(sub)  # P(x_j | y)
        factors.append(f'{k}/{len(sub)}')
    scores[c] = score
    print(f'score({c}) = ' + ' * '.join(factors) + f' = {score:.4f}')
print(f'P(no | x) = {scores["no"] / (scores["no"] + scores["yes"]):.3f}')


# %%NBQA-CELL-SEP42d9ce
# Naive Bayes for text: spam vs. normal messages
docs = [
    'free prize call now',
    'win a free prize',
    'call me later',
    'lunch tomorrow',
    'are you free for lunch',
    'claim your free cash prize',
]
labels = [1, 1, 0, 0, 0, 1]  # 1 = spam, 0 = normal message

# Bag of words: word counts per document
vec = CountVectorizer()
W = vec.fit_transform(docs)
print(
    pd.DataFrame(
        W.toarray(), columns=vec.get_feature_names_out(), index=docs
    ).to_string()
)

# Fit Naive Bayes; words typical for spam vs. normal messages
nb = MultinomialNB().fit(W, labels)
log_ratio = pd.Series(
    nb.feature_log_prob_[1] - nb.feature_log_prob_[0],
    index=vec.get_feature_names_out(),
)
print('\nlog ratio (spam vs. normal):')
print(log_ratio.sort_values(ascending=False).round(2).to_string())
print(
    '\nP(spam | "free lunch") =',
    round(nb.predict_proba(vec.transform(['free lunch']))[0, 1], 3),
)


# %%NBQA-CELL-SEP42d9ce
# SVM with a linear and an RBF kernel on simulated data
X_lin, y_lin = make_blobs(
    n_samples=60, centers=[[-2, -1], [2, 1.5]], cluster_std=0.9, random_state=3
)
X_circ, y_circ = make_circles(
    n_samples=200, noise=0.08, factor=0.4, random_state=0
)

# Figure showing the decision boundaries, margins and support vectors
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (Xs, ys, kernel, title) in zip(
    axes,
    [
        (X_lin, y_lin, 'linear', 'Linear SVM: maximal margin'),
        (X_circ, y_circ, 'rbf', 'RBF kernel: non-linear boundary'),
    ],
):
    svm = SVC(kernel=kernel, C=1).fit(Xs, ys)
    gx, gy = np.meshgrid(
        np.linspace(Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5, 300),
        np.linspace(Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5, 300),
    )
    zz = svm.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(
        gx,
        gy,
        zz > 0,
        levels=[-0.5, 0.5, 1.5],
        colors=['tab:orange', 'tab:blue'],
        alpha=0.15,
    )
    ax.contour(
        gx,
        gy,
        zz,
        levels=[-1, 0, 1],
        colors='black',
        linestyles=['--', '-', '--'],
        linewidths=[1, 2, 1],
    )
    ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=15, color='tab:orange')
    ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=15, color='tab:blue')
    sv = svm.support_vectors_
    ax.scatter(
        sv[:, 0],
        sv[:, 1],
        s=90,
        facecolors='none',
        edgecolors='darkred',
        linewidths=1.2,
    )
    ax.set_title(f'{title} ({len(sv)} support vectors)')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Effect of C on the margin and the number of support vectors
# (linear kernel, overlapping classes)
X_ov, y_ov = make_blobs(
    n_samples=100,
    centers=[[-1, -0.5], [1, 0.5]],
    cluster_std=1.0,
    random_state=1,
)
# Figure showing the margin for C = 0.01, 1 and 100
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, C in zip(axes, [0.01, 1, 100]):
    svm = SVC(kernel='linear', C=C).fit(X_ov, y_ov)
    gx, gy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-3.5, 3.5, 200))
    zz = svm.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contour(
        gx,
        gy,
        zz,
        levels=[-1, 0, 1],
        colors='black',
        linestyles=['--', '-', '--'],
    )
    ax.scatter(X_ov[y_ov == 0, 0], X_ov[y_ov == 0, 1], s=12, color='tab:orange')
    ax.scatter(X_ov[y_ov == 1, 0], X_ov[y_ov == 1, 1], s=12, color='tab:blue')
    ax.scatter(
        *svm.support_vectors_.T, s=70, facecolors='none', edgecolors='darkred'
    )
    ax.set_title(f'C = {C}: {len(svm.support_)} support vectors')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
