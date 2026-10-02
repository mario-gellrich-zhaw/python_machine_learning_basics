# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Ignore warnings
warnings.filterwarnings('ignore')


# %%NBQA-CELL-SEP42d9ce
# k-means on three simulated clusters
X, _ = make_blobs(
    n_samples=150,
    centers=[[-2, -1.5], [2.5, -1], [0, 2.5]],
    cluster_std=0.8,
    random_state=4,
)
centres = np.array(
    [[-1.5, -2.5], [-0.5, -0.5], [2.5, 2.5]]
)  # deliberately bad start

# Figure showing the first k-means iterations
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, it, title in [
    (axes[0], 0, 'Start: random centres'),
    (axes[1], 1, 'After 1 iteration'),
    (axes[2], 20, 'Converged'),
]:
    c = centres.copy()
    for _ in range(it):
        labels = np.argmin(
            ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        c = np.array([X[labels == j].mean(axis=0) for j in range(3)])
    labels = np.argmin(
        ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    colors = np.array(['tab:blue', 'tab:orange', 'tab:green'])
    ax.scatter(
        X[:, 0], X[:, 1], s=12, color='grey' if it == 0 else colors[labels]
    )
    ax.scatter(
        c[:, 0], c[:, 1], s=250, marker='X', c=colors, edgecolors='black'
    )
    wcss = ((X - c[labels]) ** 2).sum()
    ax.set_title(f'{title} (WCSS = {wcss:.0f})')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Elbow method and silhouette score for k = 1..8
ks = range(1, 9)
wcss = [
    KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in ks
]
sil = [
    silhouette_score(
        X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    )
    for k in ks[1:]
]

# Figure showing the elbow plot and the silhouette score by k
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks, wcss, marker='o')
axes[0].scatter(
    [3], [wcss[2]], s=200, color='tab:orange', zorder=3, label='elbow: k = 3'
)
axes[0].set_xlabel('number of clusters k')
axes[0].set_ylabel('within-cluster sum of squares')
axes[0].set_title('Elbow plot')
axes[0].legend()
axes[1].plot(list(ks)[1:], sil, marker='o', color='darkred')
axes[1].set_xlabel('number of clusters k')
axes[1].set_ylabel('mean silhouette score')
axes[1].set_title('Silhouette score')
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Silhouette plots for k = 3 and k = 5
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
for ax, k in zip(axes, [3, 5]):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    s = silhouette_samples(X, labels)
    y = 0
    for c in range(k):
        vals = np.sort(s[labels == c])
        ax.fill_betweenx(
            np.arange(y, y + len(vals)),
            0,
            vals,
            color=plt.colormaps['tab10'](c),
            alpha=0.8,
        )
        ax.text(-0.05, y + len(vals) / 2, str(c))
        y += len(vals) + 5
    ax.axvline(
        s.mean(),
        color='black',
        linestyle='--',
        label=f'mean silhouette = {s.mean():.2f}',
    )
    ax.set_title(f'Silhouette plot, k = {k}')
    ax.set_xlabel('silhouette value')
    ax.set_yticks([])
    ax.legend(loc='lower right')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Hierarchical clustering (Ward) of twelve customers
rng = np.random.default_rng(5)
customers = np.vstack(
    [
        rng.normal([1, 1], 0.4, (4, 2)),
        rng.normal([4, 4], 0.3, (4, 2)),
        rng.normal([6, 1.5], 0.4, (4, 2)),
    ]
)
names = list('ABCDEFGHIJKL')

# Hierarchical clustering with Ward linkage
Z = linkage(customers, method='ward')

# Figure showing the customers (left) and the dendrogram (right)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].scatter(customers[:, 0], customers[:, 1], s=60, color='black')
for (x0, y0), name in zip(customers, names):
    axes[0].annotate(name, (x0, y0), xytext=(5, 5), textcoords='offset points')
axes[0].set_title('12 customers (2 features)')
dendrogram(Z, labels=names, ax=axes[1], color_threshold=3)
axes[1].axhline(
    3, color='tab:orange', linestyle='--', label='cut here → 3 clusters'
)
axes[1].set_ylabel('merge distance')
axes[1].set_title('Dendrogram (Ward linkage)')
axes[1].legend()
plt.show()
print(
    'Clusters after the cut:',
    dict(zip(names, fcluster(Z, t=3, criterion='distance').tolist())),
)


# %%NBQA-CELL-SEP42d9ce
# PCA of two correlated features
rng = np.random.default_rng(0)
f1 = rng.normal(0, 1.8, 200)
f2 = 0.65 * f1 + rng.normal(0, 1.0, 200)
F = np.c_[f1, f2]

# Fit PCA and print the explained variance
pca = PCA().fit(F)
evr = pca.explained_variance_ratio_
print('explained variance ratio:', evr.round(3))
print(
    'loadings PC1:',
    pca.components_[0].round(2),
    ' PC2:',
    pca.components_[1].round(2),
)

# Figure showing the principal axes (left) and the explained variance (right)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(F[:, 0], F[:, 1], s=12, alpha=0.6)
for comp, var, name, color in zip(
    pca.components_,
    pca.explained_variance_,
    ['PC1', 'PC2'],
    ['tab:orange', 'navy'],
):
    v = comp * 2 * np.sqrt(var)
    axes[0].annotate(
        '',
        xy=pca.mean_ + v,
        xytext=pca.mean_,
        arrowprops={'arrowstyle': '->', 'color': color, 'lw': 2.5},
    )
    axes[0].text(*(pca.mean_ + v * 1.15), name, color=color, fontweight='bold')
axes[0].set_xlabel('feature 1')
axes[0].set_ylabel('feature 2')
axes[0].set_title('New axes along the largest spread')
axes[0].axis('equal')
axes[1].bar(['PC1', 'PC2'], evr * 100, color=['tab:orange', 'navy'])
for i, v in enumerate(evr * 100):
    axes[1].text(i, v + 1, f'{v:.0f} %', ha='center')
axes[1].set_ylabel('share of total variance [%]')
axes[1].set_title('Explained variance')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
