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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Exercise 1: k-means by hand
points = pd.DataFrame(
    {'x1': [1, 2, 4, 5, 1, 5], 'x2': [1, 1, 3, 4, 2, 5]}, index=list('ABCDEF')
)
centres = np.array([[1.0, 1.0], [4.0, 3.0]])

# k-means iterations: assign points, update centres
for it in range(1, 4):
    dist = np.sqrt(
        ((points.values[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    )
    labels = dist.argmin(axis=1)
    print(f'Iteration {it}: distances to centres =')
    print(
        pd.DataFrame(
            dist.round(2), index=points.index, columns=['to mu1', 'to mu2']
        ).T
    )
    print(
        '  assignment:', {p: int(l) + 1 for p, l in zip(points.index, labels)}
    )
    new_centres = np.array(
        [points.values[labels == c].mean(axis=0) for c in range(2)]
    )
    print('  new centres:', new_centres.round(3).tolist(), '\n')
    if np.allclose(new_centres, centres):
        print('Converged: the centres do not change any more.')
        break
    centres = new_centres

# Within-cluster sum of squares of the final clusters
wcss = sum(
    ((points.values[labels == c] - centres[c]) ** 2).sum() for c in range(2)
)
print('WCSS =', round(wcss, 3))

# e) Check with scikit-learn
km_check = KMeans(n_clusters=2, init=np.array([[1, 1], [4, 3]]), n_init=1).fit(
    points
)
print(
    'scikit-learn: centres =',
    km_check.cluster_centers_.round(3).tolist(),
    ' inertia =',
    round(km_check.inertia_, 3),
)


# %%NBQA-CELL-SEP42d9ce
# a) Import the data
bikes = pd.read_csv('../00_Data/zurich_bike_counts_2024_2025.csv', sep=',')
stations = pd.read_csv('../00_Data/zurich_bike_stations.csv', sep=',')
S = stations['column'].tolist()  # the 16 station columns

# Masks for working days, summer/winter and rainy/dry days
work = (bikes['weekend'] == 0) & (bikes['holiday'] == 0)
summer, winter = bikes['month'].isin([5, 6, 7, 8, 9]), bikes['month'].isin(
    [11, 12, 1, 2]
)
rain, dry = bikes['rain_duration_min'] > 240, bikes['rain_duration_min'] == 0

# Features per counting station
st = pd.DataFrame(index=S)
st['name'] = stations.set_index('column')['name']
st['bikes_per_day'] = bikes[S].mean()
st['weekend_ratio'] = (
    bikes.loc[bikes['weekend'] == 1, S].mean() / bikes.loc[work, S].mean()
)
st['summer_winter_ratio'] = (
    bikes.loc[work & summer, S].mean() / bikes.loc[work & winter, S].mean()
)
st['rain_ratio'] = (
    bikes.loc[work & rain, S].mean() / bikes.loc[work & dry, S].mean()
)
st['school_holiday_ratio'] = (
    bikes.loc[work & (bikes['school_holiday'] == 1), S].mean()
    / bikes.loc[work & (bikes['school_holiday'] == 0), S].mean()
)
st['temp_corr'] = bikes[S].corrwith(bikes['temp_mean'])

# b) Table sorted by weekend_ratio
print(st.sort_values('weekend_ratio').round(2).to_string())


# %%NBQA-CELL-SEP42d9ce
# Exercise 3: k-means - choosing k
features = [
    'weekend_ratio',
    'summer_winter_ratio',
    'rain_ratio',
    'school_holiday_ratio',
    'temp_corr',
]

# a) Standardise
X = StandardScaler().fit_transform(st[features])

# b) Elbow plot and silhouette score
ks = range(1, 9)
wcss, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)

# Figure showing the elbow plot and the silhouette score by k
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks, wcss, marker='o')
axes[0].set_xlabel('Number of clusters k')
axes[0].set_ylabel('WCSS (inertia)')
axes[0].set_title('Elbow plot')
axes[1].plot(ks, sil, marker='o', color='darkred')
axes[1].set_xlabel('Number of clusters k')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score')
for ax in axes:
    ax.grid(alpha=0.3)
plt.show()
print(
    pd.DataFrame(
        {'k': ks, 'WCSS': np.round(wcss, 1), 'silhouette': np.round(sil, 3)}
    ).to_string(index=False)
)


# %%NBQA-CELL-SEP42d9ce
# a) k-means with k = 3
km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
st['cluster'] = km3.labels_

# b) Cluster profiles and stations
print(
    st.groupby('cluster')[features + ['bikes_per_day']]
    .mean()
    .round(2)
    .to_string()
)
for c in range(3):
    print(f'Cluster {c}:', st.loc[st['cluster'] == c, 'name'].tolist())

# c) Weekly profile per cluster
weekdays = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]
# Weekly profile of each station relative to its mean
weekly = bikes.groupby('weekday')[S].mean().loc[weekdays] / bikes[S].mean()
# Figure showing the average weekly profile per cluster
plt.figure(figsize=(8, 4))
for c in range(3):
    plt.plot(
        weekdays,
        weekly[st.index[st['cluster'] == c]].mean(axis=1),
        marker='o',
        label=f'cluster {c}',
    )
plt.axhline(1, color='grey', linewidth=1)
plt.ylabel('Bikes / station mean')
plt.title('Average weekly profile per cluster')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Exercise 5: silhouette analysis
def silhouette_plot(ax, X, labels, names):
    """Silhouette plot: one bar per observation, sorted within clusters."""
    s = silhouette_samples(X, labels)
    y, ticks, ticklabels = 0, [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(s[idx])]
        ax.barh(
            range(y, y + len(idx)),
            s[idx],
            height=0.8,
            color=plt.colormaps['tab10'](c),
            label=f'cluster {c}',
        )
        ticks += list(range(y, y + len(idx)))
        ticklabels += [names[i] for i in idx]
        y += len(idx) + 1
    ax.axvline(
        s.mean(), color='black', linestyle='--', label=f'mean = {s.mean():.2f}'
    )
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_xlabel('silhouette value')
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    return s


# a) Silhouette plots for k = 2, 3, 4
names = st['name'].tolist()
# Figure showing silhouette plots for k = 2, 3 and 4
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, k in zip(axes, [2, 3, 4]):
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    s = silhouette_plot(ax, X, labels_k, names)
    ax.set_title(f'k = {k}')
plt.tight_layout()
plt.show()

# b) Silhouette values for k = 3
st['silhouette'] = silhouette_samples(X, km3.labels_)
print(
    st[['name', 'cluster', 'silhouette']]
    .sort_values(['cluster', 'silhouette'])
    .round(2)
    .to_string(index=False)
)
print('Mean silhouette per cluster:')
print(st.groupby('cluster')['silhouette'].mean().round(2))


# %%NBQA-CELL-SEP42d9ce
# Exercise 6: k-means with n_init=1 for random_state = 0, ..., 19;
#    WCSS and ARI compared with km3
rows = []
for seed in range(20):
    km = KMeans(n_clusters=3, n_init=1, random_state=seed).fit(X)
    rows.append(
        {
            'random_state': seed,
            'WCSS': round(km.inertia_, 2),
            'ARI vs. reference': round(
                adjusted_rand_score(km3.labels_, km.labels_), 3
            ),
        }
    )
stability = pd.DataFrame(rows)
print(stability.to_string(index=False))
print('\nReference solution (n_init=10): WCSS =', round(km3.inertia_, 2))


# %%NBQA-CELL-SEP42d9ce
# a) Ward linkage and dendrogram
Z = linkage(X, method='ward')
# Figure showing the dendrogram of the counting stations
plt.figure(figsize=(12, 5))
dendrogram(
    Z, labels=st['name'].tolist(), leaf_rotation=90, color_threshold=Z[-3, 2]
)
plt.ylabel('Merge distance (Ward)')
plt.title('Dendrogram of the 16 counting stations')
plt.show()

# b) Cut into 3 clusters and compare with k-means
st['hc_cluster'] = fcluster(Z, t=3, criterion='maxclust')
print(
    pd.crosstab(
        st['cluster'],
        st['hc_cluster'],
        rownames=['k-means'],
        colnames=['hierarchical'],
    )
)
print(
    'ARI k-means vs. hierarchical:',
    round(adjusted_rand_score(st['cluster'], st['hc_cluster']), 3),
)


# %%NBQA-CELL-SEP42d9ce
# a) PCA on the days
Z_days = StandardScaler().fit_transform(bikes[S])
pca = PCA()
scores = pca.fit_transform(Z_days)
evr = pca.explained_variance_ratio_
print(
    'Explained variance ratio (PC1-PC5):',
    evr[:5].round(3),
    ' cumulative:',
    evr[:5].cumsum().round(3),
)

# Figure showing the explained variance of the first 8 components
plt.figure(figsize=(7, 3.5))
pcs = [f'PC{i}' for i in range(1, 9)]
plt.bar(pcs, evr[:8], color='steelblue')
plt.ylabel('Share of explained variance')
plt.show()

# b) Loadings of PC1 and PC2
loadings = pd.DataFrame(
    pca.components_[:2].T, index=st['name'], columns=['PC1', 'PC2']
)
print(loadings.round(2).sort_values('PC2').to_string())

# c) Days in the PC1-PC2 plane
plt.figure(figsize=(8, 5.5))
for w, label, color in [
    (0, 'working day', 'tab:blue'),
    (1, 'weekend', 'tab:red'),
]:
    m = bikes['weekend'] == w
    plt.scatter(
        scores[m, 0], scores[m, 1], s=12, alpha=0.6, color=color, label=label
    )
plt.xlabel(f'PC1 ({evr[0]:.0%} of variance)')
plt.ylabel(f'PC2 ({evr[1]:.0%} of variance)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Days with the highest and lowest PC2 scores
days = bikes[
    ['date', 'weekday', 'temp_max', 'rain_duration_min', 'bikes_total']
].assign(PC1=scores[:, 0], PC2=scores[:, 1])
print('Highest PC2:')
print(days.nlargest(5, 'PC2').round(1).to_string(index=False))
print('Lowest PC2:')
print(days.nsmallest(5, 'PC2').round(1).to_string(index=False))


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
