import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def visualize(X, Y, exp_name, n=4):
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    X_embedded = pca.fit_transform(X_norm)

    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_norm)

    clusters = kmeans.predict(X_norm)

    dist_matr = pairwise_distances(X_norm)

    plt.figure(figsize=(20, 20))
    for i in range(n):
        idxes,  = np.where(clusters == i)
        plt.scatter(X_embedded[idxes, 0],X_embedded[idxes, 1], label=str(i))
    for i, y in enumerate(Y):
        plt.annotate(y, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=12)
    plt.legend(loc='best')
    plt.savefig('clustering_{}.png'.format(exp_name))

    y_labels = Y
    x_labels = Y

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(dist_matr)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_yticklabels(y_labels, fontsize=16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, round(dist_matr[i, j], 3),
                           ha="center", va="center", color="w", fontdict={'fontsize': 16})

    ax.set_title("Samples pairwise distances", fontsize=20)
    fig.tight_layout()
    plt.savefig('heatmap_{}.png'.format(exp_name))