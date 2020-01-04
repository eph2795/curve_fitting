import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def visualize(X, Y, exp_name, n=4, fdir=None):
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    X_embedded = pca.fit_transform(X_norm)

    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_norm)

    clusters = kmeans.predict(X_norm)

    dist_matr = pairwise_distances(X_norm)

    plt.figure(figsize=(20, 20))
    # Точки векторов, спроецированные с помощью PCA на плоскость
#     for i in range(n):
#         idxes,  = np.where(clusters == i)
#         plt.scatter(X_embedded[idxes, 0],X_embedded[idxes, 1], label=str(i), s=16000)
#     print(clusters.shape)
    df = pd.DataFrame(data={'x': X_embedded[:, 0], 
                            'y': X_embedded[:, 1], 
                            'Cluster': clusters})
    df['Cluster'] = df['Cluster'].astype(int) 
    ax = sns.scatterplot(x='x', 
                    y='y', 
                    hue='Cluster',
                    palette=sns.color_palette('bright', n),
                    s=400, 
                    legend=False, 
                    data=df)
    
    palette = sns.color_palette('bright', n)
    for i, c in enumerate(palette):
        plt.scatter([], [], 
                    c=[c], 
                    s=300,
                    label=str(i + 1))
    plt.legend(
        scatterpoints=1, 
#         frameon=True, 
#         labelspacing=0.5, 
        title='Cluster'
    )
    
    # Подписываем точки названием стека + корр функции 
    for i, y in enumerate(Y):
        plt.annotate(y, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=30)
    plt.legend(loc='best')
   
    img_path = 'clustering_{exp_name}_{n_clusters}.png'.format(exp_name=exp_name, n_clusters=n) 
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=400)

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
    
    img_path = 'heatmap_{}.png'.format(exp_name)
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=400)
    plt.close('all')