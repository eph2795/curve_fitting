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

# sns.heatmap((pd.DataFrame(pv.sum(axis=0))).transpose(), ax=ax2,  annot=True, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)
# sns.heatmap(pd.DataFrame(pv.sum(axis=1)), ax=ax3,  annot=True, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)

def plot_heatmap(data, Y, exp_name, tex, title, fdir=None, dpi=400):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot2grid((20,20), (0,0), colspan=18, rowspan=18)
    # ax2 = plt.subplot2grid((20,20), (18,0), colspan=18, rowspan=2)
    ax3 = plt.subplot2grid((20,20), (0,18), colspan=2, rowspan=18)

    #with sns.axes_style("white"):
    sns.heatmap(data, ax=ax, annot=True, cmap="YlGnBu", linecolor='b', cbar=False)
    # ax.xaxis.tick_top()

    y_labels = Y
    x_labels = Y

    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticklabels(y_labels, fontsize=12)

    # Rotate the tick labels and set their alignment.
    for tick in ax.get_yticklabels():
        tick.set_rotation(360)
#     plt.yticks(rotation=360)
    plt.setp(ax.get_xticklabels(), 
             rotation=45, 
             ha="right",
             rotation_mode="anchor")

    # sns.heatmap(dist_matr.sum(axis=0, keepdims=True).round(3), 
    #             ax=ax2,  
    #             annot=True, 
    #             cmap="YlGnBu", 
    #             fmt='g',
    #             cbar=False, 
    #             xticklabels=False, 
    #             yticklabels=False)
    ax3.set_title('Total distance', fontsize=20)
    sns.heatmap(data.sum(axis=1, keepdims=True).round(3),
                ax=ax3, 
                annot=True, 
                cmap="YlGnBu", 
                fmt='g',
                cbar=False, 
                xticklabels=False, 
                yticklabels=False)

    ax.set_title(r"Samples pairwise {tex} {title} distances".format(tex=tex, title=title), fontsize=20)
    fig.tight_layout()

    img_path = 'heatmap_{exp_name}_{title}.png'.format(exp_name=exp_name, title=title)
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=dpi)
    
    
def visualize(X, X_f, Y, exp_name, tex, n=4, fdir=None, dpi=400):
    X_norm = StandardScaler().fit_transform(X)
    X_f_norm = StandardScaler().fit_transform(X_f)
    pca = PCA(n_components=2, random_state=0)
    X_embedded = pca.fit_transform(X_norm)

    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_norm)

    clusters = kmeans.predict(X_norm)

    dist_matr = pairwise_distances(X_norm)
    dist_matr_f = pairwise_distances(X_f_norm)
    
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
        plt.annotate(y, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=25)
    plt.legend(loc='best')
   
    img_path = 'clustering_{exp_name}_{n_clusters}.png'.format(exp_name=exp_name, n_clusters=n) 
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=dpi)
    
    plot_heatmap(data=dist_matr, Y=Y, exp_name=exp_name, tex=tex, title='parameters', fdir=fdir, dpi=dpi)
    plot_heatmap(data=dist_matr_f, Y=Y, exp_name=exp_name, tex=tex, title='functions', fdir=fdir, dpi=dpi)
#     y_labels = Y
#     x_labels = Y

#     fig, ax = plt.subplots(figsize=(20, 20))
#     im = ax.imshow(dist_matr)

#     # We want to show all ticks...
#     ax.set_xticks(np.arange(len(x_labels)))
#     ax.set_yticks(np.arange(len(y_labels)))
#     # ... and label them with the respective list entries
#     ax.set_xticklabels(x_labels, fontsize=16)
#     ax.set_yticklabels(y_labels, fontsize=16)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     for i in range(len(y_labels)):
#         for j in range(len(x_labels)):
#             text = ax.text(j, i, round(dist_matr[i, j], 3),
#                            ha="center", 
#                            va="center", 
#                            color="w", 
#                            fontdict={'fontsize': 16 if len(y_labels) <= 16 else 9})

#     ax.set_title("Samples pairwise distances", fontsize=20)
#     fig.tight_layout()
    
#     img_path = 'heatmap_{}.png'.format(exp_name)
#     if fdir is not None:
#         img_path = os.path.join(fdir, img_path)
#     plt.savefig(img_path, dpi=400)
    
    plt.close('all')