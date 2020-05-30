import os

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plot_heatmap(data, Y, exp_name, tex, title, fdir=None, dpi=400, use_title=True):
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
    sns.heatmap(data.sum(axis=1, keepdims=True).round(4),
                ax=ax3, 
                annot=True, 
                cmap="YlGnBu", 
                fmt='.2g',
                cbar=False, 
                xticklabels=False, 
                yticklabels=False)

    if use_title:
        ax.set_title(r"Pairwise {tex} {title} distances".format(tex=tex, title=title), fontsize=20)
    fig.tight_layout()

    img_path = 'heatmap_{exp_name}_{title}.png'.format(exp_name=exp_name, title=title)
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=dpi)
    
    
def plot_clusters(X_norm, Y, exp_name, tex, title, n, fdir=None, dpi=400):  
    Y = np.array(Y)
    f = Y != '1 '
    X_norm = X_norm[f]
    Y = Y[f]
    
    pca = PCA(n_components=2, random_state=0)
    X_embedded = pca.fit_transform(X_norm)

    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_norm)

    clusters = kmeans.predict(X_norm)
    
    plt.figure(figsize=(20, 20))
    df = pd.DataFrame(data={'x': X_embedded[:, 0], 
                            'y': X_embedded[:, 1], 
                            'Cluster': clusters})
    df['Cluster'] = df['Cluster'].astype(int) 
    ax = sns.scatterplot(
        x='x', 
        y='y', 
        hue='Cluster',
        palette=sns.color_palette('bright', n),
        s=400, 
        legend=False, 
        data=df
    )
    
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
        title='Cluster',
        loc='best'
    )
    
    # Подписываем точки названием стека + корр функции 
    for i, y in enumerate(Y):
        plt.annotate(y, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=30)
    plt.legend(loc='best')
    plt.title(r"{tex} {title} clustering".format(tex=tex, title=title), fontsize=30)
    img_path = 'clustering_{exp_name}_{title}_{n_clusters}.png'.format(exp_name=exp_name, n_clusters=n, title=title) 
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
    plt.savefig(img_path, dpi=dpi)
    
    
def visualize(X, X_f, Y, scalers, exp_name, tex, ns=[4], fdir=None, dpi=400):
    if scalers is not None:
        tag = exp_name.split('_')[0]
        tag = tag if tag[-1] not in ['X', 'Y', 'Z'] else tag[:-1]
        X_norm = scalers[tag].transform(X)
        X_f_norm = scalers[tag + '_f'].transform(X_f)
    else:
        X_norm = X
        X_f_norm = X_f
        
    for n in ns:
        plot_clusters(X_norm, Y, exp_name=exp_name, tex=tex, title='parameters', n=n, fdir=fdir, dpi=dpi)
    
    for n in ns:
        plot_clusters(X_f_norm, Y, exp_name=exp_name, tex=tex, title='functions', n=n, fdir=fdir, dpi=dpi)
    
    dist_matr = pairwise_distances(X_norm) / np.sqrt(X_norm.shape[1])
    dist_matr_f = pairwise_distances(X_f_norm) / np.sqrt(X_f_norm.shape[1])
    dist_matr_r = sp.stats.rankdata(dist_matr.flatten()).reshape(X_norm.shape[0], X_norm.shape[0])
    dist_matr_r = dist_matr_r / dist_matr_r.max()
    dist_matr_r_f = sp.stats.rankdata(dist_matr_f.flatten()).reshape(X_f_norm.shape[0], X_f_norm.shape[0])
    dist_matr_r_f = dist_matr_r_f / dist_matr_r_f.max()
    
    plot_heatmap(data=dist_matr, Y=Y, exp_name=exp_name, tex=tex, title='parameters', fdir=fdir, dpi=dpi)
    plot_heatmap(data=dist_matr_f, Y=Y, exp_name=exp_name, tex=tex, title='functions', fdir=fdir, dpi=dpi)
    plot_heatmap(data=dist_matr_r, Y=Y, exp_name=exp_name, tex=tex, title='parameters rank', fdir=fdir, dpi=dpi)
    plot_heatmap(data=dist_matr_r_f, Y=Y, exp_name=exp_name, tex=tex, title='functions rank', fdir=fdir, dpi=dpi)
    plt.close('all')