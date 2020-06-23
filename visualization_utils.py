import os

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plot_heatmap(
        data,
        Y,
        exp_name,
        tex,
        title,
        fdir=None,
        dpi=400,
        use_title=True
):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot2grid((20,20), (0,0), colspan=20, rowspan=20)
    # ax2 = plt.subplot2grid((20,20), (18,0), colspan=18, rowspan=2)
    # ax3 = plt.subplot2grid((20,20), (0,18), colspan=2, rowspan=18)

    #with sns.axes_style("white"):
    sns.heatmap(
        data=data,
        ax=ax,
        annot=False,
        cmap="YlGnBu",
        linecolor='b',
        cbar=True,
        cbar_kws={'label': '', 'orientation': 'vertical'}
    )
    # ax.xaxis.tick_top()

    y_labels = Y
    x_labels = Y

    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_xticklabels(x_labels, fontsize=32)
    ax.set_yticklabels(y_labels, fontsize=32)

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
    # ax3.set_title('Total distance', fontsize=20)
    # sns.heatmap(
    #     data.sum(axis=1, keepdims=True).round(4),
    #     ax=ax3,
    #     annot=True,
    #     cmap="YlGnBu",
    #     fmt='.2g',
    #     cbar=False,
    #     xticklabels=False,
    #     yticklabels=False,
    # )

    if use_title:
        ax.set_title(r"Pairwise {tex} {title} distances".format(tex=tex, title=title), fontsize=20)
    fig.tight_layout()

    img_path = 'heatmap_{exp_name}_{title}.png'.format(exp_name=exp_name, title=title)
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
        plt.savefig(img_path, dpi=dpi)
    
    
def plot_clusters(
        X,
        Y,
        exp_name,
        tex,
        title,
        n,
        use_pca=True,
        clusters=None,
        categories=None,
        sizes=None,
        fdir=None,
        dpi=400,
        use_title=True
):
    Y = np.array(Y)
    f = Y != '1 '
    X = X[f]
    Y = Y[f]
    if categories is not None:
        categories = np.array(categories)[f]
    if clusters is not None:
        clusters = np.array(clusters)[f]
    if sizes is not None:
        sizes = np.array(sizes)[f]

    if use_pca:
        pca = PCA(n_components=2, random_state=0)
        X_embedded = pca.fit_transform(X)
    else:
        X_embedded = np.copy(X)

    if clusters is None:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(X)
        clusters = kmeans.predict(X) + 1

    plt.figure(figsize=(20, 20))
    data = {
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Cluster': clusters.astype(int)
    }
    if categories is not None:
        data['Category'] = categories
    if sizes is not None:
        data['Size'] = sizes
    df = pd.DataFrame(data=data)
    if sizes is not None:
        ax = sns.scatterplot(
            x='x',
            y='y',
            hue='Cluster',
            style='Category',
            size='Size',
            palette=sns.color_palette('bright', n),
            data=df,
            sizes=(300, 800),
        )
    else:
        ax = sns.scatterplot(
            x='x',
            y='y',
            hue='Cluster',
            style='Category',
            palette=sns.color_palette('bright', n),
            s=400,
            data=df
        )

    # Подписываем точки названием стека + корр функции
    for i, y in enumerate(Y):
        plt.annotate(y, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=30)
    plt.legend(loc='best', markerscale=2)
    if use_title:
        plt.title(r"{tex} {title} clustering".format(tex=tex, title=title), fontsize=30)
    n_categories = len(set(categories if categories is not None else []))
    img_path = (
        'clustering_{exp_name}_{title}_clusters{n_clusters}_categories{n_categories}.png'
        .format(exp_name=exp_name,
                n_clusters=n,
                n_categories=n_categories,
                title=title)
    )
    if fdir is not None:
        img_path = os.path.join(fdir, img_path)
        plt.savefig(img_path, dpi=dpi)
    
    
def visualize(
        X,
        X_f,
        Y,
        categories,
        scalers,
        exp_name,
        tex,
        ns=[4],
        fdir=None,
        dpi=400,
        use_title=False
):
    if scalers is not None:
        tag = exp_name.split('_')[0]
        tag = tag if tag[-1] not in ['X', 'Y', 'Z'] else tag[:-1]
        X_norm = scalers[tag].transform(X)
        X_f_norm = scalers[tag + '_f'].transform(X_f)
    else:
        X_norm = X
        X_f_norm = X_f
        
    for n in ns:
        for c in categories:
            plot_clusters(
                X_norm,
                Y,
                categories=c,
                exp_name=exp_name,
                tex=tex,
                title='parameters',
                n=n,
                fdir=fdir,
                dpi=dpi,
                use_title=use_title
            )
            plot_clusters(
                X_f_norm,
                Y,
                categories=c,
                exp_name=exp_name,
                tex=tex,
                title='functions',
                n=n,
                fdir=fdir,
                dpi=dpi,
                use_title=use_title
            )

    dist_matr = pairwise_distances(X_norm) / np.sqrt(X_norm.shape[1])
    dist_matr_f = pairwise_distances(X_f_norm) / np.sqrt(X_f_norm.shape[1])
    dist_matr_r = sp.stats.rankdata(dist_matr.flatten()).reshape(X_norm.shape[0], X_norm.shape[0])
    dist_matr_r = dist_matr_r / dist_matr_r.max()
    dist_matr_r_f = sp.stats.rankdata(dist_matr_f.flatten()).reshape(X_f_norm.shape[0], X_f_norm.shape[0])
    dist_matr_r_f = dist_matr_r_f / dist_matr_r_f.max()
    
    plot_heatmap(
        data=dist_matr,
        Y=Y,
        exp_name=exp_name,
        tex=tex,
        title='parameters',
        fdir=fdir,
        dpi=dpi,
        use_title=use_title
    )
    plot_heatmap(
        data=dist_matr_f,
        Y=Y,
        exp_name=exp_name,
        tex=tex,
        title='functions',
        fdir=fdir,
        dpi=dpi,
        use_title=use_title
    )
    plot_heatmap(
        data=dist_matr_r,
        Y=Y,
        exp_name=exp_name,
        tex=tex,
        title='parameters rank',
        fdir=fdir,
        dpi=dpi,
        use_title=use_title
    )
    plot_heatmap(
        data=dist_matr_r_f,
        Y=Y,
        exp_name=exp_name,
        tex=tex,
        title='functions rank',
        fdir=fdir,
        dpi=dpi,
        use_title=use_title
    )
    plt.close('all')