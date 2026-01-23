import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import umap

def kde_plot(query_feat,
             gallery_feat,
             pca_dim=50,
             umap_n_neighbors=15,
             umap_min_dist=0.1,
             umap_metric='euclidean',
             kde_bw_adjust=1.0,
             random_state=42,
             figsize=(8,6),
             save_path=None):
    assert query_feat.ndim == 2 and gallery_feat.ndim == 2
    assert query_feat.shape[1] == gallery_feat.shape[1]

    X = np.concatenate([query_feat, gallery_feat], axis=0)
    n_query = query_feat.shape[0]

    pca = PCA(n_components=pca_dim, random_state=random_state)
    X_pca = pca.fit_transform(X)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state
    )
    X_2d = reducer.fit_transform(X_pca)

    q_2d = X_2d[:n_query]
    g_2d = X_2d[n_query:]

    plt.figure(figsize=figsize)

    sns.kdeplot(
        x=q_2d[:, 0],
        y=q_2d[:, 1],
        fill=True,
        alpha=0.4,
        bw_adjust=kde_bw_adjust,
        label="Query",
        color='tab:blue',
    )

    sns.kdeplot(
        x=g_2d[:, 0],
        y=g_2d[:, 1],
        fill=True,
        alpha=0.4,
        bw_adjust=kde_bw_adjust,
        label="Gallery",
        color="tab:orange",
    )

    handles = [
        Patch(color="tab:blue", label="Query"),
        Patch(color="tab:orange", label="Gallery"),
    ]

    plt.legend(handles=handles)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")