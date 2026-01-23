import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_tsne_by_id(
    feats,
    ids,
    *,
    max_ids: int | None = None,          # plot할 id 개수 (빈도 높은 id부터)
    max_per_id: int | None = None,       # id당 plot할 feature 개수 (랜덤 샘플링)
    id_order: str = "freq",              # "freq" or "sorted"
    seed: int = 0,
    pca_dim: int | None = 50,            # t-SNE 전에 PCA로 차원 축소 (권장). None이면 PCA 생략
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
    tsne_n_iter: int = 1000,
    tsne_init: str = "pca",
    figsize=(10, 8),
    point_size: float = 12,
    alpha: float = 0.75,
    legend: bool = True,
    legend_max_items: int = 30,          # id가 너무 많을 때 legend 폭주 방지
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    feats: torch.Tensor (N,D) or np.ndarray (N,D)
    ids: torch.Tensor/np.ndarray/list (N,)
    반환: (fig, ax, emb2d, sel_ids, sel_indices)
    """

    # ---------- to numpy ----------
    if isinstance(feats, torch.Tensor):
        X = feats.detach().cpu().numpy()
    else:
        X = np.asarray(feats)

    if isinstance(ids, torch.Tensor):
        y = ids.detach().cpu().numpy()
    else:
        y = np.asarray(ids)

    if X.ndim != 2:
        raise ValueError(f"feats must be 2D (N,D). Got shape={X.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"feats N and ids N mismatch: {X.shape[0]} vs {y.shape[0]}")

    rng = np.random.default_rng(seed)

    # ---------- choose ids ----------
    uniq_ids = np.unique(y)
    if max_ids is not None and max_ids < 1:
        raise ValueError("max_ids must be >= 1 or None")

    if id_order == "freq":
        counts = Counter(y.tolist())
        ordered_ids = [i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    elif id_order == "sorted":
        ordered_ids = sorted(uniq_ids.tolist())
    else:
        raise ValueError("id_order must be 'freq' or 'sorted'")

    if max_ids is not None:
        ordered_ids = ordered_ids[:max_ids]

    chosen_id_set = set(ordered_ids)

    # ---------- sample per id ----------
    sel_indices = []
    for _id in ordered_ids:
        idx = np.where(y == _id)[0]
        if max_per_id is not None and len(idx) > max_per_id:
            idx = rng.choice(idx, size=max_per_id, replace=False)
        sel_indices.append(idx)

    if len(sel_indices) == 0:
        raise ValueError("No ids selected. Check max_ids/max_per_id.")

    sel_indices = np.concatenate(sel_indices, axis=0)
    X_sel = X[sel_indices]
    y_sel = y[sel_indices]

    n = X_sel.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 samples for t-SNE, got {n}")

    # ---------- PCA (optional) ----------
    if pca_dim is not None and X_sel.shape[1] > pca_dim:
        X_emb_in = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_sel)
    else:
        X_emb_in = X_sel

    # ---------- t-SNE params safety ----------
    # perplexity must be < n_samples
    # 보수적으로 n-1 보다 작게, 그리고 너무 작으면 자동으로 낮춤
    max_valid_perp = max(2.0, min(float(tsne_perplexity), (n - 1) * 0.9))
    perp = min(float(tsne_perplexity), max_valid_perp)
    # 그래도 n이 매우 작으면 perp를 더 낮춤
    if perp >= n:
        perp = max(2.0, (n - 1) / 3)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=tsne_learning_rate,
        # n_iter=tsne_n_iter,
        init=tsne_init,
        random_state=seed,
        method="barnes_hut" if n >= 1000 else "exact",
    )
    emb2d = tsne.fit_transform(X_emb_in)

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=figsize)

    plot_ids = ordered_ids  # 실제 선택된 id 순서
    k = len(plot_ids)

    # colormap 선택
    if k <= 20:
        cmap = plt.get_cmap("tab20", k)
        colors = [cmap(i) for i in range(k)]
    else:
        cmap = plt.get_cmap("gist_ncar")
        colors = [cmap(i / k) for i in range(k)]

    for ci, _id in enumerate(plot_ids):
        m = (y_sel == _id)
        ax.scatter(
            emb2d[m, 0], emb2d[m, 1],
            s=point_size, alpha=alpha,
            label=str(_id),
            color=colors[ci],
            edgecolors="none",
        )

    # ax.set_xlabel("t-SNE dim 1")
    # ax.set_ylabel("t-SNE dim 2")

    if title is None:
        title = f"t-SNE by id (ids={len(plot_ids)}, samples={n}, perp={perp:.1f})"
    # ax.set_title(title)

    if legend:
        # legend 너무 많으면 일부만 표시
        if k <= legend_max_items:
            ax.legend(loc="best", fontsize=9, frameon=True)
        else:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=8,
                frameon=True,
                ncol=1,
                title=f"ids (showing {legend_max_items}/{k})",
            )

    ax.grid(True, linewidth=0.3, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, emb2d, y_sel, sel_indices


import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D


def plot_tsne_query_gallery(
    q_feats, q_ids,
    g_feats, g_ids,
    *,
    id_mode: str = "intersection",     # "intersection" (추천) or "union"
    max_ids: int | None = None,        # plot할 id 개수 (빈도 높은 id부터)
    max_q_per_id: int | None = None,   # id당 query 샘플 수 제한
    max_g_per_id: int | None = None,   # id당 gallery 샘플 수 제한
    seed: int = 0,
    normalize: bool = False,           # L2 normalize 후 t-SNE (원하면 True)
    pca_dim: int | None = 50,          # t-SNE 전에 PCA (권장). None이면 생략
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
    tsne_n_iter: int = 1000,
    tsne_init: str = "pca",
    figsize=(10, 8),
    q_size: float = 28,
    g_size: float = 24,
    alpha: float = 0.8,
    linewidth: float = 1.2,           # query 원 테두리 두께
    show_id_legend: bool = False,      # id가 많으면 False 추천
    legend_max_ids: int = 25,
    show_modality_legend: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Returns:
      fig, ax, emb2d, meta
        meta: dict with selected indices and arrays (ids, domain)
    domain: 0=query, 1=gallery
    """

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _l2_normalize(X, eps=1e-12):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, eps)

    Q = _to_numpy(q_feats)
    qy = _to_numpy(q_ids).reshape(-1)
    G = _to_numpy(g_feats)
    gy = _to_numpy(g_ids).reshape(-1)

    if Q.ndim != 2 or G.ndim != 2:
        raise ValueError(f"feats must be 2D. Got Q{Q.shape}, G{G.shape}")
    if Q.shape[1] != G.shape[1]:
        raise ValueError(f"Feature dim mismatch: Q dim={Q.shape[1]} vs G dim={G.shape[1]}")
    if Q.shape[0] != qy.shape[0] or G.shape[0] != gy.shape[0]:
        raise ValueError("N mismatch between feats and ids")

    if normalize:
        Q = _l2_normalize(Q)
        G = _l2_normalize(G)

    rng = np.random.default_rng(seed)

    # ----- choose ids to plot -----
    q_set = set(np.unique(qy).tolist())
    g_set = set(np.unique(gy).tolist())

    if id_mode == "intersection":
        chosen_ids = list(q_set & g_set)
    elif id_mode == "union":
        chosen_ids = list(q_set | g_set)
    else:
        raise ValueError("id_mode must be 'intersection' or 'union'")

    if len(chosen_ids) == 0:
        raise ValueError("No ids selected. (intersection is empty?) Try id_mode='union'.")

    # frequency order across BOTH (q+g)
    q_counts = Counter(qy.tolist())
    g_counts = Counter(gy.tolist())
    chosen_ids.sort(key=lambda _id: (-(q_counts[_id] + g_counts[_id]), _id))

    if max_ids is not None:
        if max_ids < 1:
            raise ValueError("max_ids must be >= 1 or None")
        chosen_ids = chosen_ids[:max_ids]

    # ----- sample indices per id -----
    q_sel_idx_list, g_sel_idx_list = [], []
    for _id in chosen_ids:
        q_idx = np.where(qy == _id)[0]
        g_idx = np.where(gy == _id)[0]

        if max_q_per_id is not None and len(q_idx) > max_q_per_id:
            q_idx = rng.choice(q_idx, size=max_q_per_id, replace=False)
        if max_g_per_id is not None and len(g_idx) > max_g_per_id:
            g_idx = rng.choice(g_idx, size=max_g_per_id, replace=False)

        if len(q_idx) > 0:
            q_sel_idx_list.append(q_idx)
        if len(g_idx) > 0:
            g_sel_idx_list.append(g_idx)

    q_sel_idx = np.concatenate(q_sel_idx_list, axis=0) if len(q_sel_idx_list) else np.array([], dtype=int)
    g_sel_idx = np.concatenate(g_sel_idx_list, axis=0) if len(g_sel_idx_list) else np.array([], dtype=int)

    X_q = Q[q_sel_idx] if q_sel_idx.size else np.zeros((0, Q.shape[1]), dtype=Q.dtype)
    X_g = G[g_sel_idx] if g_sel_idx.size else np.zeros((0, G.shape[1]), dtype=G.dtype)

    y_q = qy[q_sel_idx] if q_sel_idx.size else np.array([], dtype=qy.dtype)
    y_g = gy[g_sel_idx] if g_sel_idx.size else np.array([], dtype=gy.dtype)

    X = np.concatenate([X_q, X_g], axis=0)
    y = np.concatenate([y_q, y_g], axis=0)
    domain = np.concatenate(
        [np.zeros(len(X_q), dtype=np.int32), np.ones(len(X_g), dtype=np.int32)],
        axis=0
    )

    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 samples for t-SNE, got {n}")

    # ----- PCA (optional) -----
    if pca_dim is not None and X.shape[1] > pca_dim:
        X_in = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    else:
        X_in = X

    # ----- t-SNE perplexity safety -----
    perp = float(tsne_perplexity)
    # perplexity must be < n_samples
    if perp >= n:
        perp = max(2.0, (n - 1) / 3)
    # 약간 보수적으로
    perp = min(perp, max(2.0, (n - 1) * 0.9))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=tsne_learning_rate,
        # n_iter=tsne_n_iter,
        init=tsne_init,
        random_state=seed,
        method="barnes_hut" if n >= 1000 else "exact",
    )
    emb2d = tsne.fit_transform(X_in)

    # ----- plot -----
    fig, ax = plt.subplots(figsize=figsize)

    k = len(chosen_ids)
    if k <= 20:
        cmap = plt.get_cmap("tab20", k)
        colors = [cmap(i) for i in range(k)]
    else:
        cmap = plt.get_cmap("gist_ncar")
        colors = [cmap(i / k) for i in range(k)]

    for ci, _id in enumerate(chosen_ids):
        c = colors[ci]

        m_q = (y == _id) & (domain == 0)
        m_g = (y == _id) & (domain == 1)

        # query: 테두리 있는 원 (내부 비움)
        if np.any(m_q):
            ax.scatter(
                emb2d[m_q, 0], emb2d[m_q, 1],
                marker="o",
                s=q_size,
                facecolors=[c],
                edgecolors='black',
                linewidths=linewidth,
                alpha=alpha,
                label=str(_id) if show_id_legend else None,
            )

        # gallery: 테두리 없는 x자
        if np.any(m_g):
            ax.scatter(
                emb2d[m_g, 0], emb2d[m_g, 1],
                marker="x",
                s=g_size,
                c=[c],
                alpha=alpha,
                label=None,  # id legend는 query 쪽에서만 (중복 방지)
            )

    # ax.set_xlabel("t-SNE dim 1")
    # ax.set_ylabel("t-SNE dim 2")

    if title is None:
        title = f"t-SNE (query vs gallery) | ids={k}, samples={n}, mode={id_mode}, perp={perp:.1f}"
    # ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.3)

    # modality legend (style legend)
    if show_modality_legend:
        handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor='none', markeredgecolor='black',
                   markersize=8, label='Query (o)'),
            Line2D([0], [0], marker='x', linestyle='None',
                    markersize=8, label='Gallery (x)'),
        ]
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    # id legend (optional)
    if show_id_legend:
        # id legend를 modality legend와 같이 쓰면 겹칠 수 있어 위치를 옮김
        # 너무 많으면 제한
        if k <= legend_max_ids:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)
        else:
            # 너무 많으면 아예 끄는 걸 권장하지만, 요청 시 일부만 표시하도록 타협
            shown = chosen_ids[:legend_max_ids]
            # 임시로 "query 스캐터"만 label 달아놨기 때문에 이미 제한된 상태에서만 의미 있음
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    meta = {
        "chosen_ids": chosen_ids,
        "q_sel_idx": q_sel_idx,
        "g_sel_idx": g_sel_idx,
        "ids_all": y,
        "domain_all": domain,  # 0=query, 1=gallery
    }
    return fig, ax, emb2d, meta
