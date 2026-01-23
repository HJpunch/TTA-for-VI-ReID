import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def plot_query_gallery_similarity_distribution(
    query_feats,
    query_ids,
    gallery_feats,
    gallery_ids,
    query_id_sample_ratio=0.2,
    metric='cosine',
    random_state=42,
    max_pairs=200_000,
    figsize=(8, 6),
    save_path=None
):
    """
    Plot cosine similarity distributions:
      (query vs gallery with same ID set)
      (query vs gallery with non-query ID set)
    """
    assert metric in ['cosine', 'euclidean']
    rng = np.random.RandomState(random_state)

    # --------------------------------------------------
    # 1. Sample 20% query IDs
    # --------------------------------------------------
    unique_qids = np.unique(query_ids)
    num_sample = int(len(unique_qids) * query_id_sample_ratio)

    sampled_qids = rng.choice(
        unique_qids,
        size=num_sample,
        replace=False
    )
    sampled_qids = set(sampled_qids)

    # --------------------------------------------------
    # 2. Split query / gallery
    # --------------------------------------------------
    q_mask = np.isin(query_ids, list(sampled_qids))
    g_pos_mask = np.isin(gallery_ids, list(sampled_qids))
    g_neg_mask = ~g_pos_mask

    if metric == 'euclidean':
        query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
        gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    q_feats = query_feats[q_mask]
    g_pos_feats = gallery_feats[g_pos_mask]
    g_neg_feats = gallery_feats[g_neg_mask]

    print(f"Sampled query IDs: {len(sampled_qids)}")
    print(f"Query samples: {len(q_feats)}")
    print(f"Positive gallery samples: {len(g_pos_feats)}")
    print(f"Negative gallery samples: {len(g_neg_feats)}")

    # --------------------------------------------------
    # 3. Cosine similarity
    # --------------------------------------------------
    if metric == 'cosine':
        sim_pos = cosine_similarity(q_feats, g_pos_feats).ravel()
        sim_neg = cosine_similarity(q_feats, g_neg_feats).ravel()
    elif metric == 'euclidean':
        sim_pos = euclidean_distances(q_feats, g_pos_feats).ravel()
        sim_neg = euclidean_distances(q_feats, g_neg_feats).ravel()

    # --------------------------------------------------
    # 4. Optional subsampling (for speed / memory)
    # --------------------------------------------------
    if len(sim_pos) > max_pairs:
        sim_pos = rng.choice(sim_pos, max_pairs, replace=False)

    if len(sim_neg) > max_pairs:
        sim_neg = rng.choice(sim_neg, max_pairs, replace=False)

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    sns.kdeplot(
        sim_pos,
        label="Query vs Positive Gallery",
        fill=True,
        alpha=0.4,
    )

    sns.kdeplot(
        sim_neg,
        label="Query vs Negative Gallery",
        fill=True,
        alpha=0.4,
    )

    if metric == 'cosine':
        plt.xlabel("Cosine Similarity")
    elif metric == 'euclidean':
        plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Query–Gallery Cosine Similarity Distribution")
    plt.legend()
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(f"{metric}_{save_path}", dpi=200, bbox_inches="tight")

    return sim_pos, sim_neg

