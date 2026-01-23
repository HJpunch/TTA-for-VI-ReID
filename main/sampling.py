import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

import numpy as np
import torch
import torch.nn.functional as F

def relabel_and_make_prototypes(predefined_feats: torch.Tensor,
                                predefined_ids: np.ndarray,
                                normalize_proto: bool = True):
    """
    predefined_ids를 등장 순서대로 0..K-1로 relabel하고,
    pid별 prototype(평균 feature) 생성.

    Args:
        predefined_feats: torch.Tensor, (N, D)
        predefined_ids: np.ndarray, (N,)
        normalize_proto: prototype을 L2 normalize 할지 여부

    Returns:
        relabeled_ids: np.ndarray, (N,)  # 0..K-1
        id_map: dict[old_id -> new_id]
        prototypes: torch.Tensor, (K, D)
    """
    assert predefined_feats.dim() == 2
    N, D = predefined_feats.shape
    assert len(predefined_ids) == N

    # ---- (1) relabel in order of first appearance ----
    id_map = {}
    relabeled_ids = np.empty_like(predefined_ids, dtype=np.int64)
    next_id = 0
    for i, oid in enumerate(predefined_ids.tolist()):
        if oid not in id_map:
            id_map[oid] = next_id
            next_id += 1
        relabeled_ids[i] = id_map[oid]

    K = next_id  # number of unique pids

    # ---- (2) prototypes: mean per pid ----
    device = predefined_feats.device
    relabeled_t = torch.from_numpy(relabeled_ids).to(device=device)

    prototypes = torch.zeros((K, D), device=device, dtype=predefined_feats.dtype)
    counts = torch.zeros((K,), device=device, dtype=torch.long)

    prototypes.index_add_(0, relabeled_t, predefined_feats)
    counts.index_add_(0, relabeled_t, torch.ones_like(relabeled_t, dtype=torch.long))

    prototypes = prototypes / counts.clamp_min(1).unsqueeze(1)

    if normalize_proto:
        prototypes = F.normalize(prototypes, dim=1)

    return relabeled_ids, id_map, prototypes


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def sample_queries_by_dbscan_and_prototypes(
    gallery_feats: torch.Tensor,          # (N, D)  (보통 N=num=32)
    num: int,                             # query를 총 몇 개 뽑을지 (보통 N과 같게)
    predefined_feats: torch.Tensor,       # (P, D)
    relabeled_ids: np.ndarray,            # (P,) 0..K-1
    prototypes: torch.Tensor,             # (K, D)
    sim_th: float = 0.65,
    margin: float = 0.05,
    dbscan_eps: float = 0.25,             # cosine distance eps (1 - cosine_sim)
    dbscan_min_samples: int = 2,
    unique_query: bool = True,            # query instance 중복 방지(가능하면)
    random_state: int = 42,
):
    """
    Returns:
      - pseudo_gallery_pids: torch.LongTensor (N,)   # 0..K-1 or -1
      - db_labels: np.ndarray (N,)                   # DBSCAN cluster labels (-1 = noise)
      - cluster_pid: dict[int -> int]                # cluster_id -> pid (or -1)
      - sampled_query_feats: torch.Tensor (num, D)
      - sampled_query_ids: np.ndarray (num,)         # 0..K-1
      - sampled_query_indices: np.ndarray (num,)     # indices into predefined pool
      - centroids: torch.Tensor (C, D)               # C clusters (noise 제외)
    """

    rng = np.random.RandomState(random_state)
    device = gallery_feats.device
    dtype = gallery_feats.dtype

    # ---- normalize for cosine ----
    gallery_feats = F.normalize(gallery_feats, dim=1)
    predefined_feats = predefined_feats.to(device=device, dtype=dtype)
    predefined_feats = F.normalize(predefined_feats, dim=1)
    prototypes = prototypes.to(device=device, dtype=dtype)
    prototypes = F.normalize(prototypes, dim=1)

    N, D = gallery_feats.shape
    P = predefined_feats.shape[0]
    num = min(num, P)

    # =========================================================
    # 1) DBSCAN clustering on gallery feats (CPU, cosine metric)
    # =========================================================
    g_np = gallery_feats.detach().cpu().numpy()
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
    db_labels = db.fit_predict(g_np)  # (N,)
    cluster_ids = sorted(set(db_labels) - {-1})  # noise 제외

    # edge case: 클러스터가 하나도 없으면 -> 랜덤 query 샘플링 + gallery는 전부 -1
    pseudo_gallery_pids = torch.full((N,), -1, device=device, dtype=torch.long)
    if len(cluster_ids) == 0:
        print("cluster zero!!!!!!!!!!!!!")
        idx = rng.choice(P, size=num, replace=False if num <= P else True)
        sampled_query_feats = predefined_feats[idx]
        sampled_query_ids = relabeled_ids[idx]
        return (pseudo_gallery_pids, db_labels, {}, sampled_query_feats,
                sampled_query_ids, idx, torch.empty((0, D), device=device, dtype=dtype))

    # =========================================================
    # 2) centroid 계산 + prototype 매칭 (th + margin gating)
    # =========================================================
    centroids = []
    cluster_pid = {}          # cluster_id -> pid or -1
    confident_clusters = []   # pid가 부여된 cluster_id만

    for cid in cluster_ids:
        mask = (db_labels == cid)
        c_feat = gallery_feats[torch.from_numpy(mask).to(device=device)].mean(dim=0)
        c_feat = F.normalize(c_feat, dim=0)
        centroids.append(c_feat)

    centroids = torch.stack(centroids, dim=0)  # (C, D)
    sim = centroids @ prototypes.t()           # (C, K)

    top2 = torch.topk(sim, k=min(2, sim.size(1)), dim=1)
    s1 = top2.values[:, 0]
    i1 = top2.indices[:, 0]
    s2 = top2.values[:, 1] if sim.size(1) >= 2 else torch.full_like(s1, -1.0)

    conf = s1 - s2
    # print("s1")
    # print(s1)
    # print("conf")
    # print(conf)
    assigned = (s1 >= sim_th) & (conf >= margin)  # (C,)

    # cluster_id list와 centroids index가 같은 순서이므로 매핑
    for c_idx, cid in enumerate(cluster_ids):
        if assigned[c_idx].item():
            pid = int(i1[c_idx].item())  # 0..K-1
            cluster_pid[cid] = pid
            confident_clusters.append(cid)
        else:
            cluster_pid[cid] = -1

    # gallery pseudo label 부여 (confident cluster에만)
    for cid in confident_clusters:
        pid = cluster_pid[cid]
        mask = (db_labels == cid)
        pseudo_gallery_pids[torch.from_numpy(mask).to(device=device)] = pid

    # =========================================================
    # 3) query 샘플링: confident cluster 수만큼 균등 분배
    #    (예: C=5면 [7,7,6,6,6])
    # =========================================================
    # query를 뽑을 cluster: "confidence 통과한 클러스터"만 사용 (신뢰도 우선)
    Cc = len(confident_clusters)

    # edge case: confident cluster가 0개면 -> 랜덤 query 샘플링, gallery는 전부 -1
    if Cc == 0:
        idx = rng.choice(P, size=num, replace=False if num <= P else True)
        sampled_query_feats = predefined_feats[idx]
        sampled_query_ids = relabeled_ids[idx]
        return (pseudo_gallery_pids, db_labels, cluster_pid, sampled_query_feats,
                sampled_query_ids, idx, centroids)

    base = num // Cc
    rem = num % Cc
    ks = [base + (1 if i < rem else 0) for i in range(Cc)]  # sum == num

    # pid -> indices pool (CPU)
    relabeled_ids = relabeled_ids.astype(np.int64)
    K = prototypes.size(0)
    pid_to_indices = [np.where(relabeled_ids == pid)[0] for pid in range(K)]

    selected = []
    selected_set = set()

    for c_i, cid in enumerate(confident_clusters):
        pid = cluster_pid[cid]
        need = ks[c_i]
        pool = pid_to_indices[pid]
        if pool.size == 0:
            continue

        # 중복 방지 샘플링(가능하면)
        if unique_query:
            pool = [int(x) for x in pool.tolist() if x not in selected_set]
            if len(pool) >= need:
                pick = rng.choice(pool, size=need, replace=False)
            else:
                # 부족하면 가능한 만큼 뽑고, 나머지는 나중에 fill
                pick = np.array(pool, dtype=np.int64)
            for j in pick.tolist():
                selected.append(j)
                selected_set.add(j)
        else:
            replace = (pool.size < need)
            pick = rng.choice(pool, size=need, replace=replace)
            selected.extend(pick.tolist())

    # 부족분 채우기: 전 pool에서 랜덤 (중복 방지 가능하면)
    if len(selected) < num:
        need = num - len(selected)
        if unique_query:
            remain = [i for i in range(P) if i not in selected_set]
            if len(remain) >= need:
                fill = rng.choice(remain, size=need, replace=False)
            else:
                fill = np.array(remain, dtype=np.int64)
        else:
            fill = rng.choice(np.arange(P), size=need, replace=(P < need))
        selected.extend(fill.tolist())

    selected = np.array(selected[:num], dtype=np.int64)

    sampled_query_feats = predefined_feats[selected]          # (num, D)
    sampled_query_ids = relabeled_ids[selected]               # (num,)

    return (pseudo_gallery_pids, db_labels, cluster_pid,
            sampled_query_feats, sampled_query_ids, selected, centroids)


def assign_gallery_pseudo_label(memory:torch.Tensor, threshold=0.6):
    r"""
    gallery만으로 memory 운용하는 세팅에서
    batch 내의 gallery one by one으로 similarity 비교하고,
    threshold 이상이면 가장 similarity가 높은 feature와 같은 label로,
    다르면 label 추가

    memory: normalized feature memory (64, 2048)
    """
    assert memory.dim() == 2
    N = memory.size(0)

    feats = memory
    labels = torch.empty(N, device=memory.device, dtype=torch.long)

    next_label = 0
    labels[0] = next_label
    next_label += 1

    for i in range(1, N):
        sims = feats[:i] @ feats[i]  # gallery feature는 normalized 되어있어야 함.
        # print(sims)
        best_sim, best_j = sims.max(dim=0)

        if best_sim.item() >= threshold:
            labels[i] = labels[best_j]
        else:
            labels[i] = next_label
            next_label += 1

    return labels


def sample_pids_k_instances(
    predefined_feats: torch.Tensor,  # (N, D)
    predefined_ids: np.ndarray,      # (N,)
    p: int,
    k: int,
    seed: int | None = 42,
    replace_pids: bool = False,
):
    """
    RandomIdentitySampler 스타일:
      - 랜덤으로 p개의 pid를 선택
      - 각 pid에서 k개의 instance 인덱스를 샘플링(부족하면 복원추출)
      - (p*k, D) feats, (p*k,) ids 반환

    Args:
        predefined_feats: torch.Tensor (N, D)
        predefined_ids: np.ndarray (N,)
        p: number of identities per batch
        k: number of instances per identity
        seed: random seed (optional)
        replace_pids: unique pid 수가 p보다 작을 때 pid를 중복 허용할지

    Returns:
        batch_feats: torch.Tensor (p*k, D)
        batch_ids: np.ndarray (p*k,)   # 원래 id 값 그대로 반환
    """
    assert predefined_feats.dim() == 2
    N = predefined_feats.size(0)
    assert predefined_ids.shape[0] == N

    rng = np.random.default_rng(seed)

    uniq_pids = np.unique(predefined_ids)
    if (not replace_pids) and (len(uniq_pids) < p):
        raise ValueError(f"Not enough unique pids ({len(uniq_pids)}) to sample p={p} without replacement.")

    chosen_pids = rng.choice(uniq_pids, size=p, replace=replace_pids)

    all_indices = []
    all_ids = []

    for pid in chosen_pids:
        pid_idxs = np.where(predefined_ids == pid)[0]
        if pid_idxs.size == 0:
            continue

        # pid에 속한 instance가 k개 미만이면 복원추출
        pick = rng.choice(pid_idxs, size=k, replace=(pid_idxs.size < k))
        all_indices.append(pick)
        all_ids.append(np.full(k, pid, dtype=predefined_ids.dtype))

    if len(all_indices) == 0:
        raise ValueError("Sampling failed: no indices selected. Check predefined_ids.")

    idx = np.concatenate(all_indices, axis=0)     # (p*k,)
    ids = np.concatenate(all_ids, axis=0)         # (p*k,)

    batch_feats = predefined_feats[idx]           # (p*k, D)  (CPU/GPU 모두 OK)

    return batch_feats, ids