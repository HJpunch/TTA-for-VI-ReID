import os
import logging
import numpy as np

from torch.nn import functional as F

from .rerank import re_ranking, pairwise_distance


def parse_modality_within_batch(dataset, paths):
    assert dataset in ['sysu', 'regdb', 'llcm']

    if dataset == 'llcm':
        vis_idx = [i for i, p in enumerate(paths) if 'vis' in p]
        nir_idx = [i for i, p in enumerate(paths) if 'nir' in p]
    # TODO: SYSU랑 RegDB에 대해서도 일케 하기
    
    return vis_idx, nir_idx


def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])

    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]

def get_cmc_case(sorted_indices,
                 query_ids, query_cam_ids,
                 gallery_ids, gallery_cam_ids,
                 query_cam_list,
                 gallery_cam_list):

    query_cam_mask   = np.isin(query_cam_ids, list(query_cam_list))
    gallery_cam_mask = np.isin(gallery_cam_ids, list(gallery_cam_list))

    gallery_unique_count = get_unique(gallery_ids[gallery_cam_mask]).shape[0]
    cmc_counter = np.zeros((gallery_unique_count,), dtype=np.float32)
    valid_probe = 0

    for i in range(sorted_indices.shape[0]):
        if not query_cam_mask[i]:
            continue

        gt_mask = (gallery_ids == query_ids[i]) & gallery_cam_mask
        if np.sum(gt_mask) == 0:
            continue

        filtered_rank = [idx for idx in sorted_indices[i] if gallery_cam_mask[idx]]
        if len(filtered_rank) == 0:
            continue

        valid_probe += 1

        result_ids = gallery_ids[filtered_rank]
        result_ids_unique = get_unique(result_ids)

        hit = np.where(result_ids_unique == query_ids[i])[0]
        if len(hit) == 0:
            continue

        first = hit[0]
        cmc_counter[first:] += 1.0

    cmc = cmc_counter / max(valid_probe, 1)
    return cmc, valid_probe

def get_mAP_case(sorted_indices,
                 query_ids, query_cam_ids,
                 gallery_ids, gallery_cam_ids,
                 query_cam_list,
                 gallery_cam_list):
    avg_precision_sum = 0.0
    valid_probe = 0

    query_cam_mask = np.isin(query_cam_ids, list(query_cam_list))
    gallery_cam_mask = np.isin(gallery_cam_ids, list(gallery_cam_list))

    for i in range(sorted_indices.shape[0]):
        # query camera condition
        if not query_cam_mask[i]:
            continue

        # GT existence check
        gt_mask = (gallery_ids == query_ids[i]) & gallery_cam_mask
        true_match_count = np.sum(gt_mask)

        if true_match_count == 0:
            continue

        valid_probe += 1

        # filter ranking by gallery camera
        filtered_rank = [
            idx for idx in sorted_indices[i]
            if gallery_cam_mask[idx]
        ]

        if len(filtered_rank) == 0:
            continue

        results_ids = gallery_ids[filtered_rank]

        # AP calculation
        match = (results_ids == query_ids[i])
        match_idx = np.where(match)[0]

        # precision@k for each GT
        precision_at_k = np.arange(1, len(match_idx) + 1) / (match_idx + 1)
        ap = np.mean(precision_at_k)

        avg_precision_sum += ap

    mAP = avg_precision_sum / max(valid_probe, 1)
    return mAP, valid_probe

def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]  # distance 오름차순으로 gallery id 정렬
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]  # 각 쿼리 이미지에 대한 64장의 갤러리 id를 distance 오름차순 정렬 (64,)
        result_i_unique = get_unique(result_i)

        if query_ids[probe_index] not in result_i_unique:
            raise Exception("non query 발생!!!!!")

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]

        if query_ids[probe_index] not in result_i:
            raise Exception("non query 발생!!!!!")

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def eval_llcm(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)

    sorted_indices = np.argsort(dist_mat, axis=1)

    # IR to RGB
    IR_mAP, IR_Nm = get_mAP_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[3], gallery_cam_list=[2])
    IR_cmc, IR_Nc = get_cmc_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[3], gallery_cam_list=[2])

    # RGB to IR
    RI_mAP, RI_Nm = get_mAP_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[2], gallery_cam_list=[3])
    RI_cmc, RI_Nc = get_cmc_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[2], gallery_cam_list=[3])
    # IR to IR
    II_mAP, II_Nm = get_mAP_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[3], gallery_cam_list=[3])
    II_cmc, II_Nc = get_cmc_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[3], gallery_cam_list=[3])
    # RGB to RGB
    RR_mAP, RR_Nm = get_mAP_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[2], gallery_cam_list=[2])
    RR_cmc, RR_Nc = get_cmc_case(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, query_cam_list=[2], gallery_cam_list=[2])
    # average
    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    IR_r1 = IR_cmc[0] * 100
    IR_mAP = IR_mAP * 100

    RI_r1 = RI_cmc[0] * 100
    RI_mAP = RI_mAP * 100

    II_r1 = II_cmc[0] * 100
    II_mAP = II_mAP * 100

    RR_r1 = RR_cmc[0] * 100
    RR_mAP = RR_mAP * 100

    r1 = cmc[0] * 100
    mAP = mAP * 100

    # logging.info(f"[IR to RGB] r1 precision = {IR_r1:.2f} (N={IR_Nc}), mAP = {IR_mAP:.2f} (N={IR_Nm})")
    # logging.info(f"[RGB to IR] r1 precision = {RI_r1:.2f} (N={RI_Nc}), mAP = {RI_mAP:.2f} (N={RI_Nm})")
    # logging.info(f"[IR to IR] r1 precision = {II_r1:.2f} (N={II_Nc}), mAP = {II_mAP:.2f} (N={II_Nm})")
    # logging.info(f"[RGB to RGB] r1 precision = {RR_r1:.2f} (N={RR_Nc}), mAP = {RR_mAP:.2f} (N={RR_Nm})")
    # logging.info(f"[All to All] r1 precision = {r1:.2f}, mAP = {mAP:.2f}")

    return (IR_r1, IR_mAP), (RI_r1, RI_mAP), (II_r1, II_mAP), (RR_r1, RR_mAP), (r1, mAP)

def eval_regdb(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0] * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f}, mAP = {:.2f}'
    # logging.info(perf.format(r1, mAP))

    return mAP, r1

def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0] * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f}, mAP = {:.2f}'
    # logging.info(perf.format(r1, mAP))

    return mAP, r1