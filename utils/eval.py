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

def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]  # distance 오름차순으로 gallery id 정렬
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]  # 각 쿼리 이미지에 대한 64장의 갤러리 id를 distance 오름차순 정렬 (64,)

        # cross cam evaluation을 위한 코드 같은데 내 세팅에선 안다루니 주석처리
        # remove = (result_i == query_ids[probe_index]) & (cam_locations_result[probe_index] == query_cam_ids[probe_index])
        # keep = np.invert(remove)
        # result_i = result_i[keep]

        result_i_unique = get_unique(result_i)

        # if match_counter.shape[0] != result_i_unique.shape[0]:
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

        # cross cam evaluation을 위한 코드 같은데 내 세팅에선 안다루니 주석처리
        # remove = (result_i == query_ids[probe_index]) & (cam_locations_result[probe_index] == query_cam_ids[probe_index])
        # keep = np.invert(remove)
        # result_i = result_i[keep]

        if query_ids[probe_index] not in result_i:
            # print(probe_index)
            # print(query_ids[probe_index])
            # print(result_i)
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
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)
        # dist_mat = -torch.mm(query_feats, gallery_feats.t())

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0] * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(r1, mAP))

    # return mAP, r1, r5, r10, r20
    return mAP, r1

def eval_regdb(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)
        # dist_mat = -torch.mm(query_feats, gallery_feats.t())

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0] * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(r1, mAP))

    return mAP, r1

def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)
        # dist_mat = -torch.mm(query_feats, gallery_feats.t())

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0] * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(r1, mAP))

    return mAP, r1

def eval_sysu_old(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=1, num_trials=10, rerank=False):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2
    query_feats = F.normalize(query_feats, dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)
    
    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        if rerank:
            dist_mat = re_ranking(query_feats, g_feat)
        else:
            dist_mat = pairwise_distance(query_feats, g_feat)
            # dist_mat = -torch.mm(query_feats, g_feat.permute(1,0))

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        # r5 += cmc[4]
        # r10 += cmc[9]
        # r20 += cmc[19]

    r1 = r1 / num_trials * 100
    # r5 = r5 / num_trials * 100
    # r10 = r10 / num_trials * 100
    # r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = '{} num-shot:{} r1 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(mode, num_shots, r1, mAP))

    return mAP, r1