import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

from typing import Literal
from torch.amp import autocast, GradScaler

from models.baseline import Baseline
from models.IDKL.IDKL import Baseline as IDKL_Baseline

from data import get_test_loader, get_tta_test_loader
from .data import make_data_loader_TTA
from .tta_evaluator import TTA_Evaulator
from .tta_learner import TTA_Learner
from .make_loss_tta import make_loss_tta
from .make_optimizer import make_optimizer
from engine.engine import create_eval_engine
from utils.eval_modality import eval_llcm, eval_regdb, eval_sysu
from configs.default import strategy_cfg
from configs.default.dataset import dataset_cfg

from loss.entropy import Entropy
from visualization.tsne import *

from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import huepy as hue

def extract_camera_normed_query_feats(query_init_loader, model):
    query_data, query_upids, query_feats, query_gt_pids, query_original_pids, query_cam_ids, query_img_paths = [], [], [], [], [], [],[]
    model.eval()
    for i,(data, pids, upids, camid, original_pid, img_path) in enumerate(tqdm(query_init_loader,  desc="Query feature extraction")):
        # print(pids)
        # print()
        # print(upids)
        # print()
        # print(camid)
        # print()
        # print(original_pid)
        # print()
        # print(img_path)
        # exit()
        # vutils.save_image(data, './vis/cuhk02_target/multi-cam-query/{}_imgs.jpg'.format(i), nrow=8, padding=2, normalize=True, range=(0, 1))
        data = data.cuda()
        with torch.no_grad():
            feat = model(data)

        query_data.append(data.cpu())
        query_feats.append(feat)
        query_gt_pids.extend(list(pids))
        query_upids.extend(list(upids))
        query_cam_ids.extend(list(camid))
        # 왜 둘이 뒤바뀜?;;
        query_img_paths.extend(list(original_pid))
        query_original_pids.extend(list(img_path))

    # query_feats = F.normalize(torch.cat(query_feats), dim=1)
    query_feats = torch.cat(query_feats)

    print("query_feats == ",query_feats.shape)
    ######### Estimate Per Camera mean and standard deviation ############
    unique_camids = torch.unique(torch.tensor(query_cam_ids))
    print("unique_camids == ",unique_camids)
    
    cam_mean = dict()
    cam_std = dict()

    eps=1e-6
    for i, cam_id in enumerate(unique_camids):
        
        idxs = (cam_id==torch.tensor(query_cam_ids))
        feats_c = query_feats[idxs]                  # (Nc, D)
        m = feats_c.mean(dim=0)
        s = feats_c.var(dim=0, unbiased=False).sqrt().clamp_min(eps)
        
        cam_mean[cam_id.item()] = m
        cam_std[cam_id.item()] = s


    if True:  # cfg.FEAT_NORMED
        for i, (q_feat, q_camid) in enumerate(zip(query_feats, query_cam_ids)):
            camid = q_camid.item()
            q_feat = (q_feat-cam_mean[camid])/cam_std[camid]
            query_feats[i] = q_feat
        
    # 1. concatenate query_feats and convert query_gt_pids to tensor
    query_gt_pids = torch.tensor(query_gt_pids)               # (N,)

    unique_pids = torch.unique(query_gt_pids)
    # 2. pid → mean feature mapping
    mean_feats = []
    for pid in unique_pids:
        mask = (query_gt_pids == pid)
        mean_feat = query_feats[mask].mean(dim=0)
        mean_feats.append(mean_feat)

    mean_feats = torch.stack(mean_feats, dim=0)  # (num_classes, feat_dim)
    mean_feats = F.normalize(mean_feats, dim=1)  # normalize if needed

    return query_data, query_feats, query_gt_pids, query_cam_ids, query_original_pids, query_img_paths, mean_feats, query_upids, cam_mean, cam_std



def do_online_tta(cfg, model, gallery_loader, query_init_loader, num_query_tta, logger):
    model.eval()

    query_data, query_feats, query_gt_pids, query_cam_ids, query_original_pids, query_img_paths, mean_feats, query_upids, cam_mean, cam_std = extract_camera_normed_query_feats(query_init_loader, model)

    dataset_name = cfg.dataset

    total_map, total_rank1, total_fmr1, total_tnr, total_fpr = [], [], [], [], []
    total_same_cam_fpr, total_same_cam_tnr = [],[]
    total_auroc, total_eer_threshold, total_eer, total_tpr_at_fpr = [], [], [], []

    tta_evaluator = TTA_Evaulator()
    current_stream_loader = gallery_loader

    # current_query_feats = deepcopy(query_feats)
    current_query_feats = query_feats

    # current_model = deepcopy(model)
    current_model = model
    current_model.classifier = nn.Linear(2048, num_query_tta, bias=False).cuda()
    # current_model.classifier.weight.data = deepcopy(mean_feats)
    current_model.classifier.weight.data = mean_feats

    optimizer = make_optimizer(cfg, current_model)
    

    tta_learner = TTA_Learner(cfg, current_model, optimizer, num_query_tta)
    
    tta_learner.model = current_model
    tta_learner.evaluator = tta_evaluator
    
    tta_learner.mem.query_original_pids = query_original_pids
    tta_learner.mem.query_cam_ids = query_cam_ids
    tta_learner.query_cam_ids = query_cam_ids
    tta_learner.query_gt_pids = query_gt_pids
    
    current_stream_len = int(len(current_stream_loader))
    print("current_stream_len == ",current_stream_len)
    tta_learner.total_iter = current_stream_len
    # tta_learner.cam_id = cam_id
    
    tta_learner.loss_supervised = make_loss_tta(cfg,num_query_tta)

    # coreset_trainloader = tta_learner.set_query_memory(cfg, deepcopy(query_data), deepcopy(F.normalize(query_feats,dim=1)), deepcopy(query_gt_pids), deepcopy(query_cam_ids), deepcopy(query_img_paths), deepcopy(query_upids))
    coreset_trainloader = tta_learner.set_query_memory(cfg, query_data, F.normalize(query_feats,dim=1), query_gt_pids, query_cam_ids, query_img_paths, query_upids)
    
    IR_metrics = np.zeros(2, dtype=np.float16)
    RI_metrics = np.zeros(2, dtype=np.float16)
    II_metrics = np.zeros(2, dtype=np.float16)
    RR_metrics = np.zeros(2, dtype=np.float16)
    Avg_metrics = np.zeros(2, dtype=np.float16)
    
    for i, batch in enumerate(current_stream_loader):
        torch.cuda.empty_cache()
        data, pids, _, camids, img_path, original_pid  = batch
        
        data = data.cuda()
        batch_data = (data, camids, original_pid, query_original_pids, img_path)

        query_aug_feats2, query_aug_pids2 = None, None
        
        pred_feat = tta_learner(i, batch_data, current_query_feats, query_cam_ids, query_aug_feats2, query_aug_pids2)

        eval_query_feats = F.normalize(current_query_feats, dim=1, eps=1e-6)

        rerank = True
        if cfg.dataset == 'llcm':
            query_original_pids = np.asarray(query_original_pids)
            query_cam_ids = np.asarray(query_cam_ids)
            pids = np.asarray(pids)
            camids = np.asarray(camids)

            # 1. mask 생성
            valid_query_mask = np.isin(query_original_pids, pids)
            # 2. query 정보 필터링
            eval_query_feats_eval = eval_query_feats[valid_query_mask]
            query_original_pids_eval = query_original_pids[valid_query_mask]
            query_cam_ids_eval = query_cam_ids[valid_query_mask]
        
            IR, RI, II, RR, Avg = eval_llcm(
                eval_query_feats_eval, query_original_pids_eval, query_cam_ids_eval,
                pred_feat, pids, camids, np.array(img_path),
                rerank=rerank
            )

            IR_metrics += np.array([IR[0], IR[1]], dtype=np.float16)
            RI_metrics += np.array([RI[0], RI[1]], dtype=np.float16)
            II_metrics += np.array([II[0], II[1]], dtype=np.float16)
            RR_metrics += np.array([RR[0], RR[1]], dtype=np.float16)
            Avg_metrics += np.array([Avg[0], Avg[1]], dtype=np.float16)

        else:
            raise ValueError("데이터셋이 LLCM이 아님")
        
    IR_final = IR_metrics / current_stream_len
    RI_final = RI_metrics / current_stream_len
    II_final = II_metrics / current_stream_len
    RR_final = RR_metrics / current_stream_len
    Avg_final = Avg_metrics / current_stream_len

    logging.info(f"[IR to RGB FINAL]  r1={IR_final[0]:.2f}  mAP={IR_final[1]:.2f}")
    logging.info(f"[RGB to IR FINAL]  r1={RI_final[0]:.2f}  mAP={RI_final[1]:.2f}")
    logging.info(f"[IR to IR FINAL]  r1={II_final[0]:.2f}  mAP={II_final[1]:.2f}")
    logging.info(f"[RGB to RGB FINAL]  r1={RR_final[0]:.2f}  mAP={RR_final[1]:.2f}")
    logging.info(f"[Avg FINAL]  r1={Avg_final[0]:.2f}  mAP={Avg_final[1]:.2f}")
        
    #     tta_evaluator.eval_galllery_stream2query_set(eval_query_feats, pred_feat, query_original_pids, original_pid, query_cam_ids, camids)
    #     tta_evaluator.compute_false_matching_ratio(eval_query_feats, query_original_pids, pred_feat, original_pid,  query_cam_ids, camids, cross_cam_threshold=cfg.MATCHING_THRES, same_cam_threshold=0.6)

    # all_cmc = np.asarray(tta_evaluator.rank1_list).astype(np.float32)
    # all_cmc = all_cmc.sum(0) / tta_evaluator.valid_num_query
    # mAP = np.mean(tta_evaluator.ap_list)

    # logger.info('Validation Results')
    # print("valid_num_query == ",tta_evaluator.valid_num_query)
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc[r - 1]))

    # logger.info("Mean Rank-1 similarity (query → cross-cam query): {:.4f}".format(
    # np.mean(tta_evaluator.rank1_similarities_cross_cam_query)))

    # logger.info("Mean Rank-1 similarity (query → sam-cam query): {:.4f}".format(
    # np.mean(tta_evaluator.rank1_similarities_same_cam_query)))


    # cross_cam_tnr = tta_evaluator.true_neg_cnts / tta_evaluator.valid_num_query
    # print("valid_num_query == ",tta_evaluator.valid_num_query)
    # logger.info("Cross-cam True Negative ratio: {:.2%}".format(cross_cam_tnr))

    # print("valid_num_non_query == ",tta_evaluator.valid_num_non_query)
    # cross_cam_fpr = tta_evaluator.false_pos_cnts / tta_evaluator.valid_num_non_query
    # logger.info("Cross-cam False Positive ratio: {:.2%}".format(cross_cam_fpr))
    
    
    # same_cam_tnr = tta_evaluator.same_cam_true_neg_cnts / tta_evaluator.valid_num_query
    # # print("valid_num_query == ",tta_evaluator.valid_num_query)
    # logger.info("Same-cam True Negative ratio: {:.2%}".format(same_cam_tnr))

    # # print("valid_num_non_query == ",tta_evaluator.valid_num_non_query)
    # same_cam_fpr = tta_evaluator.same_cam_false_pos_cnts / tta_evaluator.valid_num_non_query
    # logger.info("Same-cam False Positive ratio: {:.2%}".format(same_cam_fpr))

    # # ROC Curve
    # fpr, tpr, thresholds = roc_curve(tta_evaluator.all_labels, tta_evaluator.all_sims)
    
    # # AUROC
    # auroc = auc(fpr, tpr)
    
    # # EER
    # fnr = 1 - tpr
    # eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    # eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    # # TPR@FPR=1%
    # idx = np.where(fpr <= 0.01)[0]
    # tpr_at_fpr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    
    
    # total_auroc.append(auroc)
    # total_eer_threshold.append(eer_threshold)
    # total_eer.append(eer)
    # total_tpr_at_fpr.append(tpr_at_fpr)
    
    
    # total_map.append(mAP)
    # total_rank1.append(all_cmc[0])
    # total_fmr1.append(np.mean(tta_evaluator.rank1_similarities_non_query))
    
    # total_tnr.append(cross_cam_tnr)
    # total_fpr.append(cross_cam_fpr)
    
    # total_same_cam_tnr.append(same_cam_tnr)
    # total_same_cam_fpr.append(same_cam_fpr)


    # print(hue.info(hue.bold(hue.lightgreen('All Camera Stream End!!!'))))
    # logger.info('Final Validation Results')
    # logger.info("mAP: {:.1%}".format(np.mean(total_map)))
    # logger.info("Rank-{:<3}:{:.1%}".format(1, np.mean(total_rank1)))
 
    # logger.info("Cross-cam TNR: {:.2%}".format(np.mean(total_tnr)))
    # logger.info("Cross-cam FPR: {:.2%}".format(np.mean(total_fpr)))
    
    # logger.info("Same-cam TNR: {:.2%}".format(np.mean(total_same_cam_tnr)))
    # logger.info("Same-cam FPR: {:.2%}".format(np.mean(total_same_cam_fpr)))
    
    # logger.info("Cross-cam auroc: {:.2%}".format(np.mean(total_auroc)))
    # logger.info("Cross-cam eer_threshold: {:.2%}".format(np.mean(total_eer_threshold)))
    # logger.info("Cross-cam eer: {:.2%}".format(np.mean(total_eer)))
    # logger.info("Cross-cam tpr_at_fpr: {:.2%}".format(np.mean(total_tpr_at_fpr)))

def test(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "TTA_PaTTA.txt",
                        filemode="w")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # gallery_loader, query_loader = get_tta_test_loader(dataset=cfg.dataset,
    #                                                    root=cfg.data_root,
    #                                                    query_batch_size=64,
    #                                                    gallery_batch_size=64,
    #                                                    image_size=cfg.image_size,
    #                                                    num_workers=4,
    #                                                    query_set=cfg.query_set
    #                                                    )
    
    gallery_loader, num_classes, query_loader, num_query_classes,  num_query_cams = make_data_loader_TTA(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       query_batch_size=64,
                                                       gallery_batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4,
                                                       query_set=cfg.query_set
                                                       )
    
    if cfg.model == 'Baseline':
        model = Baseline(num_classes=cfg.num_id,
                        drop_last_stride=cfg.drop_last_stride,
                        triplet=cfg.triplet,
                        k_size=cfg.k_size,
                        center_cluster=cfg.center_cluster,
                        center=cfg.center,
                        margin=cfg.margin,
                        num_parts=cfg.num_parts,
                        classification=cfg.classification,)
        
    elif cfg.model == 'IDKL':
        model = IDKL_Baseline(num_classes=cfg.num_id,
                              pattern_attention=cfg.pattern_attention,
                              modality_attention=cfg.modality_attention,
                            mutual_learning=cfg.mutual_learning,
                            decompose=cfg.decompose,
                            drop_last_stride=cfg.drop_last_stride,
                            triplet=cfg.triplet,
                            k_size=cfg.k_size,
                            center_cluster=cfg.center_cluster,
                            center=cfg.center,
                            margin=cfg.margin,
                            num_parts=cfg.num_parts,
                            weight_KL=cfg.weight_KL,
                            weight_sid=cfg.weight_sid,
                            weight_sep=cfg.weight_sep,
                            update_rate=cfg.update_rate,
                            classification=cfg.classification,
                            bg_kl=cfg.bg_kl,
                            sm_kl=cfg.sm_kl,
                            fb_dt=cfg.fb_dt,
                            IP=cfg.IP,
                            distalign=cfg.distalign)
    
    model.to(device)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu")
        # ckpt가 {"state_dict": ...} 구조일 수도, 그냥 state_dict일 수도 있음
        state_dict = ckpt.get("state_dict", ckpt)

        def strip_module_prefix(sd):
            # 모델이 DataParallel/DistributedDataParallel로 저장된 경우 'module.' 제거
            if any(k.startswith("module.") for k in sd.keys()):
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        state_dict = strip_module_prefix(state_dict)

        # classifier만 제거 (weight, bias 모두)
        filtered = {
            k: v for k, v in state_dict.items()
            if not (k == "classifier.weight" or k == "classifier.bias" or k.startswith("classifier.")
                    or k.startswith("classifier_sp.") or k.startswith("C_sp_f."))
        }

        msg = model.load_state_dict(filtered, strict=False)
        print("Missing keys:", msg.missing_keys)       # 예: ['classifier.weight', 'classifier.bias']
        print("Unexpected keys:", msg.unexpected_keys) # 보통 빈 리스트
    else:
        raise Exception("Verify checkpoint path for testing!")
    
    torch.cuda.empty_cache()

    do_online_tta(cfg, model, gallery_loader, query_loader, num_query_classes, logger)

    # evaluator = create_eval_engine(model, non_blocking=True)
    # rank=True


    # dataset = cfg.dataset
    # evaluator.run(query_loader)

    # predefined_feats = torch.cat(evaluator.state.feat_list, dim=0)
    # predefined_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    # predefined_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    # predefined_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

if __name__ == "__main__":
    import argparse
    import random
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', type=str, default='Baseline', choices=['Baseline', 'IDKL'], help='Model to TTA')
    parser.add_argument('--resume', type=str, default='', help='model checkpoint path')
    parser.add_argument('--query-set', type=str, choices=['one', 'few', 'multi'], default='multi', help='query set configuration')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set random seed
    seed = 1
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # enable cudnn backend
    torch.backends.cudnn.benchmark = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    cfg['model'] = args.model
    cfg['resume'] = args.resume
    cfg['query_set'] = args.query_set

    cfg.freeze()

    test(cfg)