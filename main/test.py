import os
import logging
import torch
import numpy as np
import scipy.io as sio

from models.baseline import Baseline
from models.IDKL.IDKL import Baseline as IDKL
from models.agw import embed_net as AGW
from data import get_test_loader
from engine.engine import create_eval_engine
from utils.eval import eval_llcm, eval_regdb, eval_sysu
from configs.default import strategy_cfg
from configs.default.dataset import dataset_cfg

from visualization.tsne import *


def test(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "inference.txt",
                        filemode="w")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                   root=cfg.data_root,
                                                   batch_size=64,
                                                   image_size=cfg.image_size,
                                                   num_workers=4)

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
        model = IDKL(num_classes=cfg.num_id,
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
        state_dict = ckpt.get("state_dict", ckpt)

        def strip_module_prefix(sd):
            if any(k.startswith("module.") for k in sd.keys()):
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        state_dict = strip_module_prefix(state_dict)

        # delete classifier (both weight and bias)
        filtered = {
            k: v for k, v in state_dict.items()
            if not (k == "classifier.weight" or k == "classifier.bias" or k.startswith("classifier.")
                    or k.startswith("classifier_sp.") or k.startswith("C_sp_f."))
        }

        msg = model.load_state_dict(filtered, strict=False)
        print("Missing keys:", msg.missing_keys)       # e.g., ['classifier.weight', 'classifier.bias']
        print("Unexpected keys:", msg.unexpected_keys) # hope to be empty list
    else:
        raise Exception("Verify checkpoint path for testing!")

    evaluator = create_eval_engine(model, non_blocking=True)
    rank=True

    torch.cuda.empty_cache()

    # extract query feature
    evaluator.run(query_loader)

    q_feats = torch.cat(evaluator.state.feat_list, dim=0)
    q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

    # extract gallery feature
    evaluator.run(gallery_loader)

    g_feats = torch.cat(evaluator.state.feat_list, dim=0)
    g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

    dataset = cfg.dataset
    # plot_tsne_by_id(q_feats, 
    #                 q_ids, 
    #                 max_ids=19,
    #                 # max_per_id=20,
    #                 seed=42, 
    #                 legend=False, 
    #                 save_path=f'{cfg.model}_{dataset}_IR_tsne.png')
    
    # plot_tsne_by_id(g_feats, 
    #                 g_ids, 
    #                 max_ids=19,
    #                 # max_per_id=20,
    #                 seed=42, 
    #                 legend=False, 
    #                 save_path=f'{cfg.model}_{dataset}_RGB_tsne.png')
    # plot_tsne_query_gallery(
    #     q_feats, q_ids,
    #     g_feats, g_ids,
    #     max_ids=19,
    #     # max_q_per_id=20,
    #     # max_g_per_id=20,
    #     normalize=True,
    #     save_path=f'{cfg.model}_{dataset}_RGB+IR_tsne.png'
    # )
    # exit()

    # id별 각 modal 대표 이미지 저장
    # ir_reps = select_representative_images(q_feats, q_ids, q_img_paths)
    # rgb_reps = select_representative_images(g_feats, g_ids, g_img_paths)

    # save_rep_dict(ir_reps, f"{dataset}_ir.txt")
    # save_rep_dict(rgb_reps, f"{dataset}_rgb.txt")

    # query(IR)과 gallery(RGB) tsne plot
    # gfeatures = collect_batch_features(gallery_loader, model, device)
    # qfeatures = collect_batch_features(query_loader, model, device)
    # gfeatures = collect_pid_features(gallery_loader, model, device)
    # qfeatures = collect_pid_features(query_loader, model, device)
    # tsne_batches(dataset, feature_batches=qfeatures)
    # tsne_batches_with_query(dataset, feature_batches=gfeatures,  query_features=qfeatures)

    if dataset == 'sysu':
        perm = sio.loadmat(os.path.join(dataset_cfg.data_root, 'exp', 'rand_perm_cam.mat'))[
            'rand_perm_cam']
        eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=rank)
        eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=rank)
        eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=rank)
        eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=rank)
    elif dataset == 'regdb':
        logging.info('infrared to visible')
        eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=cfg.rerank)
        logging.info('visible to infrared')
        eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=cfg.rerank)
    elif dataset == 'llcm':
        logging.info('infrared to visible')
        eval_llcm(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=rank)
        logging.info('visible to infrared')
        eval_llcm(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=rank)


    evaluator.state.feat_list.clear()
    evaluator.state.id_list.clear()
    evaluator.state.cam_list.clear()
    evaluator.state.img_path_list.clear()
    del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

    torch.cuda.empty_cache()

import torch.nn.functional as F
def select_representative_images(feats, ids, img_paths, normalize=True):
    feats = feats.detach()
    if feats.is_cuda:
        feats = feats.cpu()
    if normalize:
        feats = F.normalize(feats)

    ids = np.asarray(ids)
    img_paths = np.asarray(img_paths)

    unique_ids = np.unique(ids)
    rep_dict = {}

    for pid in unique_ids:
        idxs = np.where(ids == pid)[0]
        pid_feats = feats[idxs]
        pid_paths = img_paths[idxs]

        mu = pid_feats.mean(dim=0, keepdim=True)
        dists = torch.norm(pid_feats - mu, p=2, dim=1)

        best_idx_local = torch.argmin(dists).item()
        best_idx_global = idxs[best_idx_local]

        rep_dict[int(pid)] = pid_paths[best_idx_local]
    return rep_dict

def save_rep_dict(rep_dict, save_path):
    with open(save_path, "w") as f:
        for pid, path in sorted(rep_dict.items()):
            f.write(f"{pid} {path}\n")


if __name__ == "__main__":
    import argparse
    import random
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', type=str, default='Baseline', choices=['Baseline', 'IDKL'], help='Model to TTA')
    parser.add_argument('--resume', type=str, default='', help='model checkpoint path')
    
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

    cfg.freeze()

    test(cfg)