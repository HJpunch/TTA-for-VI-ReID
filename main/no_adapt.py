import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

from models.baseline import Baseline
from data import get_test_loader, get_tta_test_loader
from engine.engine import create_eval_engine
from utils.eval import eval_llcm, eval_regdb, eval_sysu
from configs.default import strategy_cfg
from configs.default.dataset import dataset_cfg

@torch.no_grad()
def online_eval_by_batches(dataset: str,
                           model: torch.nn.Module,
                           query_feats: torch.Tensor,
                           query_ids_np: np.ndarray,
                           query_cams_np: np.ndarray,
                           gallery_loader,
                           rerank: bool = False,
                           perm=None,                # SYSU에서 필요
                           sysu_mode='all',          # 'all' or 'indoor'
                           sysu_num_shots=1,         # 1 or 10 등
                           device='cuda'):
    """
    갤러리 '배치별'로 eval_*을 호출해 (mAP, r1, r5, r10, r20) 를 얻고,
    '해당 배치 갤러리에 존재하는 쿼리 ID'만 골라 평가한 뒤,
    유효 쿼리 수로 가중 평균한 최종 지표를 반환.
    """
    model.eval()

    sum_metrics = np.zeros(2, dtype=np.float64)  # mAP, r1, r5, r10, r20 가중합
    total_valid_queries = 0

    for bi, batch in enumerate(gallery_loader):
        imgs, g_ids_t, g_cams_t, g_paths, _ = batch  # (B, ...)

        # 1) 해당 배치 갤러리에 '존재하는' 쿼리만 선별
        g_ids_np = g_ids_t.numpy()
        valid_mask = np.isin(query_ids_np, g_ids_np)
        vq = int(valid_mask.sum())
        if vq == 0:
            # 이 배치에는 일치 가능한 쿼리가 없음 → 스킵
            continue

        # qf_b = F.normalize(query_feats[valid_mask].cpu(), dim=1)           # (VQ, D) torch
        # qid_b  = query_ids_np[valid_mask]           # (VQ,) np
        # qcam_b = query_cams_np[valid_mask]          # (VQ,) np
        # 배치 갤러리에 존재하는 쿼리만 선택
        g_ids_np  = g_ids_t.cpu().numpy()
        g_cams_np = g_cams_t.cpu().numpy()

        pid_eq  = (query_ids_np[:, None] == g_ids_np[None, :])
        cam_ne  = (query_cams_np[:, None] != g_cams_np[None, :])
        valid_mask = (pid_eq & cam_ne).any(axis=1)

        vq = int(valid_mask.sum())
        if vq == 0:
            continue

        qf_b   = F.normalize(query_feats[valid_mask].cpu(), dim=1)
        qid_b  = query_ids_np[valid_mask]
        qcam_b = query_cams_np[valid_mask]

        # 2) 배치 갤러리 임베딩 추출
        imgs = imgs.to(device, non_blocking=True)
        g_feats_b = model(imgs)
        if isinstance(g_feats_b, tuple):
            g_feats_b = g_feats_b[0]
        g_feats_b = F.normalize(g_feats_b.cpu(), dim=1)

        # 3) 데이터셋별 eval_* 호출 (리턴: mAP, r1, r5, r10, r20)
        if dataset == 'sysu':
            if perm is None:
                raise RuntimeError("SYSU 평가에는 perm(랜덤 카메라 permut) 행렬이 필요합니다.")
            mAP, r1 = eval_sysu(
                qf_b, qid_b, qcam_b,
                g_feats_b, g_ids_t.numpy(), g_cams_t.numpy(), np.array(g_paths),
                perm, mode=sysu_mode, num_shots=sysu_num_shots, rerank=rerank
            )
        elif dataset == 'regdb':
            mAP, r1 = eval_regdb(
                qf_b, qid_b, qcam_b,
                g_feats_b, g_ids_t.numpy(), g_cams_t.numpy(), np.array(g_paths),
                rerank=rerank
            )
        elif dataset == 'llcm':
            mAP, r1 = eval_llcm(
                qf_b, qid_b, qcam_b,
                g_feats_b, g_ids_t.numpy(), g_cams_t.numpy(), np.array(g_paths),
                rerank=rerank
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # 4) 유효 쿼리 수로 가중합
        if mAP == 0.0 or r1 == 0.0:
            continue
        sum_metrics += np.array([mAP, r1], dtype=np.float64) * vq
        total_valid_queries += vq

    if total_valid_queries == 0:
        # 전 배치에서 유효 쿼리가 없었다면 0 반환
        return dict(mAP=0.0, r1=0.0, valid_q=0)

    final = sum_metrics / total_valid_queries
    return dict(mAP=float(final[0]), r1=float(final[1]), valid_q=total_valid_queries)


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

    gallery_loader, query_loader = get_tta_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)
    
    model = Baseline(num_classes=cfg.num_id,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     k_size=cfg.k_size,
                     center_cluster=cfg.center_cluster,
                     center=cfg.center,
                     margin=cfg.margin,
                     num_parts=cfg.num_parts,
                     classification=cfg.classification,)
    
    model.to(device)
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        def strip_module_prefix(sd):
            if any(k.startswith("module.") for k in sd.keys()):
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        state_dict = strip_module_prefix(state_dict)

        # classifier만 제거 (weight, bias 모두)
        filtered = {
            k: v for k, v in state_dict.items()
            if not (k == "classifier.weight" or k == "classifier.bias" or k.startswith("classifier."))
        }

        msg = model.load_state_dict(filtered, strict=False)
        print("Missing keys:", msg.missing_keys)       # 예: ['classifier.weight', 'classifier.bias']
        print("Unexpected keys:", msg.unexpected_keys) # 보통 빈 리스트
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

    dataset = cfg.dataset
    results = None
    if dataset == 'sysu':
        perm = sio.loadmat(os.path.join(cfg.data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']

        # 예: 'all', num_shots=1 기준
        results = online_eval_by_batches(
            dataset='sysu',
            model=model,
            query_feats=q_feats, query_ids_np=q_ids, query_cams_np=q_cams,
            gallery_loader=gallery_loader,
            rerank=rank,
            perm=perm,
            sysu_mode='all',
            sysu_num_shots=10,
            device=device
        )

    elif dataset == 'regdb':
        results = online_eval_by_batches(
            dataset='regdb',
            model=model,
            query_feats=q_feats, query_ids_np=q_ids, query_cams_np=q_cams,
            gallery_loader=gallery_loader,
            rerank=cfg.rerank,
            device=device
        )
    elif dataset == 'llcm':
        results = online_eval_by_batches(
            dataset='llcm',
            model=model,
            query_feats=q_feats, query_ids_np=q_ids, query_cams_np=q_cams,
            gallery_loader=gallery_loader,
            rerank=rank,
            device=device
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logging.info(f"[ONLINE-FINAL] valid_q={results['valid_q']}  "
             f"r1={results['r1']:.2f}  mAP={results['mAP']:.2f}")

    # # extract gallery feature
    # evaluator.run(gallery_loader)

    # g_feats = torch.cat(evaluator.state.feat_list, dim=0)
    # g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    # g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    # g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

    # dataset = cfg.dataset

    # if dataset == 'sysu':
    #     perm = sio.loadmat(os.path.join(dataset_cfg.data_root, 'exp', 'rand_perm_cam.mat'))[
    #         'rand_perm_cam']
    #     eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=rank)
    #     eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=rank)
    #     eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=rank)
    #     eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=rank)
    # elif dataset == 'regdb':
    #     logging.info('infrared to visible')
    #     eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=cfg.rerank)
    #     logging.info('visible to infrared')
    #     eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=cfg.rerank)
    # elif dataset == 'llcm':
    #     logging.info('infrared to visible')
    #     eval_llcm(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=rank)
    #     logging.info('visible to infrared')
    #     eval_llcm(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=rank)


    # evaluator.state.feat_list.clear()
    # evaluator.state.id_list.clear()
    # evaluator.state.cam_list.clear()
    # evaluator.state.img_path_list.clear()
    # del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

    # torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    import random
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
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

    cfg['resume'] = args.resume

    cfg.freeze()

    test(cfg)