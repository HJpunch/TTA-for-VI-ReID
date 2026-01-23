import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

from torch.amp import autocast, GradScaler

from models.baseline import Baseline
from models.IDKL.IDKL import Baseline as IDKL
from models.agw import embed_net as AGW

from data import get_edge_device_test_loader
from engine.engine import create_eval_engine
from utils.eval import eval_llcm, eval_sysu
from configs.default import strategy_cfg
from configs.default.dataset import dataset_cfg

from loss.entropy import Entropy
from visualization.tsne import *


from .sampling import assign_gallery_pseudo_label, sample_pids_k_instances
import torch.nn as nn
from models.IDKL.layers.loss.triplet_loss import TripletLoss


def adapt_module(
        dataset: str,
        model: torch.nn.Module,
        predefined_feats: torch.Tensor,
        predefined_ids: np.ndarray,
        predefined_cams: np.ndarray,
        predefined_paths: np.ndarray,
        predefined_num_classes,
        online_loader,
        *,
        rerank: bool = False,
        perm=None,                 # SYSU only
        sysu_mode='all',
        sysu_num_shots=10,
        device='cuda',
        # TTA 설정
        update='bn',               # 'bn' | 'all'
        lr=1e-4,
        weight_decay=0.0,
        amp=True,
        metric='cosine',
        lam_fgt: float = 0.0,           # >0이면 원본 파라미터로 L2 앵커
        loss_fn=Entropy,
        top_k=0,
    ):
    """각 갤러리 배치에 대해: (1) 적응 전 모델로 평가 → (2) 같은 배치로 엔트로피 TTA 1 step"""
    assert loss_fn is not None, "loss function should be defined."

    model.to(device)
    model.eval()  # 평가 모드 고정(파라미터 업데이트만 수행)
    model.classifier = nn.Linear(2048, predefined_num_classes+32, bias=False).to(device)  # 32는 gallery pseudo_label 용

    cross_entropy_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    triplet_loss_function = TripletLoss(margin=0.3)

    # 업데이트 파라미터 선택
    def _bn_layers(net):
        from torch.nn.modules.batchnorm import _BatchNorm
        bns = []
        def _walk(m):
            if isinstance(m, _BatchNorm):
                bns.append(m)
            for c in m.children():
                _walk(c)
        _walk(net)
        return bns
    
    def param_l2_loss(model, initial_param):
        loss = torch.tensor(0).float().to(device)
        for name, param in model.named_parameters():
            loss += (param - initial_param[name].to(device)).square().sum()
        return loss

    if update == 'bn':
        params = (p for m in _bn_layers(model) for p in m.parameters() if p.requires_grad)
    else:
        params = (p for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)
    ent_loss = loss_fn(metric=metric, on_gpu=('cuda' in device), top_k=top_k)

    # 원본 파라미터 스냅샷(드리프트 억제 옵션)
    if lam_fgt > 0.0:
        initial_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    sum_metrics = np.zeros(2, dtype=np.float64)  # [mAP, r1] (스케일 [0,1])

    for bi, batch in enumerate(online_loader):
        model.zero_grad()
        imgs, ids, cams, paths, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        gallery_feats = model(imgs)

        pid_eq  = (predefined_ids[:, None] == ids[None, :])
        valid_mask = (pid_eq).any(axis=1)
        vq = int(valid_mask.sum())
        if vq == 0:
            continue

        # query+gallery memory 운용
        B = imgs.size(0)
        gallery_ids  = ids.cpu().numpy()
        gallery_cams = cams.cpu().numpy()
        gallery_paths = paths

        pseudo_labels = assign_gallery_pseudo_label(F.normalize(gallery_feats[B//2:], dim=1), threshold=0.85)
        pseudo_labels += predefined_num_classes

        sampled_query_feats, sampled_query_ids = sample_pids_k_instances(predefined_feats, predefined_ids, p=8, k=4)
        sampled_query_feats = sampled_query_feats.to(device, non_blocking=True)
        sampled_query_ids = torch.from_numpy(sampled_query_ids).to(device, non_blocking=True)

        adapt_batch = torch.cat((gallery_feats[B//2:], sampled_query_feats), dim=0)
        adapt_label = torch.cat((pseudo_labels, sampled_query_ids), dim=0)

        triplet_loss = 0.0
        loss = 0.0

        # ---------- (1) 적응 ----------
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp, device_type='cuda' if 'cuda' in device else 'cpu'):
            u, c = pseudo_labels.unique(return_counts=True)
            if u.numel() < 2:
                triplet_loss = 0.0 * gallery_feats.sum()
            
            else:
                triplet_loss, *_ = triplet_loss_function(adapt_batch, adapt_label)
        loss = triplet_loss

        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()


        # ---------- (2) 평가 ----------
        # query_feats   = F.normalize(predefined_feats[valid_mask], dim=1)
        query_feats = predefined_feats[valid_mask]
        query_ids  = predefined_ids[valid_mask]
        query_cams = predefined_cams[valid_mask]
        query_paths = predefined_paths[valid_mask]

        gallery_feats = gallery_feats.detach().cpu()

        if dataset == 'sysu':
            if perm is None:
                raise RuntimeError("SYSU에는 perm 행렬이 필요합니다.")
            mAP, r1 = eval_sysu(
                query_feats, query_ids, query_cams,
                gallery_feats, gallery_ids, gallery_cams, np.array(gallery_paths),
                rerank=rerank
            )
        elif dataset == 'llcm':
            mAP, r1 = eval_llcm(
                query_feats, query_ids, query_cams,
                gallery_feats, gallery_ids, gallery_cams, np.array(gallery_paths),
                rerank=rerank
            )

        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        sum_metrics +=np.array([mAP, r1], dtype=np.float16)

        logging.info(f"Batch: {bi+1}/{len(online_loader)}  triplet_loss: {triplet_loss} total_loss: {loss}\n")

    stream_len = len(online_loader)
    final = sum_metrics / stream_len

    return dict(
        mAP=float(final[0]), r1=float(final[1])
    )


def test(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + f"edge_{cfg.output}.txt",
                        filemode="w")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    gallery_loader_dict, query_loader_dict, num_classes = get_edge_device_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       query_batch_size=64,
                                                       gallery_batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4,
                                                       relabel=True
                                                       )
    
    if cfg.model == 'resnet':
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

    elif cfg.model == 'AGW':
        model = AGW(cfg.num_id, no_local='on', gm_pool='on', arch='resnet50')
    
    model.to(device)

    torch.cuda.empty_cache()

    dataset = cfg.dataset

    for query_loader, gallery_loader, num_ids in zip(query_loader_dict.values(), gallery_loader_dict.values(), num_classes):
        if cfg.resume:
            if cfg.model == 'AGW':
                ckpt = torch.load(cfg.resume, map_location='cpu', weights_only=False)
            else:
                ckpt = torch.load(cfg.resume, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            def strip_module_prefix(sd):
                if any(k.startswith("module.") for k in sd.keys()):
                    return {k[len("module."):]: v for k, v in sd.items()}
                return sd

            state_dict = strip_module_prefix(state_dict)

            # drop classifier
            filtered = {
                k: v for k, v in state_dict.items()
                if not (k == "classifier.weight" or k == "classifier.bias" or k.startswith("classifier.")
                        or k.startswith("classifier_sp.") or k.startswith("C_sp_f."))
            }

            msg = model.load_state_dict(filtered, strict=False)
            print("Missing keys:", msg.missing_keys)
            print("Unexpected keys:", msg.unexpected_keys)
        else:
            raise Exception("Verify checkpoint path for testing!")

        evaluator = create_eval_engine(model, non_blocking=True)
        rank=True

        evaluator.run(query_loader)

        predefined_feats = torch.cat(evaluator.state.feat_list, dim=0)
        predefined_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        predefined_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        predefined_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(cfg.data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']

            # 예: 'all', num_shots=1 기준
            results = adapt_module(
                    dataset='sysu',
                    model=model,
                    predefined_feats=predefined_feats, predefined_ids=predefined_ids, predefined_cams=predefined_cams, predefined_paths=predefined_paths, predefined_num_classes=num_ids,
                    online_loader=gallery_loader,
                    rerank=rank,
                    perm=perm,
                    device=device,
                    # TTA 설정
                    update='bn',           # BN만 업데이트 권장
                    lr=0.0001,
                    weight_decay=0.0,
                    amp=True,
                    metric='cosine',
                    lam_fgt=0.0001,      # 드리프트 방지 원하면 >0로 (예: 1e-6)
                )
        elif dataset == 'llcm':
            results = adapt_module(
                    dataset='llcm',
                    model=model,
                    predefined_feats=predefined_feats, predefined_ids=predefined_ids, predefined_cams=predefined_cams, predefined_paths=predefined_paths, predefined_num_classes=num_ids,
                    online_loader=gallery_loader,
                    rerank=rank,
                    perm=None,
                    device=device,
                    # TTA 설정
                    update='bn',           # BN만 업데이트 권장
                    lr=0.0001,
                    weight_decay=0.0,
                    amp=True,
                    metric='cosine',
                    lam_fgt=0.0001,      # 드리프트 방지 원하면 >0로 (예: 1e-6)
                )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        logging.info(f"[ONLINE-FINAL]  r1={results['r1']:.2f}  mAP={results['mAP']:.2f}")
    

if __name__ == "__main__":
    import argparse
    import random
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'IDKL', 'AGW'], help='Model to TTA')
    parser.add_argument('--resume', type=str, default='', help='model checkpoint path')
    parser.add_argument('--query-set', type=str, choices=['one', 'few', 'multi'], default='multi', help='query set configuration')
    parser.add_argument('--output', type=str, default='', help='log output filename')

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
    cfg['output'] = args.output

    cfg.freeze()

    test(cfg)