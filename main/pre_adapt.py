r"""
RGB query set에 대해 supervised fashion으로 pre-adapt 진행
1. RGB 이미지를 CA
2. RGB와 CA 배치를 이용해서 ReID loss로 fine tuning.
3. 그렇게 update 된 모델로 no adapt 결과 찍어보기.
성능이 그냥 no adapt 보다 올랐으면 CA가 도움이 되는 거임.
"""
import os
import logging
import torch
import torch.nn as nn
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

from data.ChannelAug import ChannelExchange
from data import collate_fn
from data.dataset import SYSUDataset, LLCMData  # pre adapt 용으로 RGB 데이터만 가져오기 위해
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.IDKL.layers.loss.triplet_loss import TripletLoss
from data.sampler import TraditionalRandomIdentitySampler


r"""
TODO
1. 전체 RGB 이미지로 모델을 통으로 pre adapt 하고 no adapt 찍어보기  (완)
2. 쿼리가 RGB인 캠에서 그 RGB만 가지고 pre adapt 하고 그 cam에 대해서만 pre adapt 찍어보기 (진행 중)
"""

def get_rgb_loader(root, dataset, image_size):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SYSUDataset(root, mode='gallery', transform=transform) if dataset=='sysu' else LLCMData(root, mode='gallery', transform=transform)
    sampler = TraditionalRandomIdentitySampler(dataset, 8 * 4, 4)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=32,
                             sampler=sampler,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=collate_fn,
                             num_workers=4)
    
    return data_loader, dataset.num_ids


def pre_adapt(model, data_loader, num_class, channel_aug, device='cuda', lr=1e-4, weight_decay=0.0, epoch=10):
    model.to(device)
    model.eval()
    model.classifier = nn.Linear(2048, num_class, bias=False).to(device)

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
    
    params = (p for m in _bn_layers(model) for p in m.parameters() if p.requires_grad)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=True)

    # cross_entropy_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    triplet_loss_function = TripletLoss(margin=0.3)

    for ep in range(epoch):
        for i, batch in enumerate(data_loader):
            model.zero_grad()
            imgs, ids, cams, paths, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            CA_imgs = torch.stack([channel_aug(img.clone()) for img in imgs], dim=0)

            adapt_batch = torch.cat((imgs, CA_imgs), dim=0)
            adapt_label = torch.cat((ids, ids), dim=0).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=True, device_type='cuda' if 'cuda' in device else 'cpu'):
                feats = model(adapt_batch)
                # logits = model.classifier()
                loss, *_ = triplet_loss_function(feats, adapt_label)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        print(f"Epoch: {ep+1}/{epoch}  loss: {loss}")

    return model


def no_adapt(
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
        device='cuda',
    ):
    """no adapt 성능 측정"""

    model.to(device)
    model.eval()  # 평가 모드 고정(파라미터 업데이트만 수행)

    sum_metrics = np.zeros(2, dtype=np.float64)  # [mAP, r1] (스케일 [0,1])

    for bi, batch in enumerate(online_loader):
        model.zero_grad()
        imgs, ids, cams, paths, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        gallery_feats = model(imgs)

        # non query 면 성능 측정 안함
        pid_eq  = (predefined_ids[:, None] == ids[None, :])
        valid_mask = (pid_eq).any(axis=1)
        vq = int(valid_mask.sum())
        if vq == 0:
            continue

        gallery_ids  = ids.cpu().numpy()
        gallery_cams = cams.cpu().numpy()
        gallery_paths = paths

        # ---------- (1) evaluation ----------
        query_feats = predefined_feats[valid_mask]
        query_ids  = predefined_ids[valid_mask]
        query_cams = predefined_cams[valid_mask]

        gallery_feats = gallery_feats.detach().cpu()

        if dataset == 'sysu':
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

        logging.info(f"Batch: {bi+1}/{len(online_loader)}\n")

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
                        filename=log_dir + "/" + f"pre_adapt_{cfg.output}.txt",
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
    
    print("[Preprocess] Start Pre-Adapt")
    pre_adapt_loader, num_adapt_classes = get_rgb_loader(cfg.data_root, dataset, cfg.image_size)
    model = pre_adapt(model, pre_adapt_loader, num_adapt_classes, ChannelExchange())
    print("[Preprocess] End Pre-Adapt")

    for query_loader, gallery_loader, num_ids in zip(query_loader_dict.values(), gallery_loader_dict.values(), num_classes):
        evaluator = create_eval_engine(model, non_blocking=True)
        rank=True

        evaluator.run(query_loader)

        predefined_feats = torch.cat(evaluator.state.feat_list, dim=0)
        predefined_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        predefined_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        predefined_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        results = no_adapt(
                dataset=dataset,
                model=model,
                predefined_feats=predefined_feats, predefined_ids=predefined_ids, predefined_cams=predefined_cams, predefined_paths=predefined_paths, predefined_num_classes=num_ids,
                online_loader=gallery_loader,
                rerank=rank,
                device=device,
            )

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
    cfg['output'] = args.output

    cfg.freeze()

    test(cfg)