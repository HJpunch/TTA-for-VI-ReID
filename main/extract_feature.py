import os
import torch
import random
import numpy as np

from data import get_edge_device_test_loader
from models.baseline import Baseline
from models.IDKL.IDKL import Baseline as IDKL
from engine.engine import create_eval_engine

from configs.default import strategy_cfg
from configs.default.dataset import dataset_cfg

from visualization.tsne import *
from visualization.kde import kde_plot
from visualization.cosine_similarity import plot_query_gallery_similarity_distribution


def test(cfg):
    gallery_loader_dict, query_loader_dict, num_classes = get_edge_device_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       query_batch_size=64,
                                                       gallery_batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4,
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

    torch.cuda.empty_cache()

    dataset = cfg.dataset

    for i, (query_loader, gallery_loader) in enumerate(zip(query_loader_dict.values(), gallery_loader_dict.values())):
        for img, label, cam, path, item in gallery_loader:
            import cv2
            # print(path[0])
            # exit()
            # for i, (im, p) in enumerate(zip(img, path)):
            #     test = im.numpy()
            #     test = np.transpose(test, (1,2,0))
            #     test = test * 255
            #     test = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
            #     test = test.astype(np.uint8).copy()
            #     cv2.imwrite(f"aug/{os.path.basename(p)}.png", test)
            # exit()

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
                if not (k == "classifier.weight" or k == "classifier.bias" or k.startswith("classifier.")
                        or k.startswith("classifier_sp.") or k.startswith("C_sp_f."))
            }

            msg = model.load_state_dict(filtered, strict=False)
            print("Missing keys:", msg.missing_keys)       # 예: ['classifier.weight', 'classifier.bias']
            print("Unexpected keys:", msg.unexpected_keys) # 보통 빈 리스트
        else:
            raise Exception("Verify checkpoint path for testing!")

        evaluator = create_eval_engine(model, non_blocking=True)

        evaluator.run(query_loader)
        query_feats = torch.cat(evaluator.state.feat_list, dim=0)
        query_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        query_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        query_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        evaluator = create_eval_engine(model, non_blocking=True)
        evaluator.run(gallery_loader)
        gallery_feats = torch.cat(evaluator.state.feat_list, dim=0)
        gallery_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        gallery_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        gallery_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # feature 가지고 distribution 구성하기.
        # kde_plot(query_feats, gallery_feats, save_path=f'UMAP_{dataset}_{i}.png')
        # plot_query_gallery_similarity_distribution(query_feats, query_ids, gallery_feats, gallery_ids,
        #                                            metric='euclidean',
        #                                            save_path=f'euclidean_Channel_Aug_{i}.png')
        plot_tsne_query_gallery(
            query_feats, query_ids,
            gallery_feats, gallery_ids,
            max_ids=19,
            # max_q_per_id=20,
            # max_g_per_id=20,
            normalize=True,
            save_path=f'{dataset}_tsne_{i}.png'
        )

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', type=str, default='Baseline', choices=['Baseline', 'IDKL', 'AGW'], help='Model to TTA')
    parser.add_argument('--resume', type=str, default='', help='model checkpoint path')
    parser.add_argument('--output', type=str, default='no_adapt', help='log output filename')

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