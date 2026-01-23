import os
import re
import random
import torch
import os.path as osp
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Literal
from collections import defaultdict
from torch.utils.data import DataLoader

import torchvision.transforms as T


# def collate_fn(batch):  # img, label, cam_id, img_path, img_id
#     samples = list(zip(*batch))

#     data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
#     data.insert(3, samples[3])
#     return data


def collate_fn(batch):
    imgs, pids, upids, camids, img_path, original_pid = zip(*batch)
    return torch.stack(imgs, dim=0), pids, upids, camids, original_pid, img_path



def make_data_loader_TTA(dataset, root, query_batch_size, gallery_batch_size, image_size, num_workers=4, query_set='multi'):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gallery_dataset = LLCMDataset(root, mode='gallery', transform=transform, query_set=query_set)
    query_dataset = LLCMDataset(root, mode='query', transform=transform, query_set=query_set)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=query_batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=gallery_batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)
    
    num_classes = None
    num_query_classes = query_dataset.num_ids
    num_query_cams = query_dataset.cam_num


    return gallery_loader, num_classes, query_loader, num_query_classes, num_query_cams

class LLCMDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, query_set='multi'):
        # Load training images (path) and labels
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        assert query_set in ['one', 'few', 'multi']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx
        
        self.mode = mode

        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_nir.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_nir.txt','r'))

        if mode in ['gallery', 'query']:
            query_imgs = []

            if query_set == 'multi':
                rgb_img_paths = defaultdict(list)
                ir_img_paths = defaultdict(list)
                for path, pid in index_RGB:
                    rgb_img_paths[pid].append(root + '/' + path)  
                for path, pid in index_IR:
                    ir_img_paths[pid].append(root + '/' + path)

                rng = random.Random(42)

                for path in rgb_img_paths.values():
                    items = rng.sample(path, 4)  # 각 query_id 마다 RGB/IR 이미지 네 장씩
                    query_imgs.extend(items)

                for path in ir_img_paths.values():
                    items = rng.sample(path, 4)
                    query_imgs.extend(items)

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]
            img_paths = [x for x in img_paths if x not in query_imgs]
        elif mode == 'query':
            img_paths = query_imgs
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        if mode == 'gallery':
            def _parse_pid(p: str) -> int:
                return int(os.path.normpath(p).split(os.sep)[-2])

            def _parse_cam_from_fname(p: str) -> int:
                # ..._c03_... → 3
                m = re.search(r'_c(\d+)_', os.path.basename(p))
                return int(m.group(1)) if m else -1

            def _parse_frame_from_fname(p: str) -> int:
                # ..._f39555_... → 39555
                m = re.search(r'_f(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else -1

            # --------------------------------------------------
            # 1) (pid, cam)별 local 시퀀스 구성
            #    seq_key_to_seq[(pid, cam)] = [(frame, pid, cam, path), ...]
            # --------------------------------------------------
            seq_key_to_seq = defaultdict(list)
            all_cams = set()

            for p in img_paths:
                cam = _parse_cam_from_fname(p)
                frm = _parse_frame_from_fname(p)
                if cam < 0 or frm < 0:
                    continue

                pid = _parse_pid(p)
                # selected_ids만 대상으로 쓰고 싶으면 필터 유지
                if pid not in selected_ids:
                    continue

                key = (pid, cam)
                seq_key_to_seq[key].append((frm, pid, cam, p))
                all_cams.add(cam)

            all_cams = sorted(all_cams)

            # 각 (pid, cam) 트랙을 frame 기준으로 정렬 (local 시간순)
            for key in list(seq_key_to_seq.keys()):
                seq = seq_key_to_seq[key]
                if not seq:
                    del seq_key_to_seq[key]
                    continue
                seq_key_to_seq[key] = sorted(seq, key=lambda x: x[0])  # frame 오름차순

            # --------------------------------------------------
            # 2) 트랙 기반 인터리브 (SYSU 방식과 동일)
            #    - 트랙 = (pid, cam)
            #    - 한 번 선택된 트랙에서 1~max_track_len frame 연속으로 뽑고
            #      다시 다른 트랙으로 넘어감
            #    - seed 기반 재현성
            # --------------------------------------------------
            def build_online_stream_with_tracks(
                seq_key_to_seq: dict,
                max_track_len: int = 5,
                seed: int = 42,
            ):
                rng = random.Random(seed)

                # 각 트랙((pid, cam))별 포인터
                key_to_idx = {key: 0 for key in seq_key_to_seq}
                active_keys = list(seq_key_to_seq.keys())
                stream = []  # (path, pid, cam, frame)

                while active_keys:
                    # 아직 남은 트랙들 중 하나를 랜덤 선택
                    key = rng.choice(active_keys)
                    seq = seq_key_to_seq[key]
                    idx = key_to_idx[key]

                    # 이번에 이 트랙에서 몇 프레임을 연속으로 쓸지 결정
                    remaining = len(seq) - idx
                    track_len = rng.randint(1, max_track_len)
                    track_len = min(track_len, remaining)

                    # 해당 트랙에서 track_len개 프레임을 local 시간순으로 추가
                    for _ in range(track_len):
                        frm, pid, cam, p = seq[key_to_idx[key]]
                        stream.append((p, pid, cam, frm))
                        key_to_idx[key] += 1

                    # 이 트랙 시퀀스가 끝났으면 active 리스트에서 제거
                    if key_to_idx[key] >= len(seq):
                        active_keys.remove(key)

                return stream

            max_track_len = 5  # 같은 (pid, cam)에서 연속으로 나올 최대 길이

            ordered = build_online_stream_with_tracks(
                seq_key_to_seq,
                max_track_len=max_track_len,
                seed=42,
            )

            # 최종 gallery stream 경로
            img_paths = [p for (p, pid, cam, frm) in ordered]

        else:
            img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'nir') + 2 for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform
        self.cam_num = set(self.cam_ids)

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))

            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]



        self.original_ids = [int(path.split('/')[-2]) for path in img_paths]

        if mode in ['train', 'query']:
            # train, query는 relabel
            unique_pids = sorted(set(self.original_ids))
            pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}
            self.ids = [pid2label[pid] for pid in self.original_ids]
        else:
            # gallery는 original pid 유지
            self.ids = self.original_ids

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        origin_label = torch.tensor(self.original_ids[item], dtype=torch.long)

        # return img, label, cam, path, item
        if self.mode in ['train', 'query']:
            return img, label, label, cam, origin_label, path
        return img, label, label, cam, label, path
