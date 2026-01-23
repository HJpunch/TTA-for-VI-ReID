import os
import re
import random
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Literal
from collections import defaultdict
from natsort import natsorted

from .ChannelAug import ChannelExchange


class SYSUDataset(Dataset):
    def __init__(self, root, mode=None, transform=None, cam_id=None, relabel=False):
        assert os.path.isdir(root)
        assert mode in ['gallery', 'query']

        test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
        selected_ids = test_ids.strip('\n').split(',')
            
        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        self.channel_aug = ChannelExchange()
        self.stream_cam_id = cam_id

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) == self.stream_cam_id]

            def _cam_id_from_path(p: str) -> int:
                cam_dir = os.path.normpath(p).split(os.sep)[-3]  # 'cam1'
                return int(cam_dir.replace('cam', ''))

            def _parse_pid(p: str) -> int:
                return int(os.path.normpath(p).split(os.sep)[-2])

            def _parse_inst(p: str) -> int:
                fname = os.path.splitext(os.path.basename(p))[0]  # '0001'
                return int(fname)

            # --------------------------------------------------
            # 1) (pid, cam)별 local 시퀀스 구성
            #    seq_key_to_seq[(pid, cam)] = [(inst, pid, cam, path), ...]
            # --------------------------------------------------
            seq_key_to_seq = defaultdict(list)

            for p in img_paths:
                cam = _cam_id_from_path(p)

                pid = _parse_pid(p)
                # selected_ids만 대상으로 하고 싶으면 유지
                if pid not in selected_ids:
                    continue

                inst = _parse_inst(p)
                key = (pid, cam)
                seq_key_to_seq[key].append((inst, pid, cam, p))

            # 각 (pid, cam) 시퀀스를 inst 기준으로 정렬 (카메라 내 시간순)
            for key in list(seq_key_to_seq.keys()):
                seq = seq_key_to_seq[key]
                if not seq:
                    del seq_key_to_seq[key]
                    continue
                seq_key_to_seq[key] = sorted(seq, key=lambda x: x[0])  # inst 오름차순

            # --------------------------------------------------
            # 2) 전략 2-3: (pid, cam) 트랙들 간 인터리브
            #    - 한 번 선택된 트랙에서 1~max_track_len 프레임 연속으로 뽑고
            #      다시 다른 트랙으로 넘어감
            #    - seed 기반 재현성
            # --------------------------------------------------
            def build_online_stream_with_tracks(
                seq_key_to_seq: dict,
                max_track_len: int = 5,
                seed: int = 42,
            ):
                rng = random.Random(seed)

                # 각 트랙((pid, cam))별로 "다음으로 읽을 인덱스" 포인터
                key_to_idx = {key: 0 for key in seq_key_to_seq}
                active_keys = list(seq_key_to_seq.keys())
                stream = []  # (path, pid, cam, inst)

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
                        inst, pid, cam, p = seq[key_to_idx[key]]
                        stream.append((p, pid, cam, inst))
                        key_to_idx[key] += 1

                    # 이 트랙의 시퀀스가 끝났으면 active 리스트에서 제거
                    if key_to_idx[key] >= len(seq):
                        active_keys.remove(key)

                return stream

            max_track_len = 5  # 한 카메라에서 같은 사람을 연속으로 볼 최대 길이

            ordered = build_online_stream_with_tracks(
                seq_key_to_seq,
                max_track_len=max_track_len,
            )
            # 최종 gallery stream
            img_paths = [p for (p, pid, cam, inst) in ordered]

        elif mode == 'query':
            if self.stream_cam_id in (1, 2, 4, 5):
                img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

            elif self.stream_cam_id in (3, 6):
                img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if relabel:
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        # if path.split('/')[-3][-1] not in [3, 6]:
        #     img = self.channel_aug(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
    

class LLCMDataset(Dataset):
    def __init__(self, root, mode=None, transform=None, cam_id=None, relabel=False):
        assert os.path.isdir(root)
        assert mode in ['gallery', 'query']

        def _parse_cam_from_fname(p: str) -> int:
                # ..._c03_... → 3
                m = re.search(r'_c(\d+)_', os.path.basename(p))
                return int(m.group(1)) if m else -1
        
        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx
        
        def sort_by_scene_frame(img_paths):
            def _parse_scene_frame(path):
                fname = os.path.basename(path)
                # s171429, f12990 추출
                m = re.search(r'_s(\d+)_f(\d+)_', fname)
                if m is None:
                    raise ValueError(f"Invalid filename format: {fname}")
                scene = int(m.group(1))
                frame = int(m.group(2))
                return scene, frame

            return sorted(img_paths, key=_parse_scene_frame)
        
        self.channel_aug = ChannelExchange()
        self.transform = transform
        self.stream_cam_id = cam_id

        index_RGB = loadIdx(open(root + '/idx/test_vis.txt','r'))
        index_IR  = loadIdx(open(root + '/idx/test_nir.txt','r'))

        rgb_imgs = glob(f"{root}/test_vis/*[{cam_id}]*/*/*.jpg")
        ir_imgs = glob(f"{root}/test_nir/*[{cam_id}]*/*/*.jpg")

        img_paths = []
        ids = []

        # 나머지 카메라는 이미지 장수가 너무 적어서 3,4,5,6 만 사용.
        if len(rgb_imgs) >= len(ir_imgs):
            if mode == 'gallery':
                for path, pid in index_RGB:
                    if _parse_cam_from_fname(path) not in [cam_id]:
                        continue
                    img_paths.append(root+ '/' + path)
                    ids.append(int(pid))

            elif mode == 'query':
                for path, pid in index_IR:
                    if _parse_cam_from_fname(path) not in [3, 4, 5, 6]:
                        continue
                    if int(path.split('/')[-2]) in [273, 639, 693]:  # 이건 rgb 3, 4, 5, 6 카메라에 없는 id임.
                        continue
                    img_paths.append(root+ '/' + path)
                    ids.append(int(pid))

        else:
            if mode == 'gallery':
                for path, pid in index_IR:
                    if _parse_cam_from_fname(path) not in [cam_id]:
                        continue
                    if int(path.split('/')[-2]) in [273, 639, 693]:  # 이건 rgb 3, 4, 5, 6 카메라에 없는 id임.
                        continue
                    img_paths.append(root + '/' + path)
                    ids.append(int(pid))

            elif mode == 'query':
                for path, pid in index_RGB:
                    if _parse_cam_from_fname(path) not in [3, 4, 5, 6]:  
                        continue
                    img_paths.append(root+ '/' + path)
                    ids.append(int(pid))
            
        self.ids = ids
        if mode == 'gallery':
            img_paths = sort_by_scene_frame(img_paths)

        elif mode == 'query':
            img_paths = img_paths

        if not relabel:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]
        
        selected_ids = list(set(self.ids))
        num_ids = len(selected_ids)

        self.img_paths = img_paths
        self.cam_ids = [_parse_cam_from_fname(path) for path in img_paths]
        self.num_ids = num_ids


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        # if 'vis' in path:
        #     img = self.channel_aug(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
