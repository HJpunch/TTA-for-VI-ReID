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

'''
    Specific dataset classes for person re-identification dataset.
    query: one shot image per id, random modality.
    gallery: all test images except for query image.
'''

'''
    random seed를 argument로 받으면 rgb id, ir id가 고정이니까
'''

def parse_represent_imgs(path):
    rep_imgs = {}
    img_paths = open(path).readlines()
    for item in img_paths:
        pid, path = item.strip('\n').split(' ')
        rep_imgs[int(pid)] = path
    return rep_imgs

def assign_query_modality(test_ids: List[int],
                          ratio: float = 0.5,
                          seed: int = 42):
    assert (0.0 <= ratio <= 1.0)

    unique_ids = sorted(set(test_ids))
    n_ids = len(unique_ids)

    n_rgb = int(round(n_ids * ratio))

    rng = random.Random(seed)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)

    rgb_ids = set(shuffled_ids[:n_rgb])

    id_to_modality = {}
    for pid in unique_ids:
        if pid in rgb_ids:
            id_to_modality[pid] = 'rgb'
        else:
            id_to_modality[pid] = 'ir'
    
    return id_to_modality


class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, query_set=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')
            
        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode in ['gallery', 'query']:
            # img_path가 모든 test image들
            rgb_rep_imgs = parse_represent_imgs(os.path.join(root, 'sysu_rgb.txt'))
            ir_rep_imgs = parse_represent_imgs(os.path.join(root, 'sysu_ir.txt'))
            query_modality = assign_query_modality(rgb_rep_imgs, ratio=0.66)
            for pid, modality in query_modality.items():
                if modality == 'rgb':
                    ir_rep_imgs.pop(pid)
                else:
                    rgb_rep_imgs.pop(pid)
            query_imgs = list(rgb_rep_imgs.values()) + list(ir_rep_imgs.values()) 

        if mode == 'gallery':
            img_paths = [x for x in img_paths if x not in query_imgs]
            # 시간 순 라운드 로빈
            cam_pid_to_inst = defaultdict(lambda: defaultdict(dict))
            gallery_cams = (1, 2, 3, 4, 5, 6)

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
                if cam not in gallery_cams:
                    continue

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
            img_paths = query_imgs

        if mode != 'gallery':
            img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
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

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, query_set=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode in ['gallery', 'query']:
            rgb_rep_imgs = parse_represent_imgs(os.path.join(root, 'regdb_rgb.txt'))
            ir_rep_imgs = parse_represent_imgs(os.path.join(root, 'regdb_ir.txt'))
            query_modality = assign_query_modality(rgb_rep_imgs, ratio=0.5)
            for pid, modality in query_modality.items():
                if modality == 'rgb':
                    ir_rep_imgs.pop(pid)
                else:
                    rgb_rep_imgs.pop(pid)
            query_imgs = list(rgb_rep_imgs.values()) + list(ir_rep_imgs.values()) 

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

        if mode == 'gallery':  # cam round robin
            def _parse_pid(p: str) -> int:
                return int(os.path.normpath(p).split(os.sep)[-2])

            def _parse_inst_mod(p: str) -> tuple[int, str]:
                # 파일명 예: male_front_v_00007_1.bmp
                fname = os.path.basename(p)
                m = re.search(r'_([vt])_(\d+)', fname)
                if not m:
                    raise Exception(f"Invalid RegDB filename format: {fname}")
                mod  = m.group(1)           # 'v' or 't'
                inst = int(m.group(2))      # 7
                return inst, mod

            # --------------------------------------------------
            # 1) (pid, mod)별 local 시퀀스 구성
            #    seq_key_to_seq[(pid, mod)] = [(inst, pid, mod, path), ...]
            # --------------------------------------------------
            seq_key_to_seq = defaultdict(list)

            for p in img_paths:
                pid = _parse_pid(p)
                # selected_ids만 대상으로 쓰고 싶으면 필터링 유지
                if pid not in selected_ids:
                    continue

                inst, mod = _parse_inst_mod(p)
                key = (pid, mod)  # 트랙 = (pid, 'v'/'t')
                seq_key_to_seq[key].append((inst, pid, mod, p))

            # 각 (pid, mod) 트랙을 inst 기준으로 정렬 (local 시간순)
            for key in list(seq_key_to_seq.keys()):
                seq = seq_key_to_seq[key]
                if not seq:
                    del seq_key_to_seq[key]
                    continue
                seq_key_to_seq[key] = sorted(seq, key=lambda x: x[0])  # inst 오름차순

            # --------------------------------------------------
            # 2) 트랙 기반 인터리브 (전략 2-3)
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

                # 각 트랙((pid, mod))별 포인터
                key_to_idx = {key: 0 for key in seq_key_to_seq}
                active_keys = list(seq_key_to_seq.keys())
                stream = []  # (path, pid, mod, inst)

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
                        inst, pid, mod, p = seq[key_to_idx[key]]
                        stream.append((p, pid, mod, inst))
                        key_to_idx[key] += 1

                    # 이 트랙의 시퀀스가 끝났으면 active 리스트에서 제거
                    if key_to_idx[key] >= len(seq):
                        active_keys.remove(key)

                return stream

            max_track_len = 5  # 같은 (pid, mod)에서 연속으로 나올 최대 길이

            ordered = build_online_stream_with_tracks(
                seq_key_to_seq,
                max_track_len=max_track_len,
            )

            # 최종 gallery stream 경로만 추출
            img_paths = [p for (p, pid, mod, inst) in ordered]

        else:
            img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths] 
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
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

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

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

        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_nir.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_nir.txt','r'))

        if mode in ['gallery', 'query']:
            rgb_rep_imgs = parse_represent_imgs(os.path.join(root, 'llcm_rgb.txt'))
            ir_rep_imgs = parse_represent_imgs(os.path.join(root, 'llcm_ir.txt'))

            query_imgs = []

            if query_set == 'one':
                query_modality = assign_query_modality(rgb_rep_imgs, ratio=0.5)

                for pid, modality in query_modality.items():
                    if modality == 'rgb':
                        ir_rep_imgs.pop(pid)
                    else:
                        rgb_rep_imgs.pop(pid)
                query_imgs = list(rgb_rep_imgs.values()) + list(ir_rep_imgs.values())

            # 1. RGB, IR 각각 한장씩
            elif query_set == 'few':
                for pid, rgb_rep in rgb_rep_imgs.items():
                    ir_rep = ir_rep_imgs[pid]
                    query_imgs.extend([rgb_rep, ir_rep])

            elif query_set == 'multi':
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

        if mode == 'train':
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

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
