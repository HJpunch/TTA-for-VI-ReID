import os
import re
import os.path as osp
from glob import glob
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset

# handle gallery images only

class SYSUDataset(Dataset):
    def __init__(self, root, transform=None, cam_wise=False):
        # TODO cam_wise: edge device setting
        assert os.path.isdir(root)

        test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
        selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)
        self.num_ids = num_ids
        self.transform = transform

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        query_img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]
        self.query_img_paths = sorted(query_img_paths)
        self.query_cam_ids = [int(path.split('/')[-3][-1]) for path in query_img_paths]
        self.query_ids = [int(path.split('/')[-2]) for path in query_img_paths]

        gallery_img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        # self.gallery_img_paths = sorted(gallery_img_paths)
        # self.gallery_cam_ids = [int(path.split('/')[-3][-1]) for path in gallery_img_paths]
        # self.gallery_ids = [int(path.split('/')[-2]) for path in gallery_img_paths]

        # 갤러리를 시간순 + 멀티카메라로 정렬
        cam_pid_to_inst = defaultdict(lambda: defaultdict(dict))
        gallery_cams = (1, 2, 4, 5)

        def _cam_id_from_path(p: str) -> int:
            cam_dir = os.path.normpath(p).split(os.sep)[-3]  # 'cam1'
            return int(cam_dir.replace('cam', ''))

        def _parse_pid(p: str) -> int:
            return int(os.path.normpath(p).split(os.sep)[-2])

        def _parse_inst(p: str) -> int:
            fname = os.path.splitext(os.path.basename(p))[0]  # '0001'
            return int(fname)
        
        # 채우기
        for p in gallery_img_paths:
            cam = int(p.split('/')[-3][-1])
            if cam not in gallery_cams:
                continue
            pid = _parse_pid(p)
            inst = _parse_inst(p)
            cam_pid_to_inst[cam][pid][inst] = p

        all_inst = sorted({inst
                           for cam in gallery_cams
                           for pid_dict in cam_pid_to_inst[cam].values()
                           for inst in pid_dict.keys()})
        
        # selected_ids 순서를 유지하며, instance → pid → cam 순 인터리브
        ordered = []
        for inst in all_inst:                           # 시간축
            for pid in sorted(selected_ids):            # ID 순회
                for cam in gallery_cams:                # 카메라 라운드로빈
                    p_dict = cam_pid_to_inst[cam].get(pid, None)
                    if p_dict is not None and inst in p_dict:
                        p = p_dict[inst]
                        ordered.append((p, pid, cam, inst))

        self.gallery_img_paths = [p for (p, pid, cam, inst) in ordered]
        self.gallery_ids       = [pid for (p, pid, cam, inst) in ordered]
        self.gallery_cam_ids   = [cam for (p, pid, cam, inst) in ordered]


    def __len__(self):
        # gallery만 배치로 다루므로 gallery에 대해 len 계산
        return len(self.gallery_img_paths)

    def __getitem__(self, item):
        # gallery만 배치로 다루므로 gallery에 대해 getitem
        path = self.gallery_img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.gallery_ids[item], dtype=torch.long)
        cam = torch.tensor(self.gallery_cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class RegDBDataset(Dataset):
    def __init__(self, root, transform=None, cam_wise=False):
        assert os.path.isdir(root)

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
        index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))
        self.transform = transform

        query_img_paths = [root + '/' + path for path, _ in index_IR]
        query_selected_ids = [int(path.split('/')[-2]) for path in query_img_paths]
        query_selected_ids = list(set(query_selected_ids))
        query_num_ids = len(query_selected_ids)

        query_img_paths = sorted(query_img_paths)
        self.query_img_paths = query_img_paths

        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.query_cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in query_img_paths] 
        self.query_num_ids = query_num_ids
        self.query_ids = [int(path.split('/')[-2]) for path in query_img_paths]

        gallery_img_paths = [root + '/' + path for path, _ in index_RGB]

        gallery_selected_ids = [int(path.split('/')[-2]) for path in gallery_img_paths]
        gallery_selected_ids = list(set(gallery_selected_ids))
        gallery_num_ids = len(gallery_selected_ids)

        gallery_img_paths = sorted(gallery_img_paths)
        # self.gallery_img_paths = gallery_img_paths

        # self.gallery_num_ids = gallery_num_ids
        # self.gallery_ids = [int(path.split('/')[-2]) for path in gallery_img_paths]

        # 갤러리 시간순 pid 순으로 정렬
        def _parse_pid(p: str) -> int:
            return int(os.path.normpath(p).split(os.sep)[-2])

        def _parse_inst(p: str) -> int:
            # 파일명 예: male_front_v_00007_1.bmp -> inst = 7
            fname = os.path.basename(p)
            m = re.search(r'_v_(\d+)', fname)
            if m:
                return int(m.group(1))
            # 혹시 다른 포맷이면 에러
            else:
                raise Exception("Invalid RegDB filename format!")
            
         # pid -> {inst: path}
        by_pid = defaultdict(dict)
        for p in gallery_img_paths:
            pid  = _parse_pid(p)
            inst = _parse_inst(p)
            if inst >= 0:
                by_pid[pid][inst] = p

        # 전역 시간축(inst) 정렬
        all_inst = sorted({inst for d in by_pid.values() for inst in d.keys()})
        pids_order = gallery_selected_ids  # pid 정렬 순서 일관화

        ordered = []
        for inst in all_inst:             # 시간 축 우선
            for pid in pids_order:        # pid 인터리브
                p_dict = by_pid.get(pid)
                if p_dict is not None and inst in p_dict:
                    ordered.append((p_dict[inst], pid))

        # 최종 갤러리 시퀀스 확정
        self.gallery_img_paths = [p for p, _ in ordered]
        self.gallery_ids       = [pid for _, pid in ordered]
        self.gallery_cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in gallery_img_paths] 


    def __len__(self):
        return len(self.gallery_img_paths)

    def __getitem__(self, item):
        path = self.gallery_img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.gallery_ids[item], dtype=torch.long)
        cam = torch.tensor(self.gallery_cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
    

class LLCMData(Dataset):
    def __init__(self, root, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        assert os.path.isdir(root)

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        index_RGB = loadIdx(open(root + '/idx/test_vis.txt','r'))
        index_IR  = loadIdx(open(root + '/idx/test_nir.txt','r'))
        self.transform = transform

        query_img_paths = [root + '/' + path for path, _ in index_IR]
        query_selected_ids = [int(path.split('/')[-2]) for path in query_img_paths]
        query_selected_ids = list(set(query_selected_ids))
        query_num_ids = len(query_selected_ids)

        query_img_paths = sorted(query_img_paths)
        self.query_img_paths = query_img_paths

        self.query_cam_ids = [int(path.split('/')[-3] == 'nir') + 2 for path in query_img_paths]
        self.query_num_ids = query_num_ids
        self.query_ids = [int(path.split('/')[-2]) for path in query_img_paths]

        gallery_img_paths = [root + '/' + path for path, _ in index_RGB]

        gallery_selected_ids = [int(path.split('/')[-2]) for path in gallery_img_paths]
        gallery_selected_ids = list(set(gallery_selected_ids))
        gallery_num_ids = len(gallery_selected_ids)

        gallery_img_paths = sorted(gallery_img_paths)
        # self.gallery_img_paths = gallery_img_paths

        # self.gallery_cam_ids = [int(path.split('/')[-3] == 'nir') + 2 for path in gallery_img_paths]
        # self.gallery_num_ids = gallery_num_ids
        # self.gallery_ids = [int(path.split('/')[-2]) for path in gallery_img_paths]

        # 갤러리 시간순 멀티카메라 정렬
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
        
        # cam → pid → {frame: path}
        cam_pid_frame = defaultdict(lambda: defaultdict(dict))
        all_cams = set()
        for p in gallery_img_paths:
            cam = _parse_cam_from_fname(p)
            pid = _parse_pid(p)
            frm = _parse_frame_from_fname(p)
            if cam >= 0 and frm >= 0:
                cam_pid_frame[cam][pid][frm] = p
                all_cams.add(cam)

        all_cams = sorted(all_cams)
        # 전역 시간축: 모든 프레임의 합집합을 오름차순
        all_frames = sorted({frm for cam in all_cams
                                   for pid_dict in cam_pid_frame[cam].values()
                                   for frm in pid_dict.keys()})
        
        # pid 순서를 고정(안정적 인터리브)
        pids_order = gallery_selected_ids

        ordered = []
        for frm in all_frames:              # 시간축 우선
            for pid in pids_order:          # pid 인터리브
                for cam in all_cams:        # 카메라 라운드로빈
                    fdict = cam_pid_frame[cam].get(pid)
                    if fdict is not None and frm in fdict:
                        ordered.append((fdict[frm], pid, cam, frm))

        # 최종 갤러리 시퀀스 확정
        self.gallery_img_paths = [p   for (p, pid, cam, frm) in ordered]
        self.gallery_ids       = [pid for (p, pid, cam, frm) in ordered]
        self.gallery_cam_ids   = [cam for (p, pid, cam, frm) in ordered]  # c03 → 3
        self.gallery_num_ids = gallery_num_ids

    def __len__(self):
        return len(self.gallery_img_paths)

    def __getitem__(self, item):
        path = self.gallery_img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.gallery_ids[item], dtype=torch.long)
        cam = torch.tensor(self.gallery_cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item