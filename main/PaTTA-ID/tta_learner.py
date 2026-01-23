import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from .sotta_memory import HUS
from .contrast import ViewContrastiveLoss

from collections import defaultdict
import os
import torchvision.utils as vutils
import os.path as osp
from torch.utils.data import DataLoader
import numpy as np
torch.set_printoptions(precision=2, sci_mode=False)

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class IterLoader:
    def __init__(self, dataloader, length=None):
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
        
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, unlabeled=False):
        self.dataset = dataset
        self.transform = transform
        self.unlabeled = unlabeled
        # self.gen = gen
        # self.gen_root = gen_root


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, original_pid, cnts = self.dataset[index]

        # if self.gen:
        #     img_name = img_path.split('/')[-1]
        #     img_path = osp.join(self.gen_root,img_name)
        #     img = read_image(img_path)
        # else:
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.unlabeled:
            return img, pid, cnts, camid, img_path, original_pid
        else:
            return img, pid, cnts, camid, img_path, original_pid


import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                # remove pid immediately, regardless of leftover
                avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def train_collate_fn(batch):
    imgs, pids, upids, camid, img_path, original_pid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    upids = torch.tensor(upids, dtype=torch.int64)
    # original_pid = torch.tensor(original_pid, dtype=torch.int64)

    # camid = torch.tensor(camid, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, upids, camid, original_pid, img_path

class TTA_Learner(nn.Module):
    def __init__(self, cfg, model, optimizer, num_class, cam_mean=None, cam_std=None):
        super().__init__()
        self.model = model
        for param in self.model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = True

        self.cfg =cfg
        
        self.optimizer = optimizer
        self.steps = 1

        self.update_frequency = 64
        self.current_instance =0
        print("self.update_frequency == ",self.update_frequency)


        self.cam_mean = cam_mean 
        self.cam_std = cam_std

        self.model.cuda()

        self.temp = 1.0

        print("self.temp == ",self.temp)
        memory_size = 64
        lr = 0.00035
        thres = 0.3

        print("memory_size == ",memory_size)
        print("thres == ",thres)
        print("lr == ",lr)

        self.num_query_class = num_class
        print("self.num_query_class == ",self.num_query_class)
        self.mem = HUS(cfg=cfg, capacity=memory_size, num_class=num_class, threshold=thres)

        self.age_temp = 10
        print("self.age_temp == ",self.age_temp)

        top_K = 50

        self.entropy_loss = ViewContrastiveLoss(T=self.temp, top_k=top_K, hardest=0)  #loss_func
        # self.entropy_loss = ReID_HLoss(1.0, k=50)

        self.kl = nn.KLDivLoss(reduction='batchmean') 
        self.scale_kl = 0.3  ## 0.3
        self.lambda_kl = 20  ## 20
        self.kl_loss_w = 1.0
        self.vcl_loss_w = 1.0
        self.adaptive = 0
        self.coreset_old_adaptive = 0


        self.momentum = 0.5

    def build_ema(self, model):
        ema_model = deepcopy(model)
        return ema_model

    def set_query_memory(self, cfg, query_data, query_feats, query_gt_pids, query_cam_ids, query_img_paths, query_upids):
        query_cam_ids = np.array(query_cam_ids)
        query_coreset = []
        query_img_paths = np.array(query_img_paths)

        cross_cam_pairs = 0
        
        for pid in range(self.num_query_class):

            idxs = (np.array(pid)==np.array(query_gt_pids))
            # query_imgs = query_data[idxs]
            query_tmp_img_paths = query_img_paths[idxs]
            query_tmp_cams = query_cam_ids[idxs]
            query_tmp_feats =   query_feats[idxs]

            self.cam_id = query_tmp_cams

            query_tmp_upids = torch.tensor(query_upids)[idxs]
            
            # ############# Random sampling ##############
            # if len(query_tmp_img_paths)>=2:
            #     rand_idx = np.random.choice(len(query_tmp_img_paths), size=2, replace=False)
            #     for i in rand_idx:
            #         query_coreset.append((query_tmp_img_paths[i], pid, 5555, 0, 0))
            # else:
            #     print("Ignore PID : {} for query coreset, no 4 instances in".format(pid))


            cross_cam_idxs=(self.cam_id!=query_tmp_cams)# Find images that has diff cam ID to current stream cam ID
            same_cam_idxs=(self.cam_id==query_tmp_cams)# Find images that has same cam ID to current stream cam ID
            
            cross_cam_feats = query_tmp_feats[cross_cam_idxs]
            same_cam_feats = query_tmp_feats[same_cam_idxs]
            
            same_cam_img_path = query_tmp_img_paths[same_cam_idxs]
            cross_cam_img_path = query_tmp_img_paths[cross_cam_idxs]

            same_cam_upids = query_tmp_upids[same_cam_idxs]
            cross_cam_upids = query_tmp_upids[cross_cam_idxs]
            
            
            # ############# Cross-cam sampling only ##############
            # if len(same_cam_img_path)>=2:
            #     rand_idx = np.random.choice(len(same_cam_img_path), size=2, replace=False)
            #     for i in rand_idx:
            #         query_coreset.append((same_cam_img_path[i], pid, 5555, 0, 0))
            # else:
            #     print("Ignore PID : {} for query coreset, no same-cam + cross-cam instances".format(pid))

 
            if len(same_cam_img_path)>0:
                # same-cam 중 1장 무작위 선택
                if len(same_cam_img_path)>=2:
                    rand_idx = np.random.choice(len(same_cam_img_path), size=2, replace=False)
                    sampled_same_path = same_cam_img_path[rand_idx]
                    sampled_same_feat = same_cam_feats[rand_idx]
                    sampled_same_upids = same_cam_upids[rand_idx]

                    query_coreset.append((sampled_same_path[0], pid, 0, 0, sampled_same_upids[0].item()))

                    if len(cross_cam_img_path) > 0:
                        # hardest cross-cam: sampled_same_feat와 가장 feature 유사도가 낮은 것 선택
                        sim = sampled_same_feat[0] @ cross_cam_feats.t() 
                        hardest_idx = torch.argmin(sim).item()
                        sampled_cross_path = cross_cam_img_path[hardest_idx]
                        sampled_cross_upid = cross_cam_upids[hardest_idx]

                        query_coreset.append((sampled_cross_path, pid,  0, 0, sampled_cross_upid.item()))
                    else:
                        query_coreset.append((sampled_same_path[1], pid, 0, 0, sampled_same_upids[1].item()))
                else:
                    rand_idx = np.random.choice(len(same_cam_img_path), size=1, replace=False)
                    sampled_same_path = same_cam_img_path[rand_idx]
                    sampled_same_feat = same_cam_feats[rand_idx]
                    sampled_same_upids = same_cam_upids[rand_idx]


                    if len(cross_cam_img_path) > 0:
                        # hardest cross-cam: sampled_same_feat와 가장 feature 유사도가 낮은 것 선택
                        sim = sampled_same_feat @ cross_cam_feats.t() 

                        hardest_idx = torch.argmin(sim).item()
                        sampled_cross_path = cross_cam_img_path[hardest_idx]
                        sampled_cross_upid = cross_cam_upids[hardest_idx]
                        query_coreset.append((sampled_cross_path, pid, 0, 0, sampled_cross_upid.item()))
                        query_coreset.append((sampled_same_path[0], pid, 0, 0, sampled_same_upids[0].item()))
                    else:
                        print("Ignore PID : {} for query coreset, only 1 same-cam instance".format(pid))
                    

            elif len(cross_cam_img_path) >= 2:
                # cross_cam_img_path와 cross_cam_idxs는 query_tmp_img_paths[idxs] 기준
                cross_cam_cams = query_tmp_cams[cross_cam_idxs]  # 카메라 ID 정보

                # camid별 path 그룹화
                cam2paths = defaultdict(list)
                cam2upids = defaultdict(list)

                for camid, path, upid in zip(cross_cam_cams, cross_cam_img_path, cross_cam_upids):
                    cam2paths[camid].append(path)
                    cam2upids[camid].append(upid)

                unique_cams = list(cam2paths.keys())
                
                if len(unique_cams) >= 2:
                    # 서로 다른 두 camid 선택
                    selected_cams = np.random.choice(unique_cams, size=2, replace=False)
                    for camid in selected_cams:
                        path_list = cam2paths[camid]
                        upid_list = cam2upids[camid]

                        rand_idx =np.random.choice(len(path_list), size=1)[0]
                        query_coreset.append((path_list[rand_idx], pid, 0, 0, upid_list[rand_idx].item()))
                else:
                    # camid는 다르지만 2개 이상 없는 경우 -> 그냥 2장 랜덤
                    rand_idx = np.random.choice(len(cross_cam_img_path), size=2, replace=False)
                    for i in rand_idx:
                        query_coreset.append((cross_cam_img_path[i], pid, 0, 0, cross_cam_upids[i].item()))
                # print("Store PID : {} for query coreset, only cross-cam instances".format(pid))
                
                cross_cam_pairs +=1 
                
            # else:
            #     print("Ignore PID : {} for query coreset, no same-cam + cross-cam instances".format(pid))

        print("num_query_tta == ",self.num_query_class)
        
        print("cross_cam_pairs == ",cross_cam_pairs)

        iters = self.total_iter
        # train_transforms = build_transforms(cfg, is_train=True)
        train_transforms = T.Compose([
            T.Resize((384, 144)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("query_coreset == ",len(query_coreset))
        query_coreset_train = ImageDataset(query_coreset, train_transforms)

        coreset_iterloader = IterLoader(
        DataLoader(query_coreset_train,
                   batch_size=64, num_workers=8,  drop_last=False, sampler=RandomIdentitySampler(query_coreset, 32, 2))
                   , length=iters)
        
        coreset_trainloader = DataLoader(
            query_coreset_train, batch_size=64,
            sampler=RandomIdentitySampler(query_coreset, 32, 2),
            num_workers=8, collate_fn=train_collate_fn, drop_last=False)
        
        self.query_coreset_iterloader = coreset_iterloader

        return coreset_trainloader
    
    def forward(self, iter, x, query_feats, query_cam_ids, query_aug_feats2, query_aug_pids2):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(iter, x, query_feats, query_cam_ids, self.model, self.optimizer, query_aug_feats2, query_aug_pids2)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, iter, batch_data, query_feats, query_cam_ids, model, optimizer, query_aug_feats2, query_aug_pids2, feat_norm='yes'):
        # batch data
        imgs =  batch_data[0]
        query_original_pids = batch_data[-2] 
        with torch.no_grad():
            # model.eval()
            self.model.eval()

            ema_out = self.model(imgs, test_time=True) 

            if True:  # self.cfg.FEAT_NORMED
                if self.cam_mean is not None:
                    ema_out = (ema_out - self.cam_mean) / self.cam_std

            if True:
                ema_out = torch.nn.functional.normalize(ema_out, dim=1, p=2)

            sims = ema_out @ F.normalize(query_feats,dim=1).T

            ########### SOTTA #########
            predict = torch.softmax(sims, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            pseudo_conf = sims.max(1)[0].cpu().numpy()
            pseudo_cls = np.array([self.query_gt_pids[p] for p in pseudo_label if p < len(self.query_gt_pids)])

        self.mem.query_gt_pids  = self.query_gt_pids

        # add into memory
        for i, (one_img, camid, org_pid, img_path) in enumerate(zip(batch_data[0], batch_data[1], batch_data[2], batch_data[-1])):
            self.mem.add_instance([one_img, pseudo_cls[i],  pseudo_conf[i], org_pid, 0])
            self.current_instance += 1
            

            if self.current_instance % self.update_frequency == 0: ## 64
                self.update_model(iter, model, optimizer, query_feats, query_original_pids)
                self.mem.add_age()
        return ema_out
    
    def update_model(self, iter, model, optimizer, query_feats, query_original_pids):
 
        memory_data, memory_pids, memory_sims, memory_org_pids, memory_age = self.mem.get_memory()

        # model.train()
        self.model.train()
        
        old_model = deepcopy(self.model).cuda()
        old_model.eval()
        
        
        
        if len(memory_data) == 0 or len(memory_data)==1:
            if True:  # self.cfg.QUERY_CORESET_TRAIN
                
                query_data = self.query_coreset_iterloader.next()
                query_imgs, q_pids, q_upids, _, _, _  = query_data
                query_imgs = query_imgs.cuda()
                q_pids = q_pids.cuda()
                
                score, feat, bn_feat = self.model(query_imgs)  # [1, feat_dim]
                loss_ce, loss_tri = self.loss_supervised(score, feat, q_pids)
                loss = loss_ce+loss_tri 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("Camera {} Stream Iter=[{}/{}] loss_ce == {:.2} loss_tri == {:.2} ".format(self.cam_id,iter, self.total_iter, loss_ce,loss_tri))
                
            
            
                if True:  # self.cfg.QUERY_DRIFT_UPDATE:
                    with torch.no_grad():
                        self.model.eval()
                        old_feats = old_model(query_imgs)
                        new_feats = self.model(query_imgs)
                        
                        drift_vec = (new_feats - old_feats)#.mean(dim=0)


                        all_drifts = []
                        for pid in q_pids:
                            idx = (pid==q_pids)
                            id_wise_drift = drift_vec[idx].mean(0) ### one pid drift vector
                            all_drifts.append(id_wise_drift.unsqueeze(0))
                        all_drifts = torch.cat(all_drifts)
                        
                        
                    # if self.cam_mean is not None:
                    #     #  bn_feat = ( bn_feat - self.cam_mean) / self.cam_std
                    #     drift_vec = (drift_vec-self.cam_mean) / self.cam_std
                    self.update_query_memory(query_feats, bn_feat, q_upids, all_drifts)    
            
            return    
        
        else:
            
            if True:  # self.cfg.QUERY_CORESET_TRAIN
                query_data = self.query_coreset_iterloader.next()
                query_imgs, q_pids, q_upids, _, _, _  = query_data
                query_imgs = query_imgs.cuda()
                q_pids = q_pids.cuda()
                
                score, feat, bn_feat = self.model(query_imgs)  # [1, feat_dim]
                loss_ce, loss_tri = self.loss_supervised(score, feat, q_pids)
            else:
                loss_ce, loss_tri = 0.0, 0.0
                
            

            data = torch.stack(memory_data).cuda()
            data_pids = torch.tensor(memory_pids).cuda()
            
            # self.memory_plot(memory_org_pids, memory_pids, data_pids)
            dataset = torch.utils.data.TensorDataset(data, data_pids)
            data_loader = DataLoader(dataset, batch_size=64,
                                    shuffle=True, drop_last=False, pin_memory=False)

            if True:  # self.cfg.CICA:
                for batch_idx, (feats, pids) in enumerate(data_loader):
                    loss_contrast = self.step(iter, model=self.model, optimizer=optimizer, loss_fn=self.entropy_loss, feats=feats, query_feats=query_feats, memory_pids=pids)
            else:
                loss_contrast = 0.0
                

            loss = self.vcl_loss_w *  loss_contrast  + loss_ce+loss_tri 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Camera {} Stream Iter=[{}/{}] loss_ce == {:.2} loss_tri == {:.2} loss_contrast == {:.2}".format(self.cam_id,iter, self.total_iter, loss_ce,loss_tri, loss_contrast))
            
            
            if True:  # self.cfg.FEAT_NORMED:
                for batch_idx, (feats, pids) in enumerate(data_loader):
                    self.model.eval()
                    feats.cuda()
                    eps = 1e-6
                    with torch.no_grad():
                        feat = self.model(feats,test_time=True)

                        m = feat.mean(dim=0)
                        s = feat.var(dim=0, unbiased=False).sqrt().clamp_min(eps)
                        self.cam_mean = m
                        self.cam_std = s


            if True:  # self.cfg.QUERY_DRIFT_UPDATE:
                with torch.no_grad():
                    self.model.eval()
                    old_feats = old_model(query_imgs)
                    new_feats = self.model(query_imgs)
                    
                    drift_vec = (new_feats - old_feats)#.mean(dim=0)


                    all_drifts = []
                    for pid in q_pids:
                        idx = (pid==q_pids)
                        id_wise_drift = drift_vec[idx].mean(0) ### one pid drift vector
                        all_drifts.append(id_wise_drift.unsqueeze(0))
                    all_drifts = torch.cat(all_drifts)
                    
                    
                self.update_query_memory(query_feats, bn_feat, q_upids, all_drifts)    


                

    def step(self, iter, model, optimizer, loss_fn, feats=None, query_feats=None, memory_pids=None):
        assert (feats is not None)
        feats = feats.cuda()
        preds_of_data = model(feats, test_time=True)[1]
        preds_of_data = F.normalize(preds_of_data, dim=1)
        
        # loss_contrast = self.entropy_loss(preds_of_data, query_feats)# + 0.0001 * param_l2_loss(self.model, self.model_init)


        all_contrast_loss = []
        weight_flag = self.coreset_old_adaptive
        query_gt_pids = self.query_gt_pids.cuda()
        normed_query_feats = F.normalize(query_feats,dim=1)
        for i, (feat, pid) in enumerate(zip(preds_of_data, memory_pids)):

            
            loss = loss_fn(feat, normed_query_feats, pid, query_gt_pids)

            all_contrast_loss.append(loss)
        
        loss_contrast = torch.mean(torch.stack(all_contrast_loss))
        
        
        return loss_contrast


    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
            

    def update_query_memory(self, query_feats, bn_feat, targets, drift_vecs=None):
        bn_feat = torch.nn.functional.normalize(bn_feat, dim=1, p=2)

        pseudo_cls = np.array([self.query_gt_pids[p] for p in targets if p < len(self.query_gt_pids)])
        with torch.no_grad():
            for i in range(len(bn_feat)):   
                one_feat = bn_feat[i]
                mem_idx = targets[i]
                
                gt_pid = pseudo_cls[i]
                all_mem_idxs = (gt_pid==self.query_gt_pids)
                idxs = all_mem_idxs.nonzero().squeeze(1)

                query_feats[idxs] = query_feats[idxs] + drift_vecs[i]


    def memory_plot(self, memory_org_pids, memory_pids, data_pids=None):
        cnt=0  
        query_original_pids = self.mem.query_original_pids
        for i in range(len(memory_org_pids)):
            pid = memory_org_pids[i]

            if pid in query_original_pids:
                cnt+=1

        
        if len(memory_org_pids)!=0 : self.evaluator.memory_gt_pid_cnts.append(cnt/len(memory_org_pids))
        if len(memory_org_pids)!=0 : print("Memory accuracy == {}/{}".format(cnt,len(memory_org_pids)))
        

        org_gt_cnt, cnt=0, 0
        # orgpid_2cls_pid = np.arange(len(query_original_pids))
        
        # pseudo_pid = np.array([self.query_gt_pids[p] for p in memory_pids if p < len(self.query_gt_pids)])
        for i in range(len(memory_pids)):
            org_pid = memory_org_pids[i]
            pseudo_cls = memory_pids[i]#memory_pids[i]



            if org_pid in query_original_pids:
                idx = (np.array(org_pid) == np.array(query_original_pids)).nonzero()[0][0]
                if (pseudo_cls == self.query_gt_pids[idx]):
                    cnt+=1
                # else:
                #     data_pids[i] = self.query_gt_pids[idx]
                #     cnt+=1


                org_gt_cnt +=1

        if org_gt_cnt!=0 : self.evaluator.memory_pseudo_label_acc.append(cnt/org_gt_cnt)
        if org_gt_cnt!=0 : print("Memory pseudo label accuracy == {:.1%}".format(cnt/org_gt_cnt))


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)