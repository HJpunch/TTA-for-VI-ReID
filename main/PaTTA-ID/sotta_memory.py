
import random
import copy
import torch
import torch.nn.functional as F
import numpy as np
import math


class HUS:
    def __init__(self,cfg, capacity, num_class, threshold=0.5):
        self.data = [[[], [], [], [], []] for _ in
                     range(num_class)]  # feat, pseudo_cls, pseudo_sim, org_pid, age
        print(f"num class: {num_class}")
        self.cfg = cfg
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.num_class = num_class
        self.capacity = capacity
        self.threshold = threshold
        print("self.threshold == ",self.threshold)

        self.use_org_sotta = 0
        print("self.use_org_sotta == ",self.use_org_sotta)

        
    def increase_age(self, item):
        # if not self.empty():
            # self.age += 1
        if len(item)!=0:
            for i in range(len(item)):
                item[i]+=1

    def add_age(self):
        for class_list in self.data:
            for i, item in enumerate(class_list):
                if i == (len(class_list)-1):
                    self.increase_age(item)
        return

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], [], [], []]
        for data_per_cls in data:
            feats, cls, sim, org_pid, age = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(sim)
            tmp_data[3].extend(org_pid)
            tmp_data[4].extend(age)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 5)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[2] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    # def add_instance(self, instance, query_original_pids):
    #     assert (len(instance) == 5)
    #     cls = instance[1]
    #     self.counter[cls] += 1
    #     is_add = False

    #     # current_cam_id = instance[2]

    #     org_pid = instance[-2]
    #     if org_pid in query_original_pids:
    #         is_add = True

    #         idx = (np.array(org_pid) == np.array(query_original_pids)).nonzero()[0][0]
    #         gt_pseudo_cls = self.query_gt_pids[idx]

    #         instance[1] = gt_pseudo_cls

    #         # if current_cam_id==self.query_cam_ids[idx]: ### store only cross-cam query instances
    #         #     is_add = False
    #         # else:
    #         if self.get_occupancy() >= self.capacity:
    #             is_add = self.remove_instance(gt_pseudo_cls)


    #     if is_add:
    #         for i, dim in enumerate(self.data[gt_pseudo_cls]):
    #             dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices


    def get_target_index(self, sims):
        if self.use_org_sotta:
            return random.randrange(0, len(sims))
        else:
            min_idx = np.array(sims).argmin()
            return min_idx
    
    def get_oldest_index(self, largest_indices):
 
        max_age = -1       
        for idx, pid in enumerate(largest_indices):
            
            age = np.mean(self.data[pid][4]) # self.data[pid][4][0] ### age
            if age>=max_age:
                max_age = age
                max_idx = idx
            
            
        return largest_indices[max_idx]
    
    
    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            if self.use_org_sotta:
                largest = random.choice(largest_indices)  # select only one largest class
            else:
                largest = self.get_oldest_index(largest_indices)
            
            tgt_idx = self.get_target_index(self.data[largest][2])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][2])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True



