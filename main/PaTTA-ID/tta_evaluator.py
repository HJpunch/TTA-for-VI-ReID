import torch
import numpy as np
from scipy.special import softmax
class TTA_Evaulator():
    def __init__(self):
        self.max_rank = 10
        self.rank1_list = []
        self.ap_list = []
        self.valid_num_query = 0 # number of valid query during gallery streaming
        self.valid_num_non_query = 0
        self.false_pos_cnts = 0.0
        self.true_neg_cnts = 0.0
        self.same_cam_true_neg_cnts = 0
        self.same_cam_false_pos_cnts = 0
        self.rank1_similarities_non_query, self.all_similarities_non_query = [], []
        self.rank1_similarities_cross_cam_query, self.rank1_similarities_same_cam_query = [], []
        self.all_similarities_query = []

        self.all_sims, self.all_labels = [], []

        self.memory_pseudo_label_acc, self.memory_gt_pid_cnts = [], []


    def eval_galllery_stream2query_set(self, query_feats, gallery_feats,  q_pids, g_pids, q_camids, g_camids):
        ### Use predefined query set for gallery instance evaluation

        sim = gallery_feats @ query_feats.T

        sim = sim.detach().cpu().numpy()
        indices = np.argsort(-sim)
        num_q, num_g = sim.shape

        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        
        matches = (q_pids[indices] == g_pids[:, np.newaxis]).astype(np.int32)
        # matches = (np.asarray(g_pids) == q_pids).astype(np.int32)

        for q_idx in range(num_q):
            g_pid = g_pids[q_idx]
            g_camid = g_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (q_pids[order] == g_pid) & (q_camids[order] == g_camid)
            keep = np.invert(remove)

            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when gallery identity does not appear in query set
                continue

            # Compute CMC
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            rank1 = cmc[:self.max_rank]
            self.rank1_list.append(rank1)

            self.valid_num_query+=1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            self.ap_list.append(AP)

    def eval_cuhksysu_galllery_stream2query_set(self, query_feats, gallery_feats,  q_pids, g_pids):
        ### Use predefined query set for gallery instance evaluation

        sim = gallery_feats @ query_feats.T

        sim = sim.detach().cpu().numpy()
        indices = np.argsort(-sim)
        num_q, num_g = sim.shape

        q_pids = np.asarray(q_pids)
        g_pids = np.asarray(g_pids)
        
        matches = (q_pids[indices] == g_pids[:, np.newaxis]).astype(np.int32)
        # matches = (np.asarray(g_pids) == q_pids).astype(np.int32)

        for q_idx in range(num_q):
            g_pid = g_pids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]

            orig_cmc = matches[q_idx]#[keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            # Compute CMC
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            rank1 = cmc[:self.max_rank]
            self.rank1_list.append(rank1)

            self.valid_num_query+=1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            self.ap_list.append(AP)

    def compute_false_matching_ratio(self, query_feats, query_ids, gallery_feats, gallery_ids, q_camids, g_camids, cross_cam_threshold=0.6,same_cam_threshold=0.6):
        """
        query_feats: [Nq, D] query set features
        query_ids:   [Nq]    query IDs (0~50)
        gallery_feats: [Ng, D] gallery stream features (non-query only)
        gallery_ids:   [Ng]    gallery stream original IDs
        """
        import numpy as np
        query_feats = query_feats.cpu().numpy()        # [50, 512]
        # query_ids = list(range(51))        # 0~50

        gallery_feats = gallery_feats.cpu().numpy()      # [100, 512] (non-query instance들)
        # gallery_ids = np.array(...)        # [100]

        # (Ng, Nq)
        sim = gallery_feats @ query_feats.T  # cosine similarity assumed normalized

        true_neg_cnts, false_pos_cnts = 0, 0
        same_cam_false_pos_cnts, same_cam_true_neg_cnts = 0,0
        num_g = len(gallery_feats)

        # np.array로 변환
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        q_camids = np.asarray(q_camids)
        g_camids = np.asarray(g_camids)

        query_num, non_query_num = 0, 0
        for g_idx in range(num_g):
            g_camid = g_camids[g_idx]
            g_id = gallery_ids[g_idx]
            sims = sim[g_idx]

            # query와 같은 ID가 아닌 경우만 확인
            if g_id not in query_ids:
                remove = (q_camids == g_camid) 
                keep = np.invert(remove)
                cross_cam_sims = sims[keep] ### Only select cross camera sims

                top1_same_cam_max_sim = np.max(sims)
                top1_cross_cam_sim = np.max(cross_cam_sims)
                
                self.rank1_similarities_non_query.append(top1_same_cam_max_sim)
                self.all_similarities_non_query.extend(sims)

                # 유사도가 threshold보다 큰 쿼리가 하나라도 있으면 false match
                if top1_cross_cam_sim >= cross_cam_threshold:
                    false_pos_cnts += 1
                    
                if top1_same_cam_max_sim>=same_cam_threshold:
                    same_cam_false_pos_cnts += 1
                    
                self.all_sims.append(top1_cross_cam_sim)
                self.all_labels.append(0)

                    
                non_query_num +=1
            
             # gallery ID가 query set 안에 있는 경우만 확인
            if g_id in query_ids:
                # cross-cam만 남기기 위한 mask
                remove = (q_camids == g_camid)
                keep = np.invert(remove)

                cross_cam_sims = sims[keep]  # cross-cam similarity만
                same_cam_max_sim = np.max(sims)  # same-cam 포함 전체에서 top-1

                # cross-cam 중 argmax
                max_idx_in_cross = np.argmax(cross_cam_sims)

                # cross-cam subset에서 얻은 idx → 원래 query idx로 변환
                cross_cam_indices = np.where(keep)[0]   # True인 query들의 원래 위치
                max_query_idx = cross_cam_indices[max_idx_in_cross]  

                # top-1 cross-cam similarity 값
                top1_cross_cam_sim = sims[max_query_idx]

                # ground-truth query idx (같은 ID인 쿼리 위치들)
                true_positive_idx = (g_id == query_ids).nonzero()[0]

                query_num += 1

                if top1_cross_cam_sim < cross_cam_threshold:
                    true_neg_cnts += 1

                if same_cam_max_sim < same_cam_threshold:
                    same_cam_true_neg_cnts += 1

                self.all_sims.append(top1_cross_cam_sim)

                # # ✅ 여기서 원래 query_ids[max_query_idx]와 비교
                # if g_id == query_ids[max_query_idx]:
                #     self.all_labels.append(1)
                # else:
                #     self.all_labels.append(0)
                    
                self.all_labels.append(1)
                    
                    
                query_true_camid = q_camids[true_positive_idx]
                for j in range(len(query_true_camid)):

                    if g_camid!=query_true_camid[j]:
                        self.rank1_similarities_cross_cam_query.append(sims[true_positive_idx][j].tolist()) 
                    else:
                        self.rank1_similarities_same_cam_query.append(sims[true_positive_idx][j].tolist()) 

                self.all_similarities_query.extend(sims[true_positive_idx].tolist())


        self.false_pos_cnts += false_pos_cnts
        self.true_neg_cnts += true_neg_cnts
        self.valid_num_non_query+= non_query_num

        self.same_cam_true_neg_cnts = same_cam_true_neg_cnts
        self.same_cam_false_pos_cnts = same_cam_false_pos_cnts

        # self.valid_num_query+= query_num

    def compute_cuhksysu_false_matching_ratio(self, query_feats, query_ids, gallery_feats, gallery_ids, q_camids, g_camids, cross_cam_threshold=0.6,same_cam_threshold=0.6):
        """
        query_feats: [Nq, D] query set features
        query_ids:   [Nq]    query IDs (0~50)
        gallery_feats: [Ng, D] gallery stream features (non-query only)
        gallery_ids:   [Ng]    gallery stream original IDs
        """
        import numpy as np
        query_feats = query_feats.cpu().numpy()        # [50, 512]
        # query_ids = list(range(51))        # 0~50

        gallery_feats = gallery_feats.cpu().numpy()      # [100, 512] (non-query instance들)
        # gallery_ids = np.array(...)        # [100]

        # (Ng, Nq)
        sim = gallery_feats @ query_feats.T  # cosine similarity assumed normalized

        true_neg_cnts, false_pos_cnts = 0, 0
        same_cam_false_pos_cnts, same_cam_true_neg_cnts = 0,0
        num_g = len(gallery_feats)

        # np.array로 변환
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        q_camids = np.asarray(q_camids)
        g_camids = np.asarray(g_camids)

        query_num, non_query_num = 0, 0
        for g_idx in range(num_g):
            g_camid = g_camids[g_idx]
            g_id = gallery_ids[g_idx]
            sims = sim[g_idx]

            # query와 같은 ID가 아닌 경우만 확인
            if g_id not in query_ids:
                # remove = (q_camids == g_camid) 
                # keep = np.invert(remove)
                # cross_cam_sims = sims[keep] ### Only select cross camera sims

                top1_max_sim = np.max(sims)
                # top1_cross_cam_sim = np.max(cross_cam_sims)
                
                self.rank1_similarities_non_query.append(top1_max_sim)
                self.all_similarities_non_query.extend(sims)

                # # 유사도가 threshold보다 큰 쿼리가 하나라도 있으면 false match
                # if top1_cross_cam_sim >= cross_cam_threshold:
                #     false_pos_cnts += 1
                    
                if top1_max_sim>=same_cam_threshold:
                    same_cam_false_pos_cnts += 1
                    
                self.all_sims.append(top1_max_sim)
                self.all_labels.append(0)

                    
                non_query_num +=1
            
             # gallery ID가 query set 안에 있는 경우만 확인
            if g_id in query_ids:
                same_cam_max_sim = np.max(sims)  # same-cam 포함 전체에서 top-1


                # # cross-cam subset에서 얻은 idx → 원래 query idx로 변환
                # cross_cam_indices = np.where(keep)[0]   # True인 query들의 원래 위치
                # max_query_idx = cross_cam_indices[max_idx_in_cross]  

                # # top-1 cross-cam similarity 값
                # top1_cross_cam_sim = sims[max_query_idx]

                # ground-truth query idx (같은 ID인 쿼리 위치들)
                true_positive_idx = (g_id == query_ids).nonzero()[0]

                query_num += 1

                # if top1_cross_cam_sim < cross_cam_threshold:
                #     true_neg_cnts += 1

                if same_cam_max_sim < same_cam_threshold:
                    same_cam_true_neg_cnts += 1

                self.all_sims.append(same_cam_max_sim)

                # # ✅ 여기서 원래 query_ids[max_query_idx]와 비교
                # if g_id == query_ids[max_query_idx]:
                #     self.all_labels.append(1)
                # else:
                #     self.all_labels.append(0)
                    
                self.all_labels.append(1)
                    
                    
                # query_true_camid = q_camids[true_positive_idx]
                # for j in range(len(query_true_camid)):

                #     if g_camid!=query_true_camid[j]:
                #         self.rank1_similarities_cross_cam_query.append(sims[true_positive_idx][j].tolist()) 
                #     else:
                #         self.rank1_similarities_same_cam_query.append(sims[true_positive_idx][j].tolist()) 

                # self.all_similarities_query.extend(sims[true_positive_idx].tolist())


        self.false_pos_cnts += false_pos_cnts
        self.true_neg_cnts += true_neg_cnts
        self.valid_num_non_query+= non_query_num

        self.same_cam_true_neg_cnts = same_cam_true_neg_cnts
        self.same_cam_false_pos_cnts = same_cam_false_pos_cnts