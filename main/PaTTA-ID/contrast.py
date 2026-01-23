import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# class ViewContrastiveLoss(nn.Module):
#     def __init__(self, num_instance=4, T=1.0):
#         super(ViewContrastiveLoss, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()
#         # self.num_instance = num_instance
#         self.T = T
# 
#     def forward(self, q, k, label):
#         batchSize = q.shape[0]
#         N = q.size(0)
#         mat_sim = torch.matmul(q, k.transpose(0, 1))
#         mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()
#         # batch hard
#         hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq, True)
#         l_pos = hard_p.view(batchSize, 1)
#         mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
#         # positives = torch.masked_select(mat_sim, mat_eq).view(-1, 1)
#         negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
#         out = torch.cat((l_pos, negatives), dim=1) / self.T
#         # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
#         targets = torch.zeros([batchSize]).cuda().long()
#         triple_dist = F.log_softmax(out, dim=1)
#         triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
#         # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
#         loss = (- triple_dist_ref * triple_dist).mean(0).sum()
#         return loss
# 
#     def batch_hard(self, mat_sim, mat_eq, indice=False):
#         sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
#                                                            descending=False)
#         hard_p = sorted_mat_sim[:, 0]
#         hard_p_indice = positive_indices[:, 0]
#         sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
#                                                            descending=True)
#         hard_n = sorted_mat_distance[:, 0]
#         hard_n_indice = negative_indices[:, 0]
#         if (indice):
#             return hard_p, hard_n, hard_p_indice, hard_n_indice
#         return hard_p, hard_n

class ViewContrastiveLoss(nn.Module):
    def __init__(self, T=1.0, top_k=50, hardest=0):
        super(ViewContrastiveLoss, self).__init__()
        self.T = T
        self.top_k = top_k
        self.hardest = hardest
        print("self.hardest == ",self.hardest)

    def forward(self, gallery_feat, query_feats, gallery_label, query_labels, weight=None):
        """
        gallery_feat: [D]
        query_feats: [N, D]
        gallery_label: scalar
        query_labels: [N] or list
        """
        if isinstance(query_labels, list):
            query_labels = torch.tensor(query_labels).to(gallery_feat.device)

        gallery_feat = gallery_feat.to(query_feats.device)
        gallery_label = gallery_label.to(query_feats.device)

        # (1) similarity 계산
        sim = gallery_feat @ query_feats.t()   # [N]


        # (2) positive/negative 마스크
        pos_mask = (query_labels == gallery_label)
        neg_mask = (query_labels != gallery_label)

        num_positives = pos_mask.sum().item()
        if num_positives == 0:
            return torch.tensor(0.0, requires_grad=True).to(gallery_feat.device)

        pos_sims = sim[pos_mask]            # [#pos]
        neg_sims = sim[neg_mask]            # [#neg]

        if self.hardest:
            hard_pos_idxs = torch.sort(pos_sims,descending=False)[1][:1]
        else:
            hard_pos_idxs = torch.sort(pos_sims,descending=False)[1][:]

        hard_pos_sims = pos_sims[hard_pos_idxs]
        hard_pos_count = hard_pos_sims.shape[0]


        # (3) Top-K Negative selection
        if neg_sims.numel() > self.top_k:
            topk_vals, _ = torch.topk(neg_sims, self.top_k, largest=True, sorted=False)
            neg_sims = topk_vals  # [top_k]

        # (4) 로그 소프트맥스 구성
        sim_concat = torch.cat([hard_pos_sims, neg_sims], dim=0) / self.T  # [P+K]
        sim_max = sim_concat.max().detach()
        sim_stable = sim_concat - sim_max

        exp_sim = torch.exp(sim_stable)
        log_prob = sim_stable - torch.log(exp_sim.sum())

        # (5) positive만 loss로 사용
        # loss = - log_prob[:num_positives].mean()
        loss = - log_prob[:hard_pos_count].mean()


        return loss