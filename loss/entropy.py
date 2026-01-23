import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch import Tensor


def dist(query_features: Tensor,
         gallery_features: Tensor,
         metric: str = "cosine",
         top_k:int= 0,
         selection:str = 'top-k',
         on_gpu: bool = True) -> Tensor:
    
    Nq, Ng = query_features.shape[0], gallery_features.shape[0]
    q_idx, g_idx = torch.meshgrid(torch.arange(Nq), torch.arange(Ng), indexing="ij")

    if not on_gpu:
        query_features = query_features.cpu()
        gallery_features = gallery_features.cpu()

    # (Nq,Ng)
    match metric:
        case "sqeuclidean":
            dists = (query_features[q_idx] - gallery_features[g_idx]).square().sum(dim=2)
        case "prod":
            dists = -(query_features[q_idx] * gallery_features[g_idx]).sum(dim=2)
        case "cosine":
            qf_norm: Tensor = query_features / query_features.norm(dim=1, keepdim=True)
            gf_norm: Tensor = gallery_features / gallery_features.norm(dim=1, keepdim=True)
            dists = -(qf_norm[q_idx] * gf_norm[g_idx]).sum(dim=2)
        case _:
            raise ValueError(f"Invalid distance metric: {metric!r}")
        
    if top_k:
        match selection:
            case "top-k":
                dists = dists.topk(top_k, dim=1, largest=False)[0]    # (Nq,k)
            case "bottom-k":
                dists = dists.topk(top_k, dim=1, largest=True)[0]    # (Nq,k)
            case "top-bottom":
                top_dists = dists.topk(
                    top_k // 2, dim=1, largest=False)[0]   # (Nq, k//2)
                btm_dists = dists.topk(
                    top_k // 2, dim=1, largest=True)[0]    # (Nq, k//2)
                dists = torch.cat([top_dists, btm_dists], dim=1)   # (Nq, k)
            case "random":
                idx = np.stack([
                    np.random.choice(Ng, size=top_k, replace=False)
                    for _ in range(Nq)
                ])
                idx = torch.from_numpy(idx).long()  # (Nq,k)
                dim = torch.arange(Nq)
                dists = dists[:, idx][dim, dim]
            case _:
                raise ValueError(f"Invalid selection: {selection!r}")
        
    return dists.cuda()


@dataclass
class Entropy(nn.Module):
    metric: str
    on_gpu: bool = True
    top_k: int = 0
    selection: str = 'top-k'

    def __post_init__(self):
        super().__init__()

    def forward(self, query_features:Tensor, gallery_features:Tensor) -> Tensor:
        dists = dist(query_features, gallery_features, self.metric, self.top_k, self.selection, self.on_gpu)
        prob = F.softmax(-dists, dim=1)
        # prob = F.softmax((-dists) / 0.07, dim=1)
        mean_ent = (-prob * F.log_softmax(-dists, dim=1)).sum(dim=1).mean()
        return mean_ent

