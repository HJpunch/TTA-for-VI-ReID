import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def assign_pseudo_labels_from_memory(
        self,
        mem_feats,        # (M, D) normalized
        mem_flags,        # (M,) 1=query, 0=gallery
        mem_pids,         # (M,) query: GT pid, gallery: old pseudo or -1
        sim_th=0.65,
        start_pid=10000
    ):
    """
    Memory-centric clustering
    """
    M = mem_feats.size(0)
    new_pids = mem_pids.clone()

    sim_matrix = mem_feats @ mem_feats.t()
    next_pid = start_pid

    for i in range(M):
        if mem_flags[i] == 1:
            continue  # query는 GT anchor

        sims = sim_matrix[i]
        idxs = (sims >= sim_th).nonzero().flatten()

        assigned = False

        # # (1) query anchor 우선
        # for j in idxs:
        #     if mem_flags[j] == 1:
        #         new_pids[i] = mem_pids[j]
        #         assigned = True
        #         break

        if assigned:
            continue

        # (2) gallery cluster merge
        for j in idxs:
            if new_pids[j] >= start_pid:
                new_pids[i] = new_pids[j]
                assigned = True
                break

        if not assigned:
            new_pids[i] = next_pid
            next_pid += 1

    return new_pids
 
def update_model(self, iter, model, optimizer):

    memory_data, memory_pids, memory_flags = self.mem.get_memory()
    if len(memory_data) < 2:
        return


    model.eval()
    imgs = torch.stack(memory_data).cuda()
    data_pids = torch.tensor(memory_pids).cuda()
    
    q_data = imgs[(torch.tensor(memory_flags)==1)]
    q_pids = data_pids[(torch.tensor(memory_flags)==1)]

    with torch.no_grad():
        mem_feats = model(imgs)
        mem_feats = F.normalize(mem_feats, dim=1)

    mem_pids = torch.tensor(memory_pids).cuda()
    mem_flags = torch.tensor(memory_flags).cuda()

    # 🔥 Memory-centric clustering
    new_pids = self.assign_pseudo_labels_from_memory(
        mem_feats,
        mem_flags,
        mem_pids,
        sim_th=self.thres
    )

    model.train()


    dataset = torch.utils.data.TensorDataset(imgs, new_pids)
    data_loader = DataLoader(dataset, batch_size=len(imgs),
                            shuffle=True, drop_last =False, pin_memory=False)

    for batch_idx, (feats, pids) in enumerate(data_loader):
        loss_tri = self.step(iter, model=model,  data=feats,  memory_pids=pids)


    loss_tri = self.vcl_loss_w * loss_tri

    optimizer.zero_grad()
    loss_tri.backward()
    optimizer.step()

    print("Camera {} Stream Iter=[{}/{}] loss_tri == {:.2f} | mem size={}, query-num-{}".format(
            self.cam_id, iter, self.total_iter, loss_tri.item(), len(memory_data), len(q_data)))