import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.00035
        weight_decay = 0.000
        if "bias" in key:
            lr = 0.00035 * 1
            weight_decay = 0.00000
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim,'Adam')(params)
    return optimizer
