import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(hyperparams: dict, optimizer: torch.optim.Optimizer):
    if hyperparams["lr_scheduler"]:
        scheduler = ReduceLROnPlateau(optimizer, patience=hyperparams["lr_scheduler_patience"], min_lr=0, factor=0.2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=10000000, min_lr=0, factor=0.2)

    return scheduler
