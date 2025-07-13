import torch
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from transformers import get_constant_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from ..._internal.config.config_pretrain import ConfigPretrain


def get_scheduler(hyperparams: dict, optimizer: torch.optim.Optimizer) -> tuple[torch.optim.lr_scheduler.LambdaLR, ReduceLROnPlateau]:

    warmup_steps = hyperparams['warmup_steps']

    # if warmup_steps > 0:
    #     scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lambda step: min((step + 1) / warmup_steps, 1.0)
    #     )
    # else:
    #     scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lambda step: 1.0
    #     )
    
    if warmup_steps > 0:
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    else:
        scheduler_warmup = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    if hyperparams['lr_scheduler']:      
        scheduler_reduce_on_plateau = ReduceLROnPlateau(
            optimizer, 
            patience=hyperparams['lr_scheduler_patience'], 
            min_lr=0, 
            factor=0.2
        )
    else:
        # With ReduceLROnPlateau, the scheduler accepts a metric to monitor, so our dummy metric must also be a ReduceLRonPlateau scheduler
        scheduler_reduce_on_plateau = ReduceLROnPlateau(
            optimizer, 
            patience=1000000000, 
            min_lr=0, 
            factor=0.2
        )

    return scheduler_warmup, scheduler_reduce_on_plateau


def get_scheduler_pretrain(cfg: ConfigPretrain, optimizer: torch.optim.Optimizer):

    
    if cfg.optim.cosine_scheduler:
        schedule = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.warmup_steps,
            num_training_steps=cfg.optim.steps,
            min_lr_rate=0.1
        )
    else:
        schedule = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.warmup_steps
        )

    return schedule