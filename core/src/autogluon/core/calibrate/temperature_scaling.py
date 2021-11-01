import numpy as np

from ..utils import try_import_torch


def tune_temperature_scaling(y_val_probs, y_val, init_val, max_iter, lr):
    try_import_torch()
    import torch

    y_val_tensor = torch.tensor(y_val)
    temperature_param = torch.nn.Parameter(torch.ones(1).fill_(init_val))
    logits = torch.tensor(np.log(y_val_probs))
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature_param], lr=lr, max_iter=max_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def temperature_scale_step():
        optimizer.zero_grad()
        temp = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
        new_logits = (logits / temp)
        loss = nll_criterion(new_logits, y_val_tensor)
        loss.backward()
        scheduler.step()
        return loss

    optimizer.step(temperature_scale_step)

    return temperature_param.item()
