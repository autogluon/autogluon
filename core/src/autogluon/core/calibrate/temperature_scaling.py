import numpy as np

from ..utils import try_import_torch


def tune_temperature_scaling(y_val_probs: np.ndarray, y_val: np.ndarray, init_val: float = 1, max_iter: int = 1000,
                             lr: float = 0.01):
    '''
    Tunes a temperature scalar term that divides the logits produced by autogluon model. Logits are generated
    by natural log the predicted probs from model then divides by a temperature scalar, which is tuned
    to minimize cross entropy on validation set.

    Parameters:
    -----------
    y_val_probs: numpy ndarray
        Predictive probabilities by model on validation set
    y_val: numpy ndarray
        The labels to the validation set
    init_val: float
        Initial value for temperature scalar term
    max_iter: int
        The maximum number of iterations to step in tuning
    lr: float
        The initial learning rate

    Return:
    float: The temperature scaling term, returns None if infinity found in logits.
    '''
    try_import_torch()
    import torch

    # This is required to avoid error when passing np.uint16 to torch.tensor. This can occur if >255 classes (Dionis)
    y_val = y_val.astype(np.int64)

    y_val_tensor = torch.tensor(y_val)
    temperature_param = torch.nn.Parameter(torch.ones(1).fill_(init_val))
    logits = torch.tensor(np.log(y_val_probs))

    # TODO: Could alternatively add epsilon to y_val_probs in order to avoid.
    is_invalid = torch.isinf(logits).any().tolist()
    if is_invalid:
        return None

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

    temperature_scale = temperature_param.item()
    if np.isnan(temperature_scale):
        return None

    return temperature_scale
