import numpy as np

from autogluon.common.utils.try_import import try_import_torch

from ..constants import BINARY
from ..data.label_cleaner import LabelCleanerMulticlassToBinary


def tune_temperature_scaling(y_val_probs: np.ndarray, y_val: np.ndarray, init_val: float = 1, max_iter: int = 1000, lr: float = 0.01):
    """
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
    """
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
        new_logits = logits / temp
        loss = nll_criterion(new_logits, y_val_tensor)
        loss.backward()
        scheduler.step()
        return loss

    optimizer.step(temperature_scale_step)

    temperature_scale = temperature_param.item()
    if np.isnan(temperature_scale):
        return None

    return temperature_scale


def custom_softmax(logits: np.ndarray) -> np.ndarray:
    x_max = np.amax(logits, axis=1, keepdims=True)
    exp_x_shifted = np.exp(logits - x_max)
    y_pred_proba = exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
    return y_pred_proba


def apply_temperature_scaling(y_pred_proba: np.ndarray, temperature_scalar: float, problem_type: str) -> np.ndarray:
    # TODO: This is expensive to convert at inference time, try to avoid in future
    if problem_type == BINARY:
        y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)

    logits = np.log(y_pred_proba)
    y_pred_proba = custom_softmax(logits=logits / temperature_scalar)

    if problem_type == BINARY:
        y_pred_proba = y_pred_proba[:, 1]

    return y_pred_proba
