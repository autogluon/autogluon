import torch

from .enums import Task


def get_loss(task: Task):
    if task == Task.REGRESSION:
        return torch.nn.MSELoss()
    else:
        return torch.nn.CrossEntropyLoss()
