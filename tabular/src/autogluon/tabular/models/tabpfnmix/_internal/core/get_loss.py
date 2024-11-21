import torch

from .enums import Task


def get_loss(task: Task):

    match task:
        case Task.REGRESSION:
            return torch.nn.MSELoss()
        case Task.CLASSIFICATION:
            return torch.nn.CrossEntropyLoss()
