import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch import nn

from autogluon.common.utils.resource_utils import ResourceManager

from .env import is_interactive_env

logger = logging.getLogger(__name__)


def compute_num_gpus(config_num_gpus: Union[int, float, List], accelerator: str):
    """
    Compute the gpu number to initialize the lightning trainer.

    Parameters
    ----------
    config_num_gpus
        The gpu number provided by config.
    accelerator
        # "cpu", "gpu", or "auto".

    Returns
    -------
    A valid gpu number for the current environment and config.
    """
    if isinstance(accelerator, str) and accelerator.lower() not in ["gpu", "auto"]:
        return 0

    config_num_gpus = (
        math.floor(config_num_gpus) if isinstance(config_num_gpus, (int, float)) else len(config_num_gpus)
    )
    detected_num_gpus = ResourceManager.get_gpu_count_torch()

    if config_num_gpus < 0:  # In case config_num_gpus is -1, meaning using all gpus.
        num_gpus = detected_num_gpus
    else:
        num_gpus = min(config_num_gpus, detected_num_gpus)
        if detected_num_gpus < config_num_gpus:
            warnings.warn(
                f"Using the detected GPU number {detected_num_gpus}, "
                f"smaller than the GPU number {config_num_gpus} in the config.",
                UserWarning,
            )

    return num_gpus


def move_to_device(obj: Union[torch.Tensor, nn.Module, Dict, List, Tuple], device: torch.device):
    """
    Move an object to the given device.

    Parameters
    ----------
    obj
        An object, which can be a tensor, a module, a dict, or a list.
    device
        A Pytorch device instance.

    Returns
    -------
    The object on the device.
    """
    if not isinstance(device, torch.device):
        raise ValueError(f"Invalid device: {device}. Ensure the device type is `torch.device`.")

    if torch.is_tensor(obj) or isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list) or isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        raise TypeError(
            f"Invalid type {type(obj)} for move_to_device. "
            f"Make sure the object is one of these: a Pytorch tensor, a Pytorch module, "
            f"a dict or list of tensors or modules."
        )


def get_available_devices(num_gpus: int, auto_select_gpus: bool):
    """
    Get the available devices.

    Parameters
    ----------
    num_gpus
        Number of GPUs.
    auto_select_gpus
        Whether to pick GPU indices that are "accessible". See here: https://github.com/Lightning-AI/lightning/blob/accd2b9e61063ba3c683764043030545ed87c71f/src/lightning/fabric/accelerators/cuda.py#L79

    Returns
    -------
    The available devices.
    """
    if num_gpus > 0:
        if auto_select_gpus:
            if is_interactive_env():
                devices = list(range(num_gpus))
            else:
                devices = find_usable_cuda_devices(num_gpus)
        else:
            devices = num_gpus
    else:
        devices = "auto"

    return devices
