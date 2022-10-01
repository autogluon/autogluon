import contextlib
import logging
import math
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

try:
    from mmcv.parallel import DataContainer
except:
    pass

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


def is_interactive():
    """
    Return whether the current process is running under the interactive mode.
    Check also https://stackoverflow.com/a/64523765
    """
    return hasattr(sys, "ps1")


def compute_num_gpus(config_num_gpus: Union[int, float, List], strategy: str):
    """
    Compute the gpu number to initialize the lightning trainer.

    Parameters
    ----------
    config_num_gpus
        The gpu number provided by config.
    strategy
        A lightning trainer's strategy such as "ddp", "ddp_spawn", and "dp".

    Returns
    -------
    A valid gpu number for the current environment and config.
    """
    config_num_gpus = (
        math.floor(config_num_gpus) if isinstance(config_num_gpus, (int, float)) else len(config_num_gpus)
    )
    detected_num_gpus = torch.cuda.device_count()

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

    if is_interactive() and num_gpus > 1 and strategy in ["ddp", "ddp_spawn"]:
        warnings.warn(
            "Interactive environment is detected. Currently, MultiModalPredictor does not support multi-gpu "
            "training under an interactive environment due to the limitation of ddp / ddp_spawn strategies "
            "in PT Lightning. Thus, we switch to single gpu training. For multi-gpu training, you need to execute "
            "MultiModalPredictor in a script.",
            UserWarning,
        )
        num_gpus = 1

    return num_gpus


def infer_precision(num_gpus: int, precision: Union[int, str], as_torch: Optional[bool] = False):
    """
    Infer the proper precision based on the environment setup and the provided precision.

    Parameters
    ----------
    num_gpus
        GPU number.
    precision
        The precision provided in config.
    as_torch
        Whether to convert the precision to the Pytorch format.

    Returns
    -------
    The inferred precision.
    """
    if num_gpus == 0:  # CPU only prediction
        warnings.warn(
            "Only CPU is detected in the instance. "
            "This may result in slow speed for MultiModalPredictor. "
            "Consider using an instance with GPU support.",
            UserWarning,
        )
        precision = 32  # Force to use fp32 for training since fp16-based AMP is not available in CPU
    else:
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            warnings.warn(
                "bf16 is not supported by the GPU device / cuda version. "
                "Consider using GPU devices with versions after Amphere or upgrading cuda to be >=11.0. "
                "MultiModalPredictor is switching precision from bf16 to 32.",
                UserWarning,
            )
            precision = 32

    if as_torch:
        precision_mapping = {
            16: torch.float16,
            "bf16": torch.bfloat16,
            32: torch.float32,
            64: torch.float64,
        }
        if precision in precision_mapping:
            precision = precision_mapping[precision]
        else:
            raise ValueError(f"Unknown precision: {precision}")

    return precision


def move_to_device(obj: Union[torch.Tensor, nn.Module, Dict, List], device: torch.device):
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
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    elif isinstance(obj, DataContainer):
        return obj
    else:
        raise TypeError(
            f"Invalid type {type(obj)} for move_to_device. "
            f"Make sure the object is one of these: a Pytorch tensor, a Pytorch module, "
            f"a dict or list of tensors or modules."
        )


def compute_inference_batch_size(
    per_gpu_batch_size: int,
    eval_batch_size_ratio: Union[int, float],
    per_gpu_batch_size_evaluation: int,
    num_gpus: int,
    strategy: str,
):
    """
    Compute the batch size for inference.

    Parameters
    ----------
    per_gpu_batch_size
        Per gpu batch size from the config.
    eval_batch_size_ratio
        per_gpu_batch_size_evaluation = per_gpu_batch_size * eval_batch_size_ratio.
    per_gpu_batch_size_evaluation
        Per gpu evaluation batch size from the config.
    num_gpus
        Number of GPUs.
    strategy
        A pytorch lightning strategy.

    Returns
    -------
    Batch size for inference.
    """
    if per_gpu_batch_size_evaluation:
        batch_size = per_gpu_batch_size_evaluation
    else:
        batch_size = per_gpu_batch_size * eval_batch_size_ratio

    if num_gpus > 1 and strategy == "dp":
        # If using 'dp', the per_gpu_batch_size would be split by all GPUs.
        # So, we need to use the GPU number as a multiplier to compute the batch size.
        batch_size = batch_size * num_gpus

    return batch_size


@contextlib.contextmanager
def double_precision_context():
    """
    Double precision context manager.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(default_dtype)


def get_precision_context(precision: Union[int, str], device_type: Optional[str] = None):
    """
    Choose the proper context manager based on the precision.

    Parameters
    ----------
    precision
        The precision.
    device_type
        gpu or cpu.

    Returns
    -------
    A precision context manager.
    """
    if precision == 32:
        assert torch.get_default_dtype() == torch.float32
        return contextlib.nullcontext()
    elif precision in [16, "bf16"]:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16 if precision == "bf16" else torch.half)
    elif precision == 64:
        return double_precision_context()
    else:
        raise ValueError(f"Unknown precision: {precision}")
