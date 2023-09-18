import contextlib
import logging
import math
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch import nn

from autogluon.common.utils.resource_utils import ResourceManager

from ..constants import DDP, OBJECT_DETECTION, OCR

logger = logging.getLogger(__name__)


def is_interactive_env():
    """
    Return whether the current process is running under the interactive mode.
    Check also https://stackoverflow.com/a/64523765
    """
    return hasattr(sys, "ps1")


def is_interactive_strategy(strategy: str):
    if strategy:
        return strategy.startswith(("ddp_fork", "ddp_notebook"))
    else:
        return False


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


def infer_precision(
    num_gpus: int, precision: Union[int, str], as_torch: Optional[bool] = False, cpu_only_warning: bool = True
):
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
    cpu_only_warning
        Whether to turn on warning if the instance has only CPU.

    Returns
    -------
    The inferred precision.
    """
    if num_gpus == 0:  # CPU only prediction
        if cpu_only_warning:
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


def check_if_packages_installed(problem_type: str = None, package_names: List[str] = None):
    """
    Check if necessary packages are installed for some problem types.
    Raise an error if an package can't be imported.

    Parameters
    ----------
    problem_type
        Problem type
    """
    if problem_type:
        problem_type = problem_type.lower()
        if any(p in problem_type for p in [OBJECT_DETECTION, OCR]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import mmcv
            except ImportError as e:
                raise ValueError(
                    f"Encountered error while importing mmcv: {e}. {_get_mmlab_installation_guide('mmcv')}"
                )

            try:
                import mmdet
            except ImportError as e:
                raise ValueError(
                    f"Encountered error while importing mmdet: {e}. {_get_mmlab_installation_guide('mmdet')}"
                )

            if OCR in problem_type:
                try:
                    import mmocr
                except ImportError as e:
                    raise ValueError(
                        f'Encountered error while importing mmocr: {e}. Try to install mmocr: pip install "mmocr<1.0".'
                    )

    if package_names:
        for package_name in package_names:
            if package_name == "mmcv":
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import mmcv
                    from mmcv import ConfigDict
                    from mmcv.runner import load_checkpoint
                    from mmcv.transforms import Compose
                except ImportError as e:
                    f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
            elif package_name == "mmdet":
                try:
                    import mmdet
                    from mmdet.datasets.transforms import ImageToTensor
                    from mmdet.registry import MODELS
                except ImportError as e:
                    f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
            elif package_name == "mmengine":
                try:
                    import mmengine
                    from mmengine.dataset import pseudo_collate as collate
                    from mmengine.runner import load_checkpoint
                except ImportError as e:
                    warnings.warn(e)
                    raise ValueError(
                        f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
                    )
            else:
                raise ValueError(f"package_name {package_name} is not required.")


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
            devices = find_usable_cuda_devices(num_gpus)
        else:
            devices = num_gpus
    else:
        devices = "auto"

    return devices


def _get_mmlab_installation_guide(package_name):
    if package_name == "mmdet":
        err_msg = 'Please install MMDetection by: pip install "mmdet>=3.0.0".'
    elif package_name == "mmcv":
        err_msg = "Please install MMCV by: mim install mmcv"
    elif package_name == "mmengine":
        err_msg = "Please install MMCV by: mim install mmengine"
    else:
        raise ValueError("Available package_name are: mmdet, mmcv, mmengine.")

    return err_msg
