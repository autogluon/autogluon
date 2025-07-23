import contextlib
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def convert_to_torch_precision(precision: Union[int, str]):
    """
    Convert a precision integer or string to the corresponding torch precision.

    Parameters
    ----------
    precision
    a precision integer or string from the config.

    Returns
    -------
    A torch precision object.
    """
    precision_mapping = {
        16: torch.half,
        "16": torch.half,
        "16-mixed": torch.half,
        "16-true": torch.half,
        "bf16": torch.bfloat16,
        "bf16-mixed": torch.bfloat16,
        "bf16-true": torch.bfloat16,
        32: torch.float32,
        "32": torch.float32,
        "32-true": torch.float32,
        64: torch.float64,
        "64": torch.float64,
        "64-true": torch.float64,
    }

    if precision in precision_mapping:
        precision = precision_mapping[precision]
    else:
        raise ValueError(f"Unknown precision: {precision}")

    return precision


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
        precision = 32  # Force to use fp32 for training since 16-mixed is not available in CPU
    else:
        if isinstance(precision, str) and "bf16" in precision and not torch.cuda.is_bf16_supported():
            warnings.warn(
                f"{precision} is not supported by the GPU device / cuda version. "
                "Consider using GPU devices with versions after Amphere or upgrading cuda to be >=11.0. "
                f"MultiModalPredictor is switching precision from {precision} to 32.",
                UserWarning,
            )
            precision = 32

    if as_torch:
        precision = convert_to_torch_precision(precision=precision)

    return precision


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
    precision = convert_to_torch_precision(precision=precision)

    if precision in [torch.half, torch.float16, torch.bfloat16]:
        return torch.autocast(device_type=device_type, dtype=precision)
    if precision == torch.float32:
        assert torch.get_default_dtype() == torch.float32
        return contextlib.nullcontext()
    elif precision == torch.float64:
        return double_precision_context()
    else:
        raise ValueError(f"Unknown precision: {precision}")
