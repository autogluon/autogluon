import datetime
import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import pytz
import torch

from autogluon.common.utils.system_info import get_ag_system_info

from .. import version as ag_version

logger = logging.getLogger(__name__)


def make_exp_dir(
    root_path: str,
    job_name: str,
    create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.,: root_path/2022_01_01/job_name_12_00_00/
    This function is to better organize the training runs. It is recommended to call this
    function and pass the returned "exp_dir" to "MultiModalPredictor.fit(save_path=exp_dir)".

    Parameters
    ----------
    root_path
        The basic path where to create saving directories for training runs.
    job_name
        The job names to name training runs.
    create
        Whether to make the directory.

    Returns
    -------
    The formatted directory path.
    """
    tz = pytz.timezone("US/Pacific")
    ct = datetime.datetime.now(tz=tz)
    date_stamp = ct.strftime("%Y_%m_%d")
    time_stamp = ct.strftime("%H_%M_%S")

    # Group logs by day first
    exp_dir = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    exp_dir = os.path.join(exp_dir, "_".join([job_name, time_stamp]))

    if create:
        os.makedirs(exp_dir, mode=0o777, exist_ok=False)

    return exp_dir


class LogFilter(logging.Filter):
    """
    Filter log messages with patterns.
    """

    def __init__(self, blacklist: Union[str, List[str]]):
        """
        Parameters
        ----------
        blacklist
            Patterns to be suppressed in logging.
        """
        super().__init__()
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        self._blacklist = blacklist

    def filter(self, record):
        """
        Check whether to suppress a logging message.

        Parameters
        ----------
        record
            A logging message.

        Returns
        -------
        If True, no pattern exists in the message, hence printed out.
        If False, some pattern is in the message, hence filtered out.
        """
        matches = [pattern not in record.msg for pattern in self._blacklist]
        return all(matches)


def add_log_filter(target_logger, log_filter):
    """
    Add one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.addFilter(log_filter)


def remove_log_filter(target_logger, log_filter):
    """
    Remove one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.removeFilter(log_filter)


@contextmanager
def apply_log_filter(log_filter):
    """
    User contextmanager to control the scope of applying one log filter.
    Currently, it is to filter some lightning's log messages.
    But we can easily extend it to cover more loggers.

    Parameters
    ----------
    log_filter
        Log filter.
    """
    try:
        add_log_filter(logging.getLogger(), log_filter)
        add_log_filter(logging.getLogger("lightning"), log_filter)
        add_log_filter(logging.getLogger("lightning.pytorch"), log_filter)
        yield

    finally:
        remove_log_filter(logging.getLogger(), log_filter)
        remove_log_filter(logging.getLogger("lightning"), log_filter)
        remove_log_filter(logging.getLogger("lightning.pytorch"), log_filter)


def on_fit_start_message(path: Optional[str] = None):
    return get_ag_system_info(
        path=path,
        include_gpu_count=False,
        include_pytorch=True,
        include_cuda=True,
    )


def on_fit_per_run_start_message(save_path, validation_metric_name):
    return f"""\

AutoMM starts to create your model. ✨✨✨

To track the learning progress, you can open a terminal and launch Tensorboard:
    ```shell
    # Assume you have installed tensorboard
    tensorboard --logdir {save_path}
    ```
"""


def on_fit_end_message(save_path):
    return f"""\
AutoMM has created your model. 🎉🎉🎉

To load the model, use the code below:
    ```python
    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor.load("{save_path}")
    ```

If you are not satisfied with the model, try to increase the training time, 
adjust the hyperparameters (https://auto.gluon.ai/stable/tutorials/multimodal/advanced_topics/customization.html),
or post issues on GitHub (https://github.com/autogluon/autogluon/issues).

"""


def get_gpu_message(detected_num_gpus: int, used_num_gpus: int, strategy: str):
    """
    Get the GPU related info (GPU name, total memory, free memory, and CUDA version) for logging.

    Parameters
    ----------
    detected_num_gpus
        Number of detected GPUs.
    used_num_gpus
        Number of GPUs to be used.

    Returns
    -------
    A string with the GPU info.
    """

    def _bytes_to_gigabytes(bytes):
        return round((bytes / 1024) / 1024 / 1024, 2)

    gpu_message = f"GPU Count: {detected_num_gpus}\nGPU Count to be Used: {used_num_gpus}\n"
    try:
        import nvidia_smi
    except:
        return gpu_message

    for i in range(detected_num_gpus):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        gpu_mem_used = _bytes_to_gigabytes(info.used)
        gpu_mem_total = _bytes_to_gigabytes(info.total)
        if torch.cuda.is_available():
            gpu_message += f"GPU {i} Name: {torch.cuda.get_device_name(i)}\n"
        gpu_message += f"GPU {i} Memory: {gpu_mem_used}GB/{gpu_mem_total}GB (Used/Total)\n"

    return gpu_message
