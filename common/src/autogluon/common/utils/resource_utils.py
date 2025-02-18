import logging
import multiprocessing
import os
import shutil
import subprocess
from typing import Union

from autogluon.common.utils.try_import import try_import_ray

from .distribute_utils import DistributedContext
from .lite import disable_if_lite_mode
from .utils import bytes_to_mega_bytes


class ResourceManager:
    """Manager that fetches system related info"""

    @staticmethod
    def get_cpu_count():
        return multiprocessing.cpu_count()

    @staticmethod
    @disable_if_lite_mode(ret=1)
    def get_cpu_count_psutil(logical=True):
        import psutil

        return psutil.cpu_count(logical=logical)

    @staticmethod
    @disable_if_lite_mode(ret=0)
    def get_gpu_count() -> int:
        num_gpus = ResourceManager._get_gpu_count_cuda()
        if num_gpus == 0:
            num_gpus = ResourceManager.get_gpu_count_torch()
        return num_gpus

    @staticmethod
    def get_gpu_count_torch() -> int:
        try:
            import torch

            if not torch.cuda.is_available():
                num_gpus = 0
            else:
                num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 0
        return num_gpus

    @staticmethod
    def get_gpu_free_memory():
        """Grep gpu free memory from nvidia-smi tool.
        This function can fail due to many reasons(driver, nvidia-smi tool, envs, etc) so please simply use
        it as a suggestion, stay away with any rules bound to it.
        E.g. for a 4-gpu machine, the result can be list of int
        >>> print(get_gpu_free_memory)
        >>> [13861, 13859, 13859, 13863]
        """
        _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

        try:
            COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        except:
            memory_free_values = []
        return memory_free_values

    @staticmethod
    def get_memory_size(format: str = "B") -> float:
        """

        Parameters
        ----------
        format: {"B", "KB", "MB", "GB", "TB", "PB"}

        Returns
        -------
        Memory size in the provided `format`.

        """
        bytes = ResourceManager._get_memory_size()
        return ResourceManager.bytes_converter(value=bytes, format_in="B", format_out=format)

    @staticmethod
    def get_memory_rss(format: str = "B") -> float:
        bytes = ResourceManager._get_memory_rss()
        return ResourceManager.bytes_converter(value=bytes, format_in="B", format_out=format)

    @staticmethod
    def get_available_virtual_mem(format: str = "B") -> float:
        bytes = ResourceManager._get_available_virtual_mem()
        return ResourceManager.bytes_converter(value=bytes, format_in="B", format_out=format)

    @staticmethod
    def bytes_converter(value: float, format_in: str, format_out: str) -> float:
        """
        Converts bytes `value` from `format_in` to `format_out`.

        Parameters
        ----------
        value: float
        format_in: {"B", "KB", "MB", "GB", "TB", "PB"}
        format_out: {"B", "KB", "MB", "GB", "TB", "PB"}

        Returns
        -------
        value in `format_out` format.
        """
        valid_formats = ["B", "KB", "MB", "GB", "TB", "PB"]
        assert format_in in valid_formats
        assert format_out in valid_formats
        bytes = value
        for format in valid_formats:
            if format_in == format:
                break
            bytes *= 1024
        output = bytes
        for format in valid_formats:
            if format_out == format:
                break
            output /= 1024
        return output

    @staticmethod
    @disable_if_lite_mode(ret=None)
    def get_process(pid=None):
        import psutil

        return psutil.Process(pid)

    @staticmethod
    def get_available_disk_size():
        # FIXME: os.statvfs doesn't work on Windows...
        # Need to find another way to calculate disk on Windows.
        # Return None for now
        try:
            statvfs = os.statvfs(".")
            available_blocks = statvfs.f_frsize * statvfs.f_bavail
            return bytes_to_mega_bytes(available_blocks)
        except Exception:
            return None

    @staticmethod
    def get_disk_usage(path: str):
        """
        Gets the disk usage information for the given path

        Returns obj with variables `free`, `total`, `used`, representing bytes as integers.
        """
        return shutil.disk_usage(path=path)

    @staticmethod
    def _get_gpu_count_cuda():
        # FIXME: Sometimes doesn't detect GPU on Windows
        # FIXME: Doesn't ensure the GPUs are actually usable by the model (PyTorch, etc.)
        from .nvutil import cudaDeviceGetCount, cudaInit, cudaShutdown

        if not cudaInit():
            return 0
        gpu_count = cudaDeviceGetCount()
        cudaShutdown()
        return gpu_count

    @staticmethod
    def _get_custom_memory_size():
        memory_limit = float(os.environ.get("AG_MEMORY_LIMIT_IN_GB"))

        if memory_limit <= 0:
            raise ValueError("Memory set via `AG_MEMORY_LIMIT_IN_GB` must be greater than 0!")

        # Transform to bytes and return
        return max(int(memory_limit * (1024.0**3)), 1)

    @staticmethod
    @disable_if_lite_mode(ret=1073741824)  # set to 1GB as an empirical value in lite/web-browser mode.
    def _get_memory_size() -> float:
        if os.environ.get("AG_MEMORY_LIMIT_IN_GB", None) is not None:
            return ResourceManager._get_custom_memory_size()

        import psutil

        return psutil.virtual_memory().total

    @staticmethod
    @disable_if_lite_mode(ret=1073741824)  # set to 1GB as an empirical value in lite/web-browser mode.
    def _get_memory_rss() -> float:
        return ResourceManager.get_process().memory_info().rss

    @staticmethod
    @disable_if_lite_mode(ret=1073741824)  # set to 1GB as an empirical value in lite/web-browser mode.
    def _get_available_virtual_mem() -> float:
        import psutil

        if os.environ.get("AG_MEMORY_LIMIT_IN_GB", None) is not None:
            total_memory = ResourceManager._get_custom_memory_size()
            p = psutil.Process()
            return total_memory - p.memory_info().rss

        return psutil.virtual_memory().available


class RayResourceManager:
    """Manager that fetches ray cluster resources info. This class should only be used within a ray cluster."""

    @staticmethod
    def _init_ray():
        """Initialize ray runtime if not already initialized. Will force the existence of a cluster already being spinned up"""
        try_import_ray()
        import ray

        if not ray.is_initialized():
            ray.init(
                address="auto",  # Force ray to connect to an existing cluster. There should be one. Otherwise, something went wrong
                log_to_driver=False,
                logging_level=logging.ERROR,
            )

    @staticmethod
    def _get_cluster_resources(key: str, default_val: Union[int, float] = 0):
        """
        Get value of resources available in the cluster.

        Parameter
        ---------
        key: str
            The key of the value you want to get, i.e. CPU
        default_val: Union[int, float]
            Default value to get if key not available in the cluster
        """
        try_import_ray()
        import ray

        RayResourceManager._init_ray()
        return ray.cluster_resources().get(key, default_val)

    @staticmethod
    def get_cpu_count() -> int:
        """Get number of cpu cores (virtual) available in the cluster"""
        return int(RayResourceManager._get_cluster_resources("CPU"))

    @staticmethod
    def get_gpu_count() -> int:
        """Get number of gpus available in the cluster"""
        return int(RayResourceManager._get_cluster_resources("GPU"))

    @staticmethod
    def get_available_virtual_mem(format: str = "B") -> float:
        bytes = int(RayResourceManager._get_cluster_resources("memory"))
        return ResourceManager.bytes_converter(value=bytes, format_in="B", format_out=format)


def get_resource_manager():
    """Get resource manager class based on the training context"""
    return RayResourceManager if DistributedContext.is_distributed_mode() else ResourceManager
