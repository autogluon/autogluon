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
    def get_gpu_count_all():
        num_gpus = ResourceManager._get_gpu_count_cuda()
        if num_gpus == 0:
            num_gpus = ResourceManager.get_gpu_count_torch()
        return num_gpus

    @staticmethod
    def get_gpu_count_torch():
        try:
            import torch

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
    @disable_if_lite_mode(ret=4096)
    def get_memory_size():
        import psutil

        return bytes_to_mega_bytes(psutil.virtual_memory().total)

    @staticmethod
    @disable_if_lite_mode(ret=None)
    def get_process(pid=None):
        import psutil

        return psutil.Process(pid)

    @staticmethod
    @disable_if_lite_mode(ret=1073741824)  # set to 1GB as an empirical value in lite/web-browser mode.
    def get_memory_rss():
        return ResourceManager.get_process().memory_info().rss

    @staticmethod
    @disable_if_lite_mode(ret=1073741824)  # set to 1GB as an empirical value in lite/web-browser mode.
    def get_available_virtual_mem():
        import psutil

        return psutil.virtual_memory().available

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
    # TODO: find a better naming, "all" sounds unnecessary
    def get_gpu_count_all() -> int:
        """Get number of gpus available in the cluster"""
        return int(RayResourceManager._get_cluster_resources("GPU"))


def get_resource_manager():
    """Get resource manager class based on the training context"""
    return RayResourceManager if DistributedContext.is_distributed_mode() else ResourceManager
