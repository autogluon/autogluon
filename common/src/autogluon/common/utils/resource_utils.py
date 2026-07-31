import logging
import os
import shutil
import subprocess
from typing import Union

from autogluon.common.utils.try_import import try_import_ray

from .cpu_utils import get_available_cpu_count
from .distribute_utils import DistributedContext
from .utils import bytes_to_mega_bytes

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manager that fetches system related info"""

    @staticmethod
    def get_cpu_count(only_physical_cores: bool = False) -> int:
        """
        Get the number of available CPU cores.

        Parameters
        ----------
        only_physical_cores : bool, default=False
            If True, detects only physical CPU cores (not including hyperthreading/SMT).
            This can be beneficial for CPU-intensive tasks like time series forecasting
            where physical cores often provide better performance than logical cores.

        Returns
        -------
        int
            The number of available CPU cores.
        """
        return get_available_cpu_count(only_physical_cores=only_physical_cores)

    @staticmethod
    def get_cpu_count_psutil(logical=True):
        import psutil

        return psutil.cpu_count(logical=logical)

    @staticmethod
    def get_gpu_count() -> int:
        num_gpus = ResourceManager._get_gpu_count_cuda()
        if num_gpus == 0:
            num_gpus = ResourceManager.get_gpu_count_torch()
        return num_gpus

    @staticmethod
    def get_gpu_count_torch(cuda_only: bool = False) -> int:
        """
        Get the number of available GPUs

        Parameters
        ----------
        cuda_only : bool, default=False
            If True, only check for CUDA GPUs and ignore other supported accelerators.
            This is useful for models that only support CUDA and not other accelerators.

        Returns
        -------
        int
            Number of available GPUs. When cuda_only=True, returns the actual CUDA device count.
        """
        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
            elif not cuda_only and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon MPS (Metal Performance Shaders) support
                # Apple Silicon Macs have only one integrated GPU
                num_gpus = 1
            else:
                num_gpus = 0
        except Exception:
            logger.log(
                40,
                "\tFailed to import torch or check CUDA availability!"
                "Please ensure you have the correct version of PyTorch installed by running `pip install -U torch`",
            )
            num_gpus = 0
        return num_gpus

    @staticmethod
    def get_available_vram(device: int = 0) -> float | None:
        """Available GPU memory (VRAM) of `device` in bytes, or None when it cannot be determined.

        The GPU counterpart of `get_available_virtual_mem`. Three effects make this more
        than a `torch.cuda.mem_get_info` call:

        1. `mem_get_info` reports memory free on the *device*, which excludes memory
           PyTorch's caching allocator already holds. That memory is reusable by this
           process without a new device allocation, so it is added back
           (`memory_reserved - memory_allocated`); ignoring it under-reports what a fit
           can actually use and needlessly skips models.
        2. `torch.cuda.set_per_process_memory_fraction` caps this process below the
           device total. The cap is not visible to `mem_get_info`, so it is applied here —
           a process allocating past its fraction OOMs even with the device free.
        3. Without torch/CUDA, `nvidia-smi` gives device-level free memory only (no
           allocator or fraction information available).
        """
        try:
            import torch

            if torch.cuda.is_available() and device < torch.cuda.device_count():
                device_free, device_total = torch.cuda.mem_get_info(device)
                cached_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                available = float(device_free + cached_unused)

                # Respect a per-process cap when one is set (used to partition a GPU
                # across processes). The getter exists from torch 2.9; older versions
                # expose no way to read it back, so the cap is simply not applied.
                get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
                if get_fraction is not None:
                    fraction = float(get_fraction(device))
                    if fraction < 1.0:
                        process_headroom = fraction * device_total - torch.cuda.memory_allocated(device)
                        available = min(available, max(process_headroom, 0.0))
                return min(available, float(device_total))
        except Exception:
            pass
        memory_free_values = ResourceManager.get_gpu_free_memory()  # MiB per device
        if device < len(memory_free_values):
            return float(memory_free_values[device]) * 1024**2
        return None

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
    def _get_memory_size() -> float:
        if os.environ.get("AG_MEMORY_LIMIT_IN_GB", None) is not None:
            return ResourceManager._get_custom_memory_size()

        import psutil

        return psutil.virtual_memory().total

    @staticmethod
    def _get_memory_rss() -> float:
        return ResourceManager.get_process().memory_info().rss

    @staticmethod
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
