from __future__ import annotations

import logging
import os
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from autogluon.common.utils.try_import import try_import_ray

from .cpu_utils import get_available_cpu_count
from .distribute_utils import DistributedContext
from .lite import disable_if_lite_mode
from .utils import bytes_to_mega_bytes

logger = logging.getLogger(__name__)


@dataclass
class ResourcesUsageConfig:
    """Dataclass to store resources usage settings."""

    num_cpus: int | Literal["auto"] = "auto"
    """ The total amount of cpus you want AutoGluon predictor to use.
    Auto means AutoGluon will make the decision based on the total number of cpus
    available and the model requirement for best performance.
    """
    num_gpus: int | Literal["auto"] = "auto"
    """The total amount of gpus you want AutoGluon predictor to use.
    Auto means AutoGluon will make the decision based on the total number of gpus
    available and the model requirement for best performance.
    """
    memory_limit: float | str = "auto"
    """The total amount of memory in GB you want AutoGluon predictor to use.

    "auto" means AutoGluon will use all available memory on the system (that is
    detectable). Note that this is only a soft limit! AutoGluon uses this limit to
    skip training models that are expected to require too much memory or stop training
    a model that would exceed the memory limit. AutoGluon does not guarantee the
    enforcement of this limit (yet). Nevertheless, we expect AutoGluon to abide by the
    limit in most cases or, at most, go over the limit by a small margin. For most
    virtualized systems (e.g., in the cloud) and local usage on a server or laptop,
    "auto" is ideal for this parameter.

    We recommend manually setting the memory limit (and any other resources) on
    systems with shared resources that are controlled by the operating system
    (e.g., SLURM and cgroups). Otherwise, AutoGluon might wrongly assume more resources
    are available for fitting a model than the operating system allows, which can
    result in model training failing or being very inefficient.
    """
    usage_strategy: Literal["sequential", "parallel"] = "sequential"
    """The strategy used to schedule jobs on resources.
        * If "sequential", models will be fit sequentially. This is the most stable
        option with the most readable logging.
        * If "parallel", models will be fit in parallel with ray, splitting available
        compute between them. For machines with 16 or more CPU cores, it is likely that
        "parallel" will be faster than "sequential".
        Note: "parallel" is experimental and may run into issues.
    """

    @staticmethod
    def from_user_input(resource_config: dict | ResourcesUsageConfig | None):
        """Create a ResourcesUsageConfig instance from user input."""
        if resource_config is None:
            return ResourcesUsageConfig()
        if isinstance(resource_config, dict):
            return ResourcesUsageConfig(**resource_config)
        if isinstance(resource_config, ResourcesUsageConfig):
            return deepcopy(resource_config)

        raise ValueError(
            "`resource_config` must be a dict or ResourcesUsageConfig instance. "
            f"Got: {format(type(resource_config))} with value {resource_config}."
        )

    def __post_init__(self):
        """Validate the resources usage config after initialization."""
        self.validate_resources_usage_config(
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            memory_limit=self.memory_limit,
            usage_strategy=self.usage_strategy,
        )

    @staticmethod
    def validate_resources_usage_config(
        *,
        num_cpus: int | str | None = None,
        num_gpus: int | float | str | None = None,
        memory_limit: float | str | None = None,
        usage_strategy: str | None = None,
    ):
        """Validate the resources usage config."""
        if num_cpus is not None:
            ResourcesUsageConfig.validate_num_cpus(num_cpus=num_cpus)
        if num_gpus is not None:
            ResourcesUsageConfig.validate_num_gpus(num_gpus=num_gpus)
        if memory_limit is not None:
            ResourcesUsageConfig.validate_and_set_memory_limit(
                memory_limit=memory_limit
            )
        if usage_strategy is not None:
            ResourcesUsageConfig.validate_usage_strategy(usage_strategy=usage_strategy)

    @staticmethod
    def validate_num_cpus(num_cpus: int | str):
        """Validate the `num_cpus` parameter."""
        if num_cpus is None:
            raise ValueError(f"`num_cpus` must be an int or 'auto'. Value: {num_cpus}")
        if isinstance(num_cpus, str):
            if num_cpus != "auto":
                raise ValueError(
                    f"`num_cpus` must be an int or 'auto'. Value: {num_cpus}"
                )
        elif not isinstance(num_cpus, int):
            raise TypeError(
                f"`num_cpus` must be an int or 'auto'. "
                f"Found: {type(num_cpus)} | Value: {num_cpus}"
            )
        elif num_cpus < 1:
            raise ValueError(
                f"`num_cpus` must be greater than or equal to 1. (num_cpus={num_cpus})"
            )

    @staticmethod
    def validate_num_gpus(num_gpus: int | float | str):
        """Validate the `num_gpus` parameter."""
        if num_gpus is None:
            raise ValueError(
                f"`num_gpus` must be an int, float, or 'auto'. Value: {num_gpus}"
            )
        if isinstance(num_gpus, str):
            if num_gpus != "auto":
                raise ValueError(
                    f"`num_gpus` must be an int, float, or 'auto'. Value: {num_gpus}"
                )
        elif not isinstance(num_gpus, (int, float)):
            raise TypeError(
                f"`num_gpus` must be an int, float, or 'auto'. "
                f"Found: {type(num_gpus)} | Value: {num_gpus}"
            )
        elif num_gpus < 0:
            raise ValueError(
                f"`num_gpus` must be greater than or equal to 0. (num_gpus={num_gpus})"
            )

    @staticmethod
    def validate_and_set_memory_limit(memory_limit: float | str):
        """Validate and set the `memory_limit` parameter."""
        if memory_limit is None:
            raise ValueError(
                f"`memory_limit` must be an int, float, or 'auto'. "
                f"Value: {memory_limit}"
            )
        if isinstance(memory_limit, str):
            if memory_limit != "auto":
                raise ValueError(
                    f"`memory_limit` must be an int, float, or 'auto'. "
                    f"Value: {memory_limit}"
                )
        elif not isinstance(memory_limit, (int, float)):
            raise TypeError(
                "`memory_limit` must be an int, float, or 'auto'."
                f" Found: {type(memory_limit)} | Value: {memory_limit}"
            )
        elif memory_limit <= 0:
            raise ValueError(
                f"`memory_limit` must be greater than 0. (memory_limit={memory_limit})"
            )

        if memory_limit != "auto":
            logger.log(20, f"Enforcing custom memory (soft)limit of {memory_limit} GB!")
            os.environ["AG_MEMORY_LIMIT_IN_GB"] = str(memory_limit)

    @staticmethod
    def validate_usage_strategy(usage_strategy: str):
        """Validate the `usage_strategy` parameter."""
        valid_values = ["sequential", "parallel"]
        if usage_strategy not in valid_values:
            raise ValueError(
                f"usage_strategy must be one of {valid_values}. Value: {usage_strategy}"
            )


class ResourceManager:
    """Manager that fetches system related info."""

    @staticmethod
    def get_cpu_count(only_physical_cores: bool = False) -> int:
        """Get the number of available CPU cores.

        Parameters
        ----------
        only_physical_cores : bool, default=False
            If True, detects only physical CPU cores (not including hyperthreading/SMT).
            This can be beneficial for CPU-intensive tasks like time series forecasting
            where physical cores often provide better performance than logical cores.

        Returns:
        -------
        int
            The number of available CPU cores.
        """
        return get_available_cpu_count(only_physical_cores=only_physical_cores)

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
    def get_gpu_count_torch(cuda_only: bool = False) -> int:
        """Get the number of available GPUs.

        Parameters
        ----------
        cuda_only : bool, default=False
            If True, only check for CUDA GPUs and ignore other supported accelerators.
            This is useful for models that only support CUDA and not other accelerators.

        Returns:
        -------
        int
            Number of available GPUs. When cuda_only=True, returns the actual CUDA device count.
        """
        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
            elif (
                not cuda_only
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
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
    def get_gpu_free_memory():
        """Grep gpu free memory from nvidia-smi tool.
        This function can fail due to many reasons(driver, nvidia-smi tool, envs, etc) so please simply use
        it as a suggestion, stay away with any rules bound to it.
        E.g. for a 4-gpu machine, the result can be list of int
        >>> print(get_gpu_free_memory)
        >>> [13861, 13859, 13859, 13863].
        """
        _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

        try:
            COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = _output_to_list(
                subprocess.check_output(COMMAND.split())
            )[1:]
            memory_free_values = [
                int(x.split()[0]) for i, x in enumerate(memory_free_info)
            ]
        except:
            memory_free_values = []
        return memory_free_values

    @staticmethod
    def get_memory_size(format: str = "B") -> float:
        """Parameters
        ----------
        format: {"B", "KB", "MB", "GB", "TB", "PB"}

        Returns:
        -------
        Memory size in the provided `format`.

        """
        bytes = ResourceManager._get_memory_size()
        return ResourceManager.bytes_converter(
            value=bytes, format_in="B", format_out=format
        )

    @staticmethod
    def get_memory_rss(format: str = "B") -> float:
        bytes = ResourceManager._get_memory_rss()
        return ResourceManager.bytes_converter(
            value=bytes, format_in="B", format_out=format
        )

    @staticmethod
    def get_available_virtual_mem(format: str = "B") -> float:
        bytes = ResourceManager._get_available_virtual_mem()
        return ResourceManager.bytes_converter(
            value=bytes, format_in="B", format_out=format
        )

    @staticmethod
    def bytes_converter(value: float, format_in: str, format_out: str) -> float:
        """Converts bytes `value` from `format_in` to `format_out`.

        Parameters
        ----------
        value: float
        format_in: {"B", "KB", "MB", "GB", "TB", "PB"}
        format_out: {"B", "KB", "MB", "GB", "TB", "PB"}

        Returns:
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
        """Gets the disk usage information for the given path.

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
            raise ValueError(
                "Memory set via `AG_MEMORY_LIMIT_IN_GB` must be greater than 0!"
            )

        # Transform to bytes and return
        return max(int(memory_limit * (1024.0**3)), 1)

    @staticmethod
    @disable_if_lite_mode(
        ret=1073741824
    )  # set to 1GB as an empirical value in lite/web-browser mode.
    def _get_memory_size() -> float:
        if os.environ.get("AG_MEMORY_LIMIT_IN_GB", None) is not None:
            return ResourceManager._get_custom_memory_size()

        import psutil

        return psutil.virtual_memory().total

    @staticmethod
    @disable_if_lite_mode(
        ret=1073741824
    )  # set to 1GB as an empirical value in lite/web-browser mode.
    def _get_memory_rss() -> float:
        return ResourceManager.get_process().memory_info().rss

    @staticmethod
    @disable_if_lite_mode(
        ret=1073741824
    )  # set to 1GB as an empirical value in lite/web-browser mode.
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
        """Initialize ray runtime if not already initialized. Will force the existence of a cluster already being spinned up."""
        try_import_ray()
        import ray

        if not ray.is_initialized():
            ray.init(
                address="auto",  # Force ray to connect to an existing cluster. There should be one. Otherwise, something went wrong
                log_to_driver=False,
                logging_level=logging.ERROR,
            )

    @staticmethod
    def _get_cluster_resources(key: str, default_val: int | float = 0):
        """Get value of resources available in the cluster.

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
        """Get number of cpu cores (virtual) available in the cluster."""
        return int(RayResourceManager._get_cluster_resources("CPU"))

    @staticmethod
    def get_gpu_count() -> int:
        """Get number of gpus available in the cluster."""
        return int(RayResourceManager._get_cluster_resources("GPU"))

    @staticmethod
    def get_available_virtual_mem(format: str = "B") -> float:
        bytes = int(RayResourceManager._get_cluster_resources("memory"))
        return ResourceManager.bytes_converter(
            value=bytes, format_in="B", format_out=format
        )


def get_resource_manager():
    """Get resource manager class based on the training context."""
    return (
        RayResourceManager
        if DistributedContext.is_distributed_mode()
        else ResourceManager
    )
