import platform
import sys
from typing import Tuple

from .. import __version__
from .resource_utils import ResourceManager, get_resource_manager


def get_ag_system_info_disk_space(path: str) -> Tuple[str, int]:
    disk_verbosity = 20
    try:
        # TODO: Make this logic smarter, incorporate training data size and potentially models to train into the logic to define the recommended disk space.
        #  For example, `best_quality` will require more disk space than `medium_quality`, and HPO would require additional disk space.
        disk_stats = ResourceManager.get_disk_usage(path=path)
        disk_free_gb = ResourceManager.bytes_converter(value=disk_stats.free, format_in="B", format_out="GB")
        disk_total_gb = ResourceManager.bytes_converter(value=disk_stats.total, format_in="B", format_out="GB")
        disk_proportion_avail = disk_free_gb / disk_total_gb
        disk_log_extra = ""
        disk_free_gb_warning_threshold = 10
        if disk_free_gb <= disk_free_gb_warning_threshold:
            disk_log_extra += (
                f"\n\tWARNING: Available disk space is low and there is a risk that "
                f"AutoGluon will run out of disk during fit, causing an exception. "
                f"\n\tWe recommend a minimum available disk space of {disk_free_gb_warning_threshold} GB, "
                f"and large datasets may require more."
            )
            disk_verbosity = 30
        msg = (
            f"Disk Space Avail:   {disk_free_gb:.2f} GB / {disk_total_gb:.2f} GB "
            f"({disk_proportion_avail * 100:.1f}%){disk_log_extra}"
        )
        return msg, disk_verbosity
    except Exception as e:
        # Note: using a broad exception catch as it is unknown what scenarios an exception would be raised, and what exception type would be used.
        #  The broad exception ensures that we don't completely break AutoGluon for users who may be running on strange hardware or environments.
        msg = (
            f"Disk Space Avail:   WARNING, an exception ({e.__class__.__name__}) occurred while attempting to get available disk space. "
            f"Consider opening a GitHub Issue."
        )
        return msg, disk_verbosity


def get_ag_system_info(*, path: str = None, include_gpu_count=False, include_pytorch=True, include_cuda=True) -> str:
    resource_manager: ResourceManager = get_resource_manager()
    system_num_cpus = resource_manager.get_cpu_count()
    available_mem = ResourceManager.get_available_virtual_mem("GB")
    total_mem = ResourceManager.get_memory_size("GB")
    mem_avail_percent = available_mem / total_mem
    version = __version__
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    msg_list = [
        f"=================== System Info ===================",
        f"AutoGluon Version:  {version}",
        f"Python Version:     {python_version}",
        f"Operating System:   {platform.system()}",
        f"Platform Machine:   {platform.machine()}",
        f"Platform Version:   {platform.version()}",
        f"CPU Count:          {system_num_cpus}",
    ]
    if include_pytorch:
        try:
            import torch

            torch_version = torch.__version__
        except Exception as e:
            torch_version = "Can't import torch"
        msg_list.append(f"Pytorch Version:    {torch_version}")
    if include_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
            else:
                cuda_version = "CUDA is not available"
        except Exception as e:
            cuda_version = "Can't get cuda version from torch"
        msg_list.append(f"CUDA Version:       {cuda_version}")
    if include_gpu_count:
        try:
            import torch

            system_num_gpus = resource_manager.get_gpu_count_torch()
            gpu_memory_info = []
            combined_free_memory = 0
            total_allocated_memory = 0
            combined_gpu_memory = 0
            for i in range(system_num_gpus):
                total_memory_gpu = torch.cuda.get_device_properties(i).total_memory
                total_memory_gb = total_memory_gpu / (1024**3)  # Convert bytes to GB
                allocated_memory = torch.cuda.memory_allocated(i)
                allocated_memory_gb = allocated_memory / (1024**3)
                free_memory_gb = total_memory_gb - allocated_memory_gb

                combined_free_memory += free_memory_gb
                total_allocated_memory += allocated_memory_gb
                combined_gpu_memory += total_memory_gb

                gpu_memory_info.append(f"GPU {i}: {free_memory_gb:.2f}/{total_memory_gb:.2f} GB")

            gpu_memory_str = " | ".join(gpu_memory_info)
            msg_list.append(f"GPU Memory:         {gpu_memory_str}")
            msg_list.append(
                f"Total GPU Memory:   Free: {combined_free_memory:.2f} GB, Allocated: {total_allocated_memory:.2f} GB, Total: {combined_gpu_memory:.2f} GB"
            )

        except Exception as e:
            system_num_gpus = f"WARNING: Exception was raised when calculating GPU count ({e.__class__.__name__})"
        msg_list.append(f"GPU Count:          {system_num_gpus}")

    msg_list.append(
        f"Memory Avail:       {available_mem:.2f} GB / {total_mem:.2f} GB ({mem_avail_percent * 100:.1f}%)"
    )

    if path is not None:
        disk_avail_msg, _ = get_ag_system_info_disk_space(path=path)
        msg_list.append(disk_avail_msg)
    msg_list.append(f"===================================================")

    msg = "\n".join(msg_list)
    return msg
