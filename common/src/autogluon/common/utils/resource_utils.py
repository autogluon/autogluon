import multiprocessing
import os
import subprocess

from .utils import bytes_to_mega_bytes
from .lite import disable_if_lite_mode


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
            # Get num gpus from mxnet first because of https://github.com/autogluon/autogluon/issues/2042
            # TODO: stop using mxnet to determine num gpus once mxnet is removed from AG
            num_gpus = ResourceManager.get_gpu_count_mxnet()
            if num_gpus == 0:
                num_gpus = ResourceManager.get_gpu_count_torch()
        return num_gpus

    @staticmethod
    def get_gpu_count_mxnet():
        # TODO: Remove this once AG get rid off mxnet
        try:
            import mxnet
            num_gpus = mxnet.context.num_gpus()
        except Exception:
            num_gpus = 0
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
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

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
    def _get_gpu_count_cuda():
        # FIXME: Sometimes doesn't detect GPU on Windows
        # FIXME: Doesn't ensure the GPUs are actually usable by the model (MXNet, PyTorch, etc.)
        from .nvutil import cudaInit, cudaDeviceGetCount, cudaShutdown
        if not cudaInit(): return 0
        gpu_count = cudaDeviceGetCount()
        cudaShutdown()
        return gpu_count
