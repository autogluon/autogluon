from .backend import backend

__all__ = ['cpu_count', 'gpu_count']

def cpu_count():
    import multiprocessing as mp
    return mp.cpu_count()

def gpu_count():
    if backend == 'mxnet':
        import mxnet as mx
        return len(mx.test_utils.list_gpus())
    elif backend == 'torch':
        import torch
        return torch.cuda.device_count()
