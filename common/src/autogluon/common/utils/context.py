from contextlib import contextmanager


@contextmanager
def set_torch_num_threads(num_cpus):
    """Set torch number of threads and recover it upon exist"""
    import torch

    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_cpus)
    yield
    torch.set_num_threads(original_num_threads)
