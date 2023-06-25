import torch


class TorchThreadManager:
    """
    Temporary updates torch num_threads to the specified value within the context, then reverts to the original num_threads upon exit.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.num_threads_og = None

    def __enter__(self):
        self.num_threads_og = torch.get_num_threads()
        torch.set_num_threads(self.num_threads)

    def __exit__(self, exc_type, exc_value, exc_tb):
        torch.set_num_threads(self.num_threads_og)
