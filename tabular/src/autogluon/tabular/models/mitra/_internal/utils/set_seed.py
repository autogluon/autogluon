import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)