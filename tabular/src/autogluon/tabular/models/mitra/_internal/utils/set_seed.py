import random

import numpy as np
import torch

from autogluon.common.utils.random import get_numpy_seed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(get_numpy_seed(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = get_numpy_seed(torch.initial_seed())
    set_seed(worker_seed)
