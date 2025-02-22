import logging

from ..constants import DDP_STRATEGIES

logger = logging.getLogger(__name__)


def is_interactive_strategy(strategy: str):
    if isinstance(strategy, str) and strategy:
        return strategy.startswith(("ddp_fork", "ddp_notebook"))
    else:
        return False


def run_ddp_only_once(num_gpus: int, strategy: str):
    if strategy in DDP_STRATEGIES:
        global FIRST_DDP_RUN  # Use the global variable to make sure it is tracked per process
        if "FIRST_DDP_RUN" in globals() and not FIRST_DDP_RUN:
            # not the first time running DDP, set number of devices to 1 (use single GPU)
            return min(1, num_gpus), "auto"
        else:
            if num_gpus > 1:
                FIRST_DDP_RUN = False  # run DDP for the first time, disable the following runs
    return num_gpus, strategy
