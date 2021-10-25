
import logging
import time
import psutil

logger = logging.getLogger(__name__)


class MemoryCheckCallback:
    """
    Callback to ensure memory usage is safe, otherwise early stops the model to avoid OOM errors.

    This callback is CatBoost specific.

    Parameters
    ----------
    period : int, default = 10
        Number of iterations between checking memory status. Higher values are less precise but use less compute.
    verbose : bool, default = False
        Whether to log information on memory status even if memory usage is low.
    """
    def __init__(self, period: int = 10, verbose=False):
        self.period = period
        self.mem_status = psutil.Process()
        self.init_mem_rss = self.mem_status.memory_info().rss
        self.init_mem_avail = psutil.virtual_memory().available
        self.verbose = verbose

    def after_iteration(self, info):
        if info.iteration % self.period == 0:
            not_enough_memory = self.memory_check()
            if not_enough_memory:
                logger.log(20, f'\tRan low on memory, early stopping on iteration {info.iteration}.')
                return False
        return True

    def memory_check(self) -> bool:
        """Checks if memory usage is unsafe. If so, then returns True to signal the model to stop training early."""
        available_bytes = psutil.virtual_memory().available
        cur_rss = self.mem_status.memory_info().rss

        if cur_rss < self.init_mem_rss:
            self.init_mem_rss = cur_rss
        estimated_model_size_mb = (cur_rss - self.init_mem_rss) >> 20
        available_mb = available_bytes >> 20
        model_size_memory_ratio = estimated_model_size_mb / available_mb

        early_stop = False
        if model_size_memory_ratio > 1.0:
            logger.warning(f'Warning: Large model size may cause OOM error if training continues')
            early_stop = True

        if available_mb < 512:  # Less than 500 MB
            logger.warning(f'Warning: Low available memory may cause OOM error if training continues')
            early_stop = True

        if early_stop:
            logger.warning('Warning: Early stopped model prior to optimal result to avoid OOM error. '
                           'Please increase available memory to avoid subpar model quality.')
            logger.warning(f'Available Memory: {available_mb} MB, Estimated Model size: {estimated_model_size_mb} MB')
            return True
        elif self.verbose or (model_size_memory_ratio > 0.25):
            logging.debug(f'Available Memory: {available_mb} MB, Estimated Model size: {estimated_model_size_mb} MB')
        return False


class TimeCheckCallback:
    """
    Callback to ensure time limit is respected, otherwise early stops the model to avoid going over time.

    This callback is CatBoost specific.

    Parameters
    ----------
    time_start : float
        The starting time (usually obtained via `time.time()`) of the training.
    time_limit : float
        The time in seconds before stopping training.
    """
    def __init__(self, time_start, time_limit):
        self.time_end = time_start + time_limit
        self.time_start = time_start

    def after_iteration(self, info):
        time_cur = time.time()
        time_per_iter = (time_cur - self.time_start) / info.iteration
        if self.time_end < (time_cur + 2*time_per_iter):
            logger.log(20, f'\tRan out of time, early stopping on iteration {info.iteration}.')
            return False
        return True
