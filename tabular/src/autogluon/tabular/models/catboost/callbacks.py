import logging
import time

from autogluon.common.utils.resource_utils import ResourceManager

from .catboost_utils import CATBOOST_QUANTILE_PREFIX

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
        self.init_mem_rss = ResourceManager.get_memory_rss()
        self.init_mem_avail = ResourceManager.get_available_virtual_mem()
        self.verbose = verbose

        self._cur_period = 1

    def after_iteration(self, info):
        iteration = info.iteration
        if iteration % self._cur_period == 0:
            not_enough_memory = self.memory_check(iteration)
            if not_enough_memory:
                logger.log(20, f"\tRan low on memory, early stopping on iteration {info.iteration}.")
                return False
        return True

    def memory_check(self, iter: int) -> bool:
        """
        Checks if memory usage is unsafe. If so, signals the model to stop training early.

        Parameters
        ----------
        iter: int
            The current training iteration.

        Returns
        -------
        bool: True if training should stop due to memory constraints, False otherwise.
        """
        available_bytes = ResourceManager.get_available_virtual_mem()
        cur_rss = ResourceManager.get_memory_rss()

        # Update initial memory usage if current usage is lower
        if cur_rss < self.init_mem_rss:
            self.init_mem_rss = cur_rss

        # Convert memory values to MB
        estimated_model_size_mb = (cur_rss - self.init_mem_rss) / (1024 ** 2)
        available_mb = available_bytes / (1024 ** 2)

        model_size_memory_ratio = estimated_model_size_mb / available_mb
        early_stop = False

        if model_size_memory_ratio > 1.0:
            logger.warning(
                f"Iteration {iter}: Model size exceeds available memory. "
                f"Available memory: {available_mb:.2f} MB, "
                f"Estimated model size: {estimated_model_size_mb:.2f} MB."
            )
            early_stop = True

        if available_mb < 512:  # Less than 512 MB
            logger.warning(
                f"Iteration {iter}: Low available memory (<512 MB). "
                f"Available memory: {available_mb:.2f} MB, "
                f"Estimated model size: {estimated_model_size_mb:.2f} MB."
            )
            early_stop = True

        if early_stop:
            logger.warning(
                "Early stopping model prior to optimal result to avoid OOM error. "
                "Please increase available memory to avoid subpar model quality."
            )
            return True
        elif self.verbose or model_size_memory_ratio > 0.25:
            logger.debug(
                f"Iteration {iter}: "
                f"Available memory: {available_mb:.2f} MB, "
                f"Estimated model size: {estimated_model_size_mb:.2f} MB."
            )

        # Adjust memory check frequency based on model size
        if model_size_memory_ratio > 0.5:
            self._cur_period = 1  # Increase frequency of memory checks
        elif iter > self.period:
            self._cur_period = self.period

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
        if self.time_end < (time_cur + 2 * time_per_iter):
            logger.log(20, f"\tRan out of time, early stopping on iteration {info.iteration}.")
            return False
        return True


class EarlyStoppingCallback:
    """
    Early stopping callback.

    This callback is CatBoost specific.

    Parameters
    ----------
    stopping_rounds : int or tuple
       If int, The possible number of rounds without the trend occurrence.
       If tuple, contains early stopping class as first element and class init kwargs as second element.
    eval_metric : str
       The eval_metric to use for early stopping. Must also be specified in the CatBoost model params.
    compare_key : str, default = 'validation'
        The data to use for scoring. It is recommended to keep as default.
    """

    def __init__(self, stopping_rounds, eval_metric, compare_key="validation"):
        if isinstance(stopping_rounds, int):
            from autogluon.core.utils.early_stopping import SimpleES

            self.es = SimpleES(patience=stopping_rounds)
        else:
            self.es = stopping_rounds[0](**stopping_rounds[1])
        self.best_score = None
        self.compare_key = compare_key

        if isinstance(eval_metric, str):
            from catboost.core import is_maximizable_metric

            is_max_optimal = is_maximizable_metric(eval_metric)
            eval_metric_name = eval_metric
        else:
            is_max_optimal = eval_metric.is_max_optimal()
            # FIXME: Unsure if this works for custom metrics!
            eval_metric_name = eval_metric.__class__.__name__

        self.eval_metric_name = eval_metric_name
        self.is_max_optimal = is_max_optimal
        self.is_quantile = self.eval_metric_name.startswith(CATBOOST_QUANTILE_PREFIX)

    def after_iteration(self, info):
        is_best_iter = False
        if self.is_quantile:
            # FIXME: CatBoost adds extra ',' in the metric name if quantile levels are not balanced
            # e.g., 'MultiQuantile:alpha=0.1,0.25,0.5,0.95' becomes 'MultiQuantile:alpha=0.1,,0.25,0.5,0.95'
            eval_metric_name = [k for k in info.metrics[self.compare_key] if k.startswith(CATBOOST_QUANTILE_PREFIX)][0]
        else:
            eval_metric_name = self.eval_metric_name
        cur_score = info.metrics[self.compare_key][eval_metric_name][-1]
        if not self.is_max_optimal:
            cur_score *= -1
        if self.best_score is None:
            self.best_score = cur_score
        elif cur_score > self.best_score:
            is_best_iter = True
            self.best_score = cur_score

        should_stop = self.es.update(cur_round=info.iteration, is_best=is_best_iter)
        return not should_stop
