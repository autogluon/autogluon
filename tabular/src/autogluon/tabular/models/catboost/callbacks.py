
import logging
import time

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
        import psutil
        self.period = period
        self.mem_status = psutil.Process()
        self.init_mem_rss = self.mem_status.memory_info().rss
        self.init_mem_avail = psutil.virtual_memory().available
        self.verbose = verbose

        self._cur_period = 1

    def after_iteration(self, info):
        iteration = info.iteration
        if iteration % self._cur_period == 0:
            not_enough_memory = self.memory_check(iteration)
            if not_enough_memory:
                logger.log(20, f'\tRan low on memory, early stopping on iteration {info.iteration}.')
                return False
        return True

    def memory_check(self, iter) -> bool:
        """Checks if memory usage is unsafe. If so, then returns True to signal the model to stop training early."""
        import psutil
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

        if model_size_memory_ratio > 0.5:
            self._cur_period = 1  # Increase rate of memory check if model gets large enough to cause OOM potentially
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
        if self.time_end < (time_cur + 2*time_per_iter):
            logger.log(20, f'\tRan out of time, early stopping on iteration {info.iteration}.')
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
    def __init__(self, stopping_rounds, eval_metric, compare_key='validation'):
        if isinstance(stopping_rounds, int):
            from autogluon.core.utils.early_stopping import SimpleES
            self.es = SimpleES(patience=stopping_rounds)
        else:
            self.es = stopping_rounds[0](**stopping_rounds[1])
        self.best_score = None
        self.compare_key = compare_key

        if isinstance(eval_metric, str):
            # FIXME: Avoid using private API! (https://github.com/awslabs/autogluon/issues/1381)
            from catboost._catboost import is_maximizable_metric
            is_max_optimal = is_maximizable_metric(eval_metric)
            eval_metric_name = eval_metric
        else:
            is_max_optimal = eval_metric.is_max_optimal()
            # FIXME: Unsure if this works for custom metrics!
            eval_metric_name = eval_metric.__class__.__name__

        self.eval_metric_name = eval_metric_name
        self.is_max_optimal = is_max_optimal

    def after_iteration(self, info):
        is_best_iter = False
        cur_score = info.metrics[self.compare_key][self.eval_metric_name][-1]
        if not self.is_max_optimal:
            cur_score *= -1
        if self.best_score is None:
            self.best_score = cur_score
        elif cur_score > self.best_score:
            is_best_iter = True
            self.best_score = cur_score

        should_stop = self.es.update(cur_round=info.iteration, is_best=is_best_iter)
        return not should_stop
