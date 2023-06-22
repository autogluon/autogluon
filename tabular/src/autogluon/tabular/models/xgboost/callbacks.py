import logging
import time

from xgboost.callback import EarlyStopping

from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.utils.early_stopping import SimpleES

logger = logging.getLogger(__name__)


class EarlyStoppingCustom(EarlyStopping):
    """
    Augments early stopping in XGBoost to also consider time_limit, memory usage, and usage of adaptive early stopping methods.

    Parameters
    ----------
    rounds : int or tuple
       If int, The possible number of rounds without the trend occurrence.
       If tuple, contains early stopping class as first element and class init kwargs as second element.
    """

    def __init__(self, rounds, time_limit=None, start_time=None, verbose=False, min_delta=2e-6, **kwargs):
        if rounds is None:
            # Disable early stopping via rounds
            rounds = 999999
        # Add a tiny min_delta so training doesn't go on for extremely long if only tiny improvements are being made
        #  (can occur when validation error is almost 0, such as val log_loss <0.00005)
        super().__init__(rounds=999999, min_delta=min_delta, **kwargs)
        if isinstance(rounds, int):
            self.es = SimpleES(patience=rounds)
        else:
            self.es = rounds[0](**rounds[1])
        self.time_limit = time_limit
        self.start_time = start_time
        self.verbose = verbose
        self._mem_status = None
        self._mem_init_rss = None

    @disable_if_lite_mode(ret=lambda self, model: super().before_training(model=model))
    def before_training(self, model):
        model = super().before_training(model=model)
        if self.start_time is None:
            self.start_time = time.time()
        self._mem_status = ResourceManager.get_process()
        self._mem_init_rss = self._mem_status.memory_info().rss
        return model

    def after_iteration(self, model, epoch, evals_log):
        should_stop = super().after_iteration(model, epoch, evals_log)
        if should_stop:
            return should_stop
        is_best_iter = self.current_rounds == 0
        should_stop = self.es.update(cur_round=epoch, is_best=is_best_iter)
        if should_stop:
            return should_stop
        if self._time_check(model=model, epoch=epoch):
            return True
        if epoch % 10 == 0 and self._memory_check(model=model):
            return True
        return should_stop

    def _time_check(self, model, epoch):
        if self.time_limit is not None:
            time_elapsed = time.time() - self.start_time
            time_left = self.time_limit - time_elapsed
            if time_left <= 0:
                if self.verbose:
                    logger.log(
                        20,
                        f"Ran out of time, early stopping on iteration {epoch}. Best iteration is: \t[{model.attr('best_iteration')}]\t{model.attr('best_score')}",
                    )
                return True
        return False

    @disable_if_lite_mode(ret=False)
    def _memory_check(self, model):
        available = ResourceManager.get_available_virtual_mem()
        cur_rss = self._mem_status.memory_info().rss
        if cur_rss < self._mem_init_rss:
            self._mem_init_rss = cur_rss
        estimated_model_size_mb = (cur_rss - self._mem_init_rss) >> 20
        available_mb = available >> 20

        model_size_memory_ratio = estimated_model_size_mb / available_mb

        if (model_size_memory_ratio > 1.0) or (available_mb < 512):
            logger.warning("Warning: Large XGB model size may cause OOM error if training continues")
            logger.warning(f"Available Memory: {available_mb} MB")
            logger.warning(f"Estimated XGB model size: {estimated_model_size_mb} MB")
            if self.verbose:
                logger.warning(
                    f"Warning: Early stopped XGB model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality.\n"
                )
                logger.warning(f"Early stopping. Best iteration is: \t[{model.attr('best_iteration')}]\t{model.attr('best_score')}")
            return True
        elif self.verbose and (model_size_memory_ratio > 0.25):
            logger.log(15, f"Available Memory: {available_mb} MB")
            logger.log(15, f"Estimated XGB model size: {estimated_model_size_mb} MB")
        return False
