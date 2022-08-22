import logging
import time

from gluonts.mx.trainer.callback import Callback

from autogluon.core.utils.early_stopping import AdaptiveES

logger = logging.getLogger(__name__)


class EpochCounter(Callback):
    def __init__(self):
        self.count = 0

    def on_epoch_end(self, **kwargs) -> bool:
        self.count += 1
        return True


class TimeLimitCallback(Callback):
    """GluonTS callback object to terminate training early if autogluon time limit
    is reached."""

    def __init__(self, time_limit=None):
        self.start_time = None
        self.time_limit = time_limit

    def on_train_start(self, **kwargs) -> None:
        self.start_time = time.time()

    def on_epoch_end(
        self,
        **kwargs,
    ) -> bool:
        if self.time_limit is not None:
            cur_time = time.time()
            if cur_time - self.start_time > self.time_limit:
                logger.warning("Time limit exceed during training, stop training.")
                return False
        return True


class GluonTSEarlyStoppingCallback(Callback):
    """GluonTS callback to early stop the training if the validation loss
    is not improved for `patience' round. For the GluonTS models used in autogluon,
    the loss is always minimized.
    """

    def __init__(self, patience):
        self.patience = patience
        self.best_round = 0
        self.best_loss = float("inf")

    def on_validation_epoch_end(self, epoch_no, epoch_loss, **kwargs):
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_round = epoch_no
            return True
        else:
            to_continue = (epoch_no - self.best_round) < self.patience
            if not to_continue:
                logger.warning(f"Early stopping triggered, stop training. Best epoch {self.best_round}")
            return to_continue


class GluonTSAdaptiveEarlyStoppingCallback(Callback):
    """Adaptive early stopping where the patience is determined by `best_round' and `_update_patience' function.
    Please also refer to autogluon.core.utils.early_stopping.py.

    Parameters
    ----------
    adaptive_rate : float, default 0.3
        The rate of increase in patience.
        Set to 0 to disable, or negative to shrink patience during training.
    adaptive_offset : int, default 10
        The initial patience when cur_round is 0.
    min_patience : int, default 10
        The minimum value of patience.
    max_patience : int, default 10000
        The maximum value of patience.

    Attributes
    ----------
    best_round : int
        The most recent round passed to self.update with `is_best=True`.
        Dictates patience and is used to determine if self.early_stop() returns True.
    patience : int
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
        patience is dictated by the following formula:
        patience = min(self.max_patience, (max(self.min_patience, round(self.best_round * self.adaptive_rate + self.adaptive_offset))))
        Effectively, patience = self.best_round * self.adaptive_rate + self.adaptive_offset, bound by min_patience and max_patience
    """

    def __init__(self, adaptive_rate=0.3, adaptive_offset=10, min_patience=10, max_patience=10000):
        self.es = AdaptiveES(adaptive_rate, adaptive_offset, min_patience, max_patience)
        self.best_round = 0
        self.best_loss = float("inf")
        self.patience = self.es._update_patience(self.best_round)

    def on_validation_epoch_end(self, epoch_no, epoch_loss, **kwargs):
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_round = epoch_no
            self.patience = self.es._update_patience(self.best_round)
            return True
        else:
            to_continue = (epoch_no - self.best_round) < self.patience
            if not to_continue:
                logger.warning(f"Early stopping triggered, stop training. Best epoch {self.best_round}")
            return to_continue
