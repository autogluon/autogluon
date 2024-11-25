import logging
import time

import numpy as np
from fastai.callback.core import Callback, CancelFitException
from fastai.callback.tracker import TrackerCallback
from fastcore.basics import store_attr

logger = logging.getLogger(__name__)


class BatchTimeTracker(Callback):
    """
    Training callback which allows collecting batch training times. The primary use is epoch training time estimation in adaptive epoch number selection.
    """

    def __init__(self, batches_to_measure):
        self.batches_to_measure = batches_to_measure
        self.batches_finished = 0
        self.batch_start_time = None
        self.batch_measured_time = None

    def after_batch(self):
        self.batches_finished += 1
        if self.batches_finished == 1:
            # skip first batch due to initialization overhead
            self.batch_start_time = self._time_now()
        if self.batches_finished > self.batches_to_measure:
            self.batch_measured_time = (self._time_now() - self.batch_start_time) / self.batches_to_measure
            raise CancelFitException()

    def _time_now(self):
        return time.time()


class EarlyStoppingCallbackWithTimeLimit(TrackerCallback):
    def __init__(self, monitor="valid_loss", comp=None, min_delta=0.0, patience=1, reset_on_fit=True, time_limit=None, best_epoch_stop=None):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        self.patience = patience
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_epoch_stop = best_epoch_stop
        self.wait = None

    def before_fit(self):
        self.wait = 0
        super().before_fit()

    def after_epoch(self):
        if self.best_epoch_stop is not None:
            if self.epoch >= self.best_epoch_stop:
                logger.log(20, f"\tStopping at the best epoch learned earlier - {self.epoch}.")
                raise CancelFitException()

        super().after_epoch()

        if self.new_best:
            self.wait = 0
        else:
            loss_val = self.recorder.values[-1][self.idx]
            if np.isnan(loss_val):
                if self.epoch == 0:
                    raise AssertionError(f"WARNING: NaN loss encountered in epoch {self.epoch}!")
                else:
                    logger.log(30, f"WARNING: NaN loss encountered in epoch {self.epoch}: early stopping")
                    raise CancelFitException()
            self.wait += 1
            if self.wait >= self.patience:
                logger.log(20, f"No improvement since epoch {self.epoch - self.wait}: early stopping")
                raise CancelFitException()

        if self.time_limit:
            time_elapsed = time.time() - self.start_time
            time_left = self.time_limit - time_elapsed
            time_per_epoch = time_elapsed / (self.epoch + 1)
            if time_left < time_per_epoch:
                logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {self.epoch})")
                raise CancelFitException()


class AgSaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."

    _only_train_loop = True

    def __init__(
        self, monitor="valid_loss", comp=None, min_delta=0.0, fname="model", every_epoch=False, with_opt=False, reset_on_fit=True, best_epoch_stop=None
    ):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        # keep track of file path for loggers
        self.last_saved_path = None
        self.best_epoch_stop = best_epoch_stop
        store_attr("fname,every_epoch,with_opt")

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.best_epoch_stop is not None:  # use epoch learned earlier
            if self.epoch >= self.best_epoch_stop:
                logger.log(15, f"Saving model model at the best epoch learned earlier - {self.epoch}.")
                self.best_epoch = self.epoch
                self.learn.save(f"{self.fname}")
        if self.every_epoch:
            self._save(f"{self.fname}_{self.epoch}")
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                logger.log(15, f"Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.")
                self.best_epoch = self.epoch
                self._save(f"{self.fname}")

    def after_fit(self, **kwargs):
        if not self.every_epoch:
            self.learn.load(f"{self.fname}", with_opt=self.with_opt, weights_only=False)   # nosec B614
