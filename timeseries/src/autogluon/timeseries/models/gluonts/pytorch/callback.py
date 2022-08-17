"""PyTorch Lightning equivalents of GluonTS callbacks"""
import time
import logging

import pytorch_lightning as pl


logger = logging.getLogger(__name__)


class PLTimeLimitCallback(pl.callbacks.Callback):
    """GluonTS callback object to terminate training early if autogluon time limit
    is reached."""

    def __init__(self, time_limit=None):
        self.start_time = None
        self.time_limit = time_limit

    def on_train_start(self, **kwargs) -> None:
        self.start_time = time.time()

    def on_epoch_end(
        self, trainer: pl.Trainer, **kwargs
    ) -> bool:
        if self.time_limit is not None:
            cur_time = time.time()
            if cur_time - self.start_time > self.time_limit:
                logger.warning("Time limit exceed during training, stop training.")
                trainer.should_stop = True
