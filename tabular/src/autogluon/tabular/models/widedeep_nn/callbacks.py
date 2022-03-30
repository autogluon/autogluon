import logging
import time
from typing import Optional, Dict

from pytorch_widedeep.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


# TODO: add time limit
# TODO: refit_all support
class EarlyStoppingCallbackWithTimeLimit(EarlyStopping):

    def __init__(self, monitor: str = "val_loss", min_delta: float = 0.0, patience: int = 10, verbose: int = 0, mode: str = "auto",
                 baseline: Optional[float] = None, restore_best_weights: bool = False,
                 time_limit: Optional[int] = None, best_epoch_stop: Optional[int] = None):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.best_epoch_stop = best_epoch_stop
        self.time_limit = time_limit
        self.start_time = time.time()

    def on_train_begin(self, logs: Optional[Dict] = None):
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None):
        if self.best_epoch_stop is not None:
            if epoch >= self.best_epoch_stop:
                logger.log(20, f'\tStopping at the best epoch learned earlier - {epoch + 1}.')
                self.stopped_epoch = epoch
                self.trainer.early_stop = True
        else:
            super().on_epoch_end(epoch, logs, metric)
            if self.time_limit:
                time_elapsed = time.time() - self.start_time
                time_left = self.time_limit - time_elapsed
                time_per_epoch = time_elapsed / (epoch + 1)
                if time_left < time_per_epoch:
                    logger.log(20, f'\tRan out of time, stopping training early. (Stopping on epoch {epoch + 1})')
                    self.stopped_epoch = epoch
                    self.trainer.early_stop = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            logger.log(20, "Restoring model weights from the end of the best epoch")
                        self.model.load_state_dict(self.state_dict)
