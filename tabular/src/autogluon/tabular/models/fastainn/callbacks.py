import logging
import os
import time
from typing import Any

from fastai.basic_train import Learner
from fastai.callbacks import EarlyStoppingCallback, TrackerCallback
from torch import Tensor

logger = logging.getLogger(__name__)


class EarlyStoppingCallbackWithTimeLimit(EarlyStoppingCallback):

    def __init__(self, learn: Learner, time_limit=None, best_epoch_stop=None, **kwargs):
        super().__init__(learn, **kwargs)
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_epoch_stop = best_epoch_stop

    def on_epoch_end(self, epoch, **kwargs):
        if self.best_epoch_stop is not None:
            if epoch >= self.best_epoch_stop:
                logger.log(20, f'\tStopping at the best epoch learned earlier - {epoch}.')
                return {'stop_training': True}
        if self.time_limit:
            time_elapsed = time.time() - self.start_time
            time_left = self.time_limit - time_elapsed
            if time_left <= 0:
                logger.log(20, '\tRan out of time, stopping training early.')
                return {'stop_training': True}
        return super().on_epoch_end(epoch, **kwargs)


class SaveModelCallback(TrackerCallback):
    """A `TrackerCallback` that saves the model when monitored quantity is best."""

    def __init__(self, learn: Learner, monitor: str = 'valid_loss', mode: str = 'auto', every: str = 'improvement', name: str = 'bestmodel',
                 best_epoch_stop=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name, self.best, self.best_epoch = every, name, None, None
        self.best_epoch_stop = best_epoch_stop
        if self.every not in ['improvement', 'epoch']:
            logger.warning(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch - 1}', purge=False)
            logger.info(15, f"Loaded {self.name}_{epoch - 1}")
        except:
            logger.info(15, f'Model {self.name}_{epoch - 1} not found.')

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Compare the value monitored to its best score and maybe save the model."""
        if self.best_epoch_stop is not None:  # use epoch learned earlier
            if epoch >= self.best_epoch_stop:
                logger.log(15, f'Saving model model at the best epoch learned earlier - {epoch}.')
                self.best_epoch = epoch
                self.learn.save(f'{self.name}')
        elif self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value()
            if isinstance(current, Tensor): current = current.cpu()
            if current is not None and self.operator(current, self.best):
                logger.log(15, f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.best_epoch = epoch
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        if self.every == "improvement" and os.path.isfile(self.path / self.model_dir / f'{self.name}.pth'):
            self.learn.load(f'{self.name}', purge=False)
