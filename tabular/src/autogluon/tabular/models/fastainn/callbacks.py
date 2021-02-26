import logging
import time

from fastai.callback.tracker import EarlyStoppingCallback, TrackerCallback
from fastcore.basics import store_attr

logger = logging.getLogger(__name__)


class EarlyStoppingCallbackWithTimeLimit(EarlyStoppingCallback):

    def __init__(self, time_limit=None, best_epoch_stop=None, **kwargs):
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_epoch_stop = best_epoch_stop

    def after_epoch(self):
        if self.best_epoch_stop is not None:
            if self.epoch >= self.best_epoch_stop:
                logger.log(20, f'\tStopping at the best epoch learned earlier - {self.epoch}.')
                return {'stop_training': True}
        if self.time_limit:
            time_elapsed = time.time() - self.start_time
            time_left = self.time_limit - time_elapsed
            if time_left <= 0:
                logger.log(20, '\tRan out of time, stopping training early.')
                return {'stop_training': True}
        super().after_epoch()


class AgSaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    _only_train_loop = True

    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., fname='model', every_epoch=False,
                 with_opt=False, reset_on_fit=True, best_epoch_stop=None):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        # keep track of file path for loggers
        self.last_saved_path = None
        self.best_epoch_stop = best_epoch_stop
        store_attr('fname,every_epoch,with_opt')

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.best_epoch_stop is not None:  # use epoch learned earlier
            if self.epoch >= self.best_epoch_stop:
                logger.log(15, f'Saving model model at the best epoch learned earlier - {self.epoch}.')
                self.best_epoch = self.epoch
                self.learn.save(f'{self.model_name}')
        if self.every_epoch:
            self._save(f'{self.fname}_{self.epoch}')
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self.best_epoch = self.epoch
                self._save(f'{self.fname}')

    def after_fit(self, **kwargs):
        if not self.every_epoch:
            self.learn.load(f'{self.fname}', with_opt=self.with_opt)
