import logging
import time

from fastai.basic_train import Learner
from fastai.callbacks import EarlyStoppingCallback

logger = logging.getLogger(__name__)


class EarlyStoppingCallbackWithTimeLimit(EarlyStoppingCallback):

    def __init__(self, learn: Learner, time_limit=None, **kwargs):
        super().__init__(learn, **kwargs)
        self.time_limit = time_limit
        self.start_time = time.time()

    def on_epoch_end(self, epoch, **kwargs):
        if self.time_limit:
            time_elapsed = time.time() - self.start_time
            time_left = self.time_limit - time_elapsed
            if time_left <= 0:
                logger.log(20, "\tRan out of time, stopping training early.")
                return {'stop_training': True}
        return super().on_epoch_end(epoch, **kwargs)
