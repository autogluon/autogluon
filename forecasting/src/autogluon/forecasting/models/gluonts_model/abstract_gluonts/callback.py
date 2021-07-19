from gluonts.mx.trainer.callback import Callback
import time
import logging


class EpochCounter(Callback):

    def __init__(self):
        self.count = 0

    def on_epoch_end(
        self,
        **kwargs
    ) -> bool:
        self.count += 1
        return True


class TimeLimitCallback(Callback):

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
                logging.warning("Time limit exceed during training, stop training.")
                return False
        return True