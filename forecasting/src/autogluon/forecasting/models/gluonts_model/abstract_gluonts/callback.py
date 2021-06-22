from gluonts.mx.trainer.callback import Callback


class EpochCounter(Callback):

    def __init__(self):
        self.count = 0

    def on_epoch_end(
        self,
        **kwargs
    ) -> bool:
        self.count += 1
        return True