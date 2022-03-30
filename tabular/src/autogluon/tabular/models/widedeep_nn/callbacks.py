from typing import Optional, Dict

from pytorch_widedeep.callbacks import EarlyStopping


# TODO: add time limit
# TODO: refit_all support
class EarlyStoppingCallbackWithTimeLimit(EarlyStopping):

    def on_train_begin(self, logs: Optional[Dict] = None):
        super().on_train_begin(logs)

