from __future__ import annotations

import typing
from logging import Logger

from ._abstract_callback import AbstractCallback
from ._smooth_count import max_models_from_num_samples_val

if typing.TYPE_CHECKING:
    # avoid circular import for type hints
    from ..trainer import AbstractTrainer


class EarlyStoppingCountCallback(AbstractCallback):
    """
    A simple early stopping callback.
    Will early stop AutoGluon's training process after `patience` number of models fitted sequentially.

    Parameters
    ----------
    patience : int, default = 10
        The number of models fit before early stopping the training process.
    patience_per_level : bool, default = True
        If True, patience is reset after each stack level.
        Instead of stopping the trainer's fit process, reaching patience threshold will instead skip to fitting the next stack layer.
        If False, the entire trainer fit process will be stopped when reaching threshold, and patience will not be reset after each stack level.
        It is recommended to keep as `True` for the best result quality.
    verbose : bool, default = True
        If True, will log a stopping message when early stopping triggers.
    """

    skip_if_trainer_stopped: bool = True

    def __init__(self, patience: int | list | None = 10, patience_per_level: bool = True, verbose: bool = True):
        super().__init__()
        self.patience = patience
        self.patience_per_level = patience_per_level
        self.last_level = None
        self.logged_stopping_msg = False  # skips logging if True
        self.verbose = verbose
        self.models_fit = 0

    def before_trainer_fit(self, trainer: AbstractTrainer, **kwargs):
        super().before_trainer_fit(trainer=trainer, **kwargs)
        if isinstance(self.patience, list):
            n_samples = kwargs["X"].shape[0]
            patience_new = max_models_from_num_samples_val(num_samples_val=n_samples, points=self.patience)

            if self.verbose:
                if patience_new is not None:
                    msg = f"Initializing patience to {patience_new}. Reason: num_rows_train={n_samples}, patience_curve={self.patience}"
                else:
                    msg = f"Disabling callback. Reason: num_rows_train={n_samples}, which is larger than patience_curve={self.patience}"
                self._log(trainer.logger, 20, msg=msg)
                self.patience = patience_new

    def _before_model_fit(self, trainer: AbstractTrainer, stack_name: str = "core", level: int = 1, **kwargs) -> tuple[bool, bool]:
        if self.patience_per_level and (self.last_level is None or self.last_level != level):
            self.models_fit = 0
            self.last_level = level
            self.logged_stopping_msg = False
        early_stop = self._early_stop()
        if self.verbose and early_stop:
            if not self.logged_stopping_msg:
                self.logged_stopping_msg = True
                if self.patience_per_level:
                    msg = f"Early stopping trainer fit for level={level}. Reason: Fit {self.models_fit} models (max_models={self.patience})"
                else:
                    msg = f"Early stopping trainer fit. Reason: Fit {self.models_fit} models (max_models={self.patience})"
                self._log(trainer.logger, 20, msg=msg)
        if self.patience_per_level:
            return False, early_stop
        else:
            return early_stop, False

    def _after_model_fit(
        self,
        trainer: AbstractTrainer,
        model_names: list[str],
        stack_name: str = "core",
        **kwargs,
    ) -> bool:
        if stack_name == "core":
            # Ignore weighted ensembles
            self.models_fit += len(model_names)
        return False

    def _early_stop(self):
        if self.patience is not None and self.models_fit >= self.patience:
            return True
        else:
            return False

    def _log(self, logger: Logger, level, msg: str):
        msg = f"{self.__class__.__name__}: {msg}"
        logger.log(
            level,
            msg,
        )
