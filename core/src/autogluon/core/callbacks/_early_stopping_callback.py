from __future__ import annotations

import typing
from logging import Logger

from ._abstract_callback import AbstractCallback

if typing.TYPE_CHECKING:
    # avoid circular import for type hints
    from ..trainer import AbstractTrainer


class EarlyStoppingCallback(AbstractCallback):
    """
    A simple early stopping callback.
    Will early stop AutoGluon's training process after `patience` number of models fitted sequentially without improvement to score_val.
    Sensitive to `infer_limit` if it was specified in the fit call. Will not consider models that go above the infer_limit.

    [Note] This callback is primarily for example purposes. Using this callback as-is will likely lead to a performance drop in AutoGluon.
    For better results, consider using `EarlyStoppingEnsembleCallback`.

    Parameters
    ----------
    patience : int, default = 10
        The number of models fit in a row without improvement in score_val before early stopping the training process.
    patience_per_level : bool, default = True
        If True, patience is reset after each stack level.
        Instead of stopping the trainer's fit process, reaching patience threshold will instead skip to fitting the next stack layer.
        If False, the entire trainer fit process will be stopped when reaching threshold, and patience will not be reset after each stack level.
        It is recommended to keep as `True` for the best result quality.
    verbose : bool, default = True
        If True, will log a stopping message when early stopping triggers.
    """

    skip_if_trainer_stopped: bool = True

    def __init__(self, patience: int = 10, patience_per_level: bool = True, verbose: bool = True):
        super().__init__()
        self.patience = patience
        self.patience_per_level = patience_per_level
        self.last_improvement = 0
        self.last_level = None
        self.logged_stopping_msg = False  # skips logging if True
        self.score_best = None
        self.verbose = verbose
        self.model_best: str | None = None

        # Set in `before_trainer_fit` call
        self.infer_limit = None

    def before_trainer_fit(self, trainer: AbstractTrainer, **kwargs):
        self.infer_limit = kwargs.get("infer_limit", None)

    def _before_model_fit(
        self, trainer: AbstractTrainer, stack_name: str = "core", level: int = 1, **kwargs
    ) -> tuple[bool, bool]:
        if self.patience_per_level and (self.last_level is None or self.last_level != level):
            self.last_improvement = 0
            self.last_level = level
            self.logged_stopping_msg = False
        early_stop = self._early_stop()
        if self.verbose and early_stop:
            if not self.logged_stopping_msg:
                self.logged_stopping_msg = True
                if self.patience_per_level:
                    msg = (
                        f"Early stopping trainer fit for level={level}. "
                        f"Reason: No score_val improvement in the past {self.last_improvement} models."
                    )
                else:
                    msg = f"Early stopping trainer fit. Reason: No score_val improvement in the past {self.last_improvement} models."
                self._log(trainer.logger, 20, msg=msg)
        if self.patience_per_level:
            return False, early_stop
        else:
            return early_stop, False

    def _after_model_fit(self, trainer: AbstractTrainer, **kwargs) -> bool:
        self.calc_new_best(trainer=trainer, **kwargs)
        if self.verbose:
            if self.score_best is None:
                msg_score = f"{self.score_best}"
            else:
                msg_score = f"{self.score_best:.4f}"
            msg = f"Best Score: {msg_score} | Patience: {self.last_improvement}/{self.patience} | Best Model: {self.model_best}"
            if self.last_improvement == 0:
                msg += " (New Best)"
            self._log(trainer.logger, 20, msg=msg)
        return False

    def calc_new_best(self, trainer: AbstractTrainer, **kwargs):
        """
        Computes the new best model and validation score, and then resets `self.last_improvement` to 0 if an improvement is observed or increments it otherwise.
        """
        self._calc_new_best(trainer=trainer)

    def _calc_new_best(self, trainer: AbstractTrainer):
        model_cur, score_cur = self._cur_best(trainer=trainer)
        if score_cur is None:
            self.last_improvement += 1
        elif self.score_best is None or score_cur > self.score_best:
            self.score_best = score_cur
            self.model_best = model_cur
            self.last_improvement = 0
        else:
            self.last_improvement += 1

    def _cur_best(self, trainer: AbstractTrainer) -> tuple[str, float]:
        """
        Returns the current best model in terms of validation score that satisfies `self.infer_limit` (if specified)

        Returns
        -------
        model : str
            The model name with the best validation score.
            If no models exist, returns None.
        val_score: float
            The validation score of `model`.
            If `model=None`, returns None.
        """
        val_score_dict = trainer.get_models_attribute_dict("val_score")
        if len(val_score_dict) == 0:
            score_best = None
            model_best = None
        else:
            # TODO: infer_limit_as_child should be controlled by trainer, but currently trainer always uses True
            #  Trainer should be updated to adjust `infer_limit_as_child` value based on if refit_full is specified.
            model_best = trainer.get_model_best(
                can_infer=None, infer_limit=self.infer_limit, infer_limit_as_child=True
            )
            score_best = trainer.get_model_attribute(model=model_best, attribute="val_score")
        return model_best, score_best

    def _early_stop(self):
        if self.last_improvement >= self.patience:
            return True
        else:
            return False

    def _log(self, logger: Logger, level, msg: str):
        msg = f"{self.__class__.__name__}: {msg}"
        logger.log(
            level,
            msg,
        )
