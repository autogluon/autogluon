from __future__ import annotations

import typing

from ._early_stopping_callback import EarlyStoppingCallback

if typing.TYPE_CHECKING:
    # avoid circular import for type hints
    from ..trainer import AbstractTrainer


class EarlyStoppingEnsembleCallback(EarlyStoppingCallback):
    """
    Identical to `EarlyStoppingCallback`, except that it fits a weighted ensemble model after every normal model fit.
    This should generally lead to a better solution than the simpler `EarlyStoppingCallback` because it captures the improvement in the ensemble strength.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set in `before_trainer_fit` call
        self.infer_limit_batch_size = None

    def before_trainer_fit(self, trainer: AbstractTrainer, **kwargs):
        super().before_trainer_fit(trainer=trainer, **kwargs)
        self.infer_limit_batch_size = kwargs.get("infer_limit_batch_size", None)

    def calc_new_best(self, trainer: AbstractTrainer, **kwargs):
        if kwargs["stack_name"] == "core" and len(kwargs["model_names"]) != 0:
            # only fit weighted ensemble if stack_name == "core" and at least one new model has been fit.
            self._fit_weighted_ensemble(trainer=trainer)
        return super().calc_new_best(trainer=trainer, **kwargs)

    def _fit_weighted_ensemble(self, trainer: AbstractTrainer):
        """
        Fits a weighted ensemble using the available models.
        """
        base_model_names = trainer.get_model_names(stack_name="core")
        if len(base_model_names) < 2:
            # Skip ensemble fitting if 0 or 1 base models exist (no benefit to gain).
            return
        use_val = trainer._X_val_saved and trainer._y_val_saved

        # TODO: Can optimize this with some code refactoring in AbstractTrainer
        #  It shouldn't be necessary to load `X` since the features are not used by the weighted ensemble.
        if use_val:  # holdout
            X = trainer.load_X_val()
            y = trainer.load_y_val()
            fit = False
        else:  # out-of-fold
            X = trainer.load_X()
            y = trainer.load_y()
            fit = True
        time_limit = trainer.time_left
        if time_limit is not None:
            time_limit = min(time_limit * 0.9, 360.0)
        # Fit weighted ensemble
        trainer.stack_new_level_aux(
            X=X,
            y=y,
            base_model_names=base_model_names,
            fit=fit,
            infer_limit=self.infer_limit,
            infer_limit_batch_size=self.infer_limit_batch_size,
            time_limit=time_limit,
            name_extra="_ES",
        )
