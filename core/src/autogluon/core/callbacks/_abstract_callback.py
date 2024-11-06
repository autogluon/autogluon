from __future__ import annotations

import typing
from abc import ABCMeta

from ..models import AbstractModel

if typing.TYPE_CHECKING:
    # avoid circular import for type hints
    from ..trainer import AbstractTrainer


# TODO: Open design questions:
#  1. Should trainer be a class variable for ease of access?
class AbstractCallback(object, metaclass=ABCMeta):
    """
    Abstract callback class for AutoGluon's TabularPredictor.
    The inner API and logic within `trainer` is considered private API. It may change without warning between releases.

    Attributes
    ----------
    allow_recursive_calls : bool, default = False
        If True, will allow recursive calls to this callback.
        For example, a recursive call can happen if inside `self.before_model_fit` or `self.after_model_fit`, the callback initiates a model fit in trainer.
        This model fit will then trigger `self.before_model_fit` again, which could lead to an infinite loop if the callback is not implemented carefully.
        If False, guarantees that the callback logic will be skipped if it is part of a recursive call.
    skip_if_trainer_stopped : bool, default = False
        If True, will skip self.before_model_fit and self.after_model_fit logic if `early_stop=True` was returned from any callback,
        indicating that the trainer is stopping training.
        This matters if you have 2 or more callbacks, and the first callback returns `early_stop=True`.
        If the second callback has `skip_if_trainer_stopped=True`, it will skip its callback logic.
        Otherwise, its callback logic will still trigger.

    Examples
    --------
    >>> from autogluon.core.callbacks import ExampleCallback
    >>> from autogluon.tabular import TabularDataset, TabularPredictor
    >>> callbacks = [ExampleCallback()]
    >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> label = 'class'
    >>> predictor = TabularPredictor(label=label).fit(train_data, callbacks=callbacks)
    """

    allow_recursive_calls: bool = False
    skip_if_trainer_stopped: bool = False

    def __init__(self):
        self._skip = False

    def before_trainer_fit(self, trainer: AbstractTrainer, **kwargs):
        """
        Called before fitting a trainer.
        This will be the first method call to the callback at the start of the training process (prior to `before_model_fit`).
        All available arguments are passed as kwargs along with the trainer object.
        This allows this method to theoretically override the entire training logic if desired.

        Parameters
        ----------
        trainer : AbstractTrainer
            The AutoGluon trainer object
        **kwargs
            Arguments passed to the trainer object's `train_multi_levels` method.
            Contains all relevant information to completely override the trainer's logic if desired.
            Refer to the source code for more details, or use a debugger to see the contents of `**kwargs`.
        """
        pass

    def after_trainer_fit(self, trainer: AbstractTrainer):
        """
        Called after fitting a trainer.
        This will be the final method call to the callback before AutoGluon training completes.
        Example usages of this method include logging a final summary of the training, saving information to disk, or executing additional post-fit logic.
        """
        pass

    def before_model_fit(
        self,
        trainer: AbstractTrainer,
        model: AbstractModel,
        time_limit: float | None = None,
        stack_name: str = "core",
        level: int = 1,
    ) -> tuple[bool, bool]:
        """
        Called before fitting a model.

        Parameters
        ----------
        trainer : AbstractTrainer
            The AutoGluon trainer object
        model : AbstractModel
            The AutoGluon model object to be fit
        time_limit : float | None, default = None
            The time limit in seconds remaining to fit the model
        stack_name : str, default = "core"
            [Advanced] The stack_name the model originates from.
            You can use this value to toggle logic on and off. For example, skipping core models while still fitting weighted ensemble models.
            Potential Values:
                "core": The default stack_name for all models that don't fit special criteria for other values. Most models will be under this stack_name.
                "aux1": Used for WeightedEnsemble models fit at the end of each stack layer.
        level : int, default = 1
            [Advanced] The stack level of the model.
            Model's that are not stacker models are always `level=1`.
            `level` corresponds to the `Lx` suffix in the model name. For example, `WeightedEnsemble_L2` would have `level=2`.

        Returns
        -------
        early_stop : bool
            If True, the trainer skips fitting all models (including `model`) and ends the trainer fit process immediately.
            If False, the trainer continues with its normal logic.
        skip_model : bool
            If True, the trainer skips fitting this model.
            if False, the trainer continues with its normal logic.
            Ignored if `early_stop=True`.
        """
        if self._skip or (self.skip_if_trainer_stopped and trainer._callback_early_stop):
            return False, False
        if not self.allow_recursive_calls:
            self._skip = True
        early_stop, skip_model = self._before_model_fit(
            trainer=trainer,
            model=model,
            time_limit=time_limit,
            stack_name=stack_name,
            level=level,
        )
        if not self.allow_recursive_calls:
            self._skip = False
        return early_stop, skip_model

    def _before_model_fit(
        self,
        trainer: AbstractTrainer,
        model: AbstractModel,
        time_limit: float | None = None,
        stack_name: str = "core",
        level: int = 1,
    ) -> tuple[bool, bool]:
        """
        Recommended method to implement in subclasses for logic that triggers before a model fit.

        By default, simply returns False, False to continue training as usual.
        """
        return False, False

    def after_model_fit(
        self,
        trainer: AbstractTrainer,
        model_names: list[str],
        stack_name: str = "core",
        level: int = 1,
    ) -> bool:
        """
        Called after fitting a model.

        Parameters
        ----------
        trainer : AbstractTrainer
            The AutoGluon trainer object
        model_names : list[str]
            The list of successfully fit model names in the most recent model fit.
            In most cases this will be a list of size 1 corresponding to `model.name` from the previous `before_fit` call.
            If hyperparameter tuning is enabled, the size can be >1.
            If the model crashed or failed to train for some reason, the size will be 0.
            You can load the model artifact using the model name via `model = trainer.load_model(model_name)`
        stack_name : str, default = "core"
            [Advanced] The stack_name the model originates from.
            You can use this value to toggle logic on and off. For example, skipping core models while still fitting weighted ensemble models.
            Potential Values:
                "core": The default stack_name for all models that don't fit special criteria for other values. Most models will be under this stack_name.
                "aux1": Used for WeightedEnsemble models fit at the end of each stack layer.
        level : int, default = 1
            [Advanced] The stack level of the model.
            Model's that are not stacker models are always `level=1`.
            `level` corresponds to the `Lx` suffix in the model name. For example, `WeightedEnsemble_L2` would have `level=2`.

        Returns
        -------
        early_stop : bool
            If True, the trainer stops training additional models and ends the fit process immediately.
            If False, the trainer continues with its normal logic.
        """
        if self._skip or (self.skip_if_trainer_stopped and trainer._callback_early_stop):
            return False
        if not self.allow_recursive_calls:
            self._skip = True
        early_stop = self._after_model_fit(
            trainer=trainer,
            model_names=model_names,
            stack_name=stack_name,
            level=level,
        )
        if not self.allow_recursive_calls:
            self._skip = False
        return early_stop

    def _after_model_fit(
        self,
        trainer: AbstractTrainer,
        model_names: list[str],
        stack_name: str = "core",
        level: int = 1,
    ) -> bool:
        """
        Recommended method to implement in subclasses for logic that triggers after a model fit.

        By default, simply returns False to continue training as usual.
        """
        return False
