from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Tuple

from ..models import AbstractModel
from ..trainer import AbstractTrainer


class AbstractCallback(object, metaclass=ABCMeta):
    """
    Abstract callback class for AutoGluon's TabularPredictor.
    The inner API and logic within `trainer` is considered private API. It may change without warning between releases.

    Examples
    --------
    >>> from autogluon.core.callbacks import ExampleCallback
    >>> from autogluon.tabular import TabularDataset, TabularPredictor
    >>> callbacks = [ExampleCallback()]
    >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> label = 'class'
    >>> predictor = TabularPredictor(label=label).fit(train_data, callbacks=callbacks)
    """

    @abstractmethod
    def before_fit(
        self,
        trainer: AbstractTrainer,
        model: AbstractModel,
        logger: Logger,
        time_limit: float | None = None,
        stack_name: str = "core",
        level: int = 1,
    ) -> Tuple[bool, bool]:
        """
        Called before fitting a model.

        Parameters
        ----------
        trainer : AbstractTrainer
            The AutoGluon trainer object
        model : AbstractModel
            The AutoGluon model object to be fit
        logger : Logger
            The Logger object used by trainer
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
        raise NotImplementedError

    @abstractmethod
    def after_fit(
        self,
        trainer: AbstractTrainer,
        model_names: List[str],
        logger: Logger,
        stack_name: str = "core",
        level: int = 1,
    ) -> bool:
        """
        Called after fitting a model.

        Parameters
        ----------
        trainer : AbstractTrainer
            The AutoGluon trainer object
        model_names : List[str]
            The list of successfully fit model names in the most recent model fit.
            In most cases this will be a list of size 1 corresponding to `model.name` from the previous `before_fit` call.
            If hyperparameter tuning is enabled, the size can be >1.
            If the model crashed or failed to train for some reason, the size will be 0.
            You can load the model artifact using the model name via `model = trainer.load_model(model_name)`
        logger : Logger
            The Logger object used by trainer
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
        raise NotImplementedError
