from __future__ import annotations

import time
import typing

import pandas as pd

from ..models import AbstractModel
from ._abstract_callback import AbstractCallback

if typing.TYPE_CHECKING:
    # avoid circular import for type hints
    from ..trainer import AbstractTrainer


class ExampleCallback(AbstractCallback):
    """
    Example callback showcasing how to access and log information from the trainer.
    """

    def _before_model_fit(
        self,
        trainer: AbstractTrainer,
        model: AbstractModel,
        time_limit: float | None = None,
        stack_name: str = "core",
        level: int = 1,
        **kwargs,
    ) -> tuple[bool, bool]:
        time_limit_trainer = trainer._time_limit
        if time_limit_trainer is not None and trainer._time_train_start is not None:
            time_left_total = time_limit_trainer - (time.time() - trainer._time_train_start)
        else:
            time_left_total = None

        time_limit_log = f"\ttime_limit = {time_limit:.1f}\t(model)\n" if time_limit else ""
        time_limit_trainer_log = f"\ttime_limit = {time_limit_trainer:.1f}\t(trainer)\n" if time_limit_trainer else ""
        time_left_log = f"\ttime_left  = {time_left_total:.1f}\t(trainer)\n" if time_left_total else ""
        time_used_log = (
            f"\ttime_used  = {time_limit_trainer - time_left_total:.1f}\t(trainer)\n" if time_limit_trainer else ""
        )
        trainer.log(
            20,
            f"{self.__class__.__name__}.before_model_fit\n"
            f"\tmodel      = {model.name}\n"
            f"{time_limit_log}"
            f"{time_limit_trainer_log}"
            f"{time_left_log}"
            f"{time_used_log}"
            f"\tmodels_fit = {len(trainer.get_model_names())}\n"
            f"\tstack_name = {stack_name}\n"
            f"\tlevel      = {level}",
        )

        return False, False

    def _after_model_fit(
        self,
        trainer: AbstractTrainer,
        **kwargs,
    ) -> bool:
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            trainer.log(20, f"{self.__class__.__name__}.after_model_fit | Leaderboard:\n{trainer.leaderboard()}")
        return False

    def before_trainer_fit(self, trainer: AbstractTrainer, **kwargs):
        trainer.log(20, f"{self.__class__.__name__}.before_trainer_fit | kwargs:\n")
        for k, v in kwargs.items():
            trainer.log(20, f"\t{k}={v}")

    def after_trainer_fit(self, trainer: AbstractTrainer):
        trainer.log(20, f"{self.__class__.__name__}.after_trainer_fit | Training Complete")
