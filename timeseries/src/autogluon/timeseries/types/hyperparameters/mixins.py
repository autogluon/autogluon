from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

from ._base import (
    SearchableBool,
    SearchableFloat,
    SearchableInt,
)

if TYPE_CHECKING:
    from gluonts.torch.distributions import Output as GluonTSOutput

    from autogluon.common import space


class NJobsMixIn(TypedDict, total=False):
    n_jobs: int | float


class SeasonalPeriodMixIn(TypedDict, total=False):
    seasonal_period: None | SearchableInt


class MaxTsLengthMixIn(TypedDict, total=False):
    max_ts_length: None | SearchableInt


class ModelMixIn(TypedDict, total=False):
    model: str | space.Categorical


class DampedMixIn(TypedDict, total=False):
    damped: SearchableBool


class ContextLengthMixIn(TypedDict, total=False):
    context_length: SearchableInt | None


class DistrOutputMixIn(TypedDict, total=False):
    distr_output: GluonTSOutput | space.Categorical


class DeepLearningMixIn(TypedDict, total=False):
    max_epochs: SearchableInt
    batch_size: SearchableInt
    num_batches_per_epoch: SearchableInt
    lr: SearchableFloat
    keep_lightning_logs: bool


class TrainerMixIn(TypedDict, total=False):
    trainer_kwargs: dict[str, Any]
    early_stopping_patience: SearchableInt | None


class PredictBatchSizeMixIn(TypedDict, total=False):
    predict_batch_size: SearchableInt


class TabularModelMixIn(TypedDict, total=False):
    date_features: list[str | Callable] | None
    target_scaler: Literal["standard", "mean_abs", "min_max", "robust"] | None
    model_name: str | space.Categorical
    # TODO: Fill in with model-specific hyperparameters
    model_hyperparameters: dict[str, Any]
    max_num_items: int | None
    max_num_samples: int | None
