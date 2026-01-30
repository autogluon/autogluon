from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal

if TYPE_CHECKING:
    from autogluon.common import space

from ._base import SearchableBool, SearchableFloat, SearchableInt
from .mixins import (
    ContextLengthMixIn,
    DeepLearningMixIn,
    DistrOutputMixIn,
    PredictBatchSizeMixIn,
    TrainerMixIn,
)


class DeepARModel(
    ContextLengthMixIn,
    DistrOutputMixIn,
    DeepLearningMixIn,
    TrainerMixIn,
    PredictBatchSizeMixIn,
    total=False,
):
    disable_static_features: SearchableBool
    disable_known_covariates: SearchableBool
    num_layers: SearchableInt
    hidden_size: SearchableInt
    dropout_rate: SearchableFloat
    embedding_dimension: SearchableInt
    max_cat_cardinality: SearchableInt
    scaling: SearchableBool


class DLinearModel(
    ContextLengthMixIn,
    DistrOutputMixIn,
    DeepLearningMixIn,
    TrainerMixIn,
    PredictBatchSizeMixIn,
    total=False,
):
    hidden_dimension: SearchableInt
    scaling: Literal["mean", "std", None] | space.Categorical
    weight_decay: SearchableFloat


class PatchTSTModel(ContextLengthMixIn, DistrOutputMixIn, DeepLearningMixIn, total=False):
    patch_len: SearchableInt
    stride: SearchableInt
    d_model: SearchableInt
    nhead: SearchableInt
    num_encoder_layers: SearchableInt
    scaling: Literal["mean", "std", None] | space.Categorical
    weight_decay: SearchableFloat


class SimpleFeedForwardModel(
    ContextLengthMixIn,
    DistrOutputMixIn,
    DeepLearningMixIn,
    TrainerMixIn,
    PredictBatchSizeMixIn,
    total=False,
):
    hidden_dimensions: List[SearchableInt]
    batch_normalization: SearchableBool
    mean_scaling: SearchableBool


class TemporalFusionTransformerModel(
    ContextLengthMixIn,
    DistrOutputMixIn,
    DeepLearningMixIn,
    TrainerMixIn,
    PredictBatchSizeMixIn,
    total=False,
):
    disable_static_features: SearchableBool
    disable_known_covariates: SearchableBool
    disable_past_covariates: SearchableBool
    hidden_dim: SearchableInt
    variable_dim: SearchableInt
    num_heads: SearchableInt
    dropout_rate: SearchableFloat


class TiDEModel(ContextLengthMixIn, DeepLearningMixIn, TrainerMixIn, PredictBatchSizeMixIn, total=False):
    disable_static_features: SearchableBool
    disable_known_covariates: SearchableBool
    feat_proj_hidden_dim: SearchableInt
    encoder_hidden_dim: SearchableInt
    decoder_hidden_dim: SearchableInt
    temporal_hidden_dim: SearchableInt
    distr_hidden_dim: SearchableInt
    num_layers_encoder: SearchableInt
    num_layers_decoder: SearchableInt
    decoder_output_dim: SearchableInt
    dropout_rate: SearchableFloat
    num_feat_dynamic_proj: SearchableInt
    embedding_dimension: SearchableInt | List[SearchableInt]
    layer_norm: SearchableBool
    scaling: Literal["mean", "std", None] | space.Categorical


class WaveNetModel(DeepLearningMixIn, TrainerMixIn, PredictBatchSizeMixIn, total=False):
    num_bins: SearchableInt
    num_residual_channels: SearchableInt
    num_skip_channels: SearchableInt
    dilation_depth: SearchableInt | None
    num_stacks: SearchableInt
    temperature: SearchableFloat
    seasonality: SearchableInt
    embedding_dimension: SearchableInt
    use_log_scale_feature: SearchableBool
    negative_data: SearchableBool
    max_cat_cardinality: SearchableInt
    weight_decay: SearchableFloat


class Chronos2Model(ContextLengthMixIn, total=False):
    model_path: str
    batch_size: SearchableInt
    device: str | None
    cross_learning: SearchableBool
    fine_tune: SearchableBool
    fine_tune_mode: str
    fine_tune_lr: SearchableFloat
    fine_tune_steps: SearchableInt
    fine_tune_batch_size: SearchableInt
    fine_tune_context_length: SearchableInt
    eval_during_fine_tune: SearchableBool
    fine_tune_eval_max_items: SearchableInt
    fine_tune_lora_config: dict[str, Any]
    fine_tune_trainer_kwargs: dict[str, Any]
    revision: str
    disable_known_covariates: SearchableBool
    disable_past_covariates: SearchableBool


class ChronosModel(ContextLengthMixIn, total=False):
    model_path: str
    batch_size: SearchableInt
    num_samples: SearchableInt
    device: str
    torch_dtype: Any
    data_loader_num_workers: int
    fine_tune: SearchableBool
    fine_tune_lr: SearchableFloat
    fine_tune_steps: SearchableInt
    fine_tune_batch_size: SearchableInt
    fine_tune_shuffle_buffer_size: int
    eval_during_fine_tune: SearchableBool
    fine_tune_eval_max_items: SearchableInt
    fine_tune_trainer_kwargs: dict[str, Any]
    keep_transformers_logs: SearchableBool
    revision: str


class TotoModel(ContextLengthMixIn, total=False):
    model_path: str
    batch_size: SearchableInt
    num_samples: SearchableInt
    device: str
    compile_model: bool
