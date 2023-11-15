"""
Module including wrappers for PyTorch implementations of models in GluonTS
"""
import logging
from typing import Any, Dict, Type

from gluonts.model.estimator import Estimator as GluonTSEstimator

from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.utils.datetime import (
    get_lags_for_frequency,
    get_seasonality,
    get_time_features_for_frequency,
)

# NOTE: We avoid imports for torch and lightning.pytorch at the top level and hide them inside class methods.
# This is done to skip these imports during multiprocessing (which may cause bugs)

# FIXME: introduces cpflows dependency. We exclude this model until a future release.
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

# FIXME: DeepNPTS does not implement the GluonTS PyTorch API, and does not use
# PyTorch Lightning. We exclude this model until a future release.
# from gluonts.torch.model.deep_npts import DeepNPTSEstimator


logger = logging.getLogger(__name__)


class DeepARModel(AbstractGluonTSModel):
    """Autoregressive forecasting model based on a recurrent neural network [Salinas2020]_.

    Based on `gluonts.torch.model.deepar.DeepAREstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Salinas2020] Salinas, David, et al.
        "DeepAR: Probabilistic forecasting with autoregressive recurrent networks."
        International Journal of Forecasting. 2020.


    Other Parameters
    ----------------
    context_length : int, default = max(10, 2 * prediction_length)
        Number of steps to unroll the RNN for before computing predictions
    disable_static_features : bool, default = False
        If True, static features won't be used by the model even if they are present in the dataset.
        If False, static features will be used by the model if they are present in the dataset.
    disable_known_covariates : bool, default = False
        If True, known covariates won't be used by the model even if they are present in the dataset.
        If False, known covariates will be used by the model if they are present in the dataset.
    num_layers : int, default = 2
        Number of RNN layers
    hidden_size : int, default = 40
        Number of RNN cells for each layer
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    embedding_dimension : int, optional
        Dimension of the embeddings for categorical features
        (if None, defaults to [min(50, (cat+1)//2) for cat in cardinality])
    distr_output : gluonts.torch.distributions.DistributionOutput, default = StudentTOutput()
        Distribution to use to evaluate observations and sample predictions
    scaling: bool, default = True
        Whether to automatically scale the target values
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    predict_batch_size : int, default = 500
        Size of batches used during prediction.
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    trainer_kwargs : dict, optional
        Optional keyword arguments passed to ``lightning.Trainer``.
    early_stopping_patience : int or None, default = 20
        Early stop training if the validation loss doesn't improve for this many epochs.
    """

    supports_known_covariates = True

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.deepar import DeepAREstimator

        return DeepAREstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs["num_feat_static_cat"] = self.num_feat_static_cat
        init_kwargs["num_feat_static_real"] = self.num_feat_static_real
        init_kwargs["cardinality"] = self.feat_static_cat_cardinality
        init_kwargs["num_feat_dynamic_real"] = self.num_feat_dynamic_real
        init_kwargs.setdefault("lags_seq", get_lags_for_frequency(self.freq))
        init_kwargs.setdefault("time_features", get_time_features_for_frequency(self.freq))
        return init_kwargs


class SimpleFeedForwardModel(AbstractGluonTSModel):
    """Simple feedforward neural network that simultaneously predicts all future values.

    Based on `gluonts.torch.model.simple_feedforward.SimpleFeedForwardEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.simple_feedforward.html>`_.
    See GluonTS documentation for additional hyperparameters.


    Other Parameters
    ----------------
    context_length : int, default = max(10, 2 * prediction_length)
        Number of time units that condition the predictions
    hidden_dimensions: List[int], default = [20, 20]
        Size of hidden layers in the feedforward network
    distr_output : gluonts.torch.distributions.DistributionOutput, default = StudentTOutput()
        Distribution to fit.
    batch_normalization : bool, default = False
        Whether to use batch normalization
    mean_scaling : bool, default = True
        Scale the network input by the data mean and the network output by its inverse
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    predict_batch_size : int, default = 500
        Size of batches used during prediction.
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    trainer_kwargs : dict, optional
        Optional keyword arguments passed to ``lightning.Trainer``.
    early_stopping_patience : int or None, default = 20
        Early stop training if the validation loss doesn't improve for this many epochs.
    """

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

        return SimpleFeedForwardEstimator


class TemporalFusionTransformerModel(AbstractGluonTSModel):
    """Combines LSTM with a transformer layer to predict the quantiles of all future target values [Lim2021]_.

    Based on `gluonts.torch.model.tft.TemporalFusionTransformerEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.tft.html>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Lim2021] Lim, Bryan, et al.
        "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting."
        International Journal of Forecasting. 2021.


    Other Parameters
    ----------------
    context_length : int, default = max(64, 2 * prediction_length)
        Number of past values used for prediction.
    disable_static_features : bool, default = False
        If True, static features won't be used by the model even if they are present in the dataset.
        If False, static features will be used by the model if they are present in the dataset.
    disable_known_covariates : bool, default = False
        If True, known covariates won't be used by the model even if they are present in the dataset.
        If False, known covariates will be used by the model if they are present in the dataset.
    disable_past_covariates : bool, default = False
        If True, past covariates won't be used by the model even if they are present in the dataset.
        If False, past covariates will be used by the model if they are present in the dataset.
    hidden_dim : int, default = 32
        Size of the LSTM & transformer hidden states.
    variable_dim : int, default = 32
        Size of the feature embeddings.
    num_heads : int, default = 4
        Number of attention heads in self-attention layer in the decoder.
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    predict_batch_size : int, default = 500
        Size of batches used during prediction.
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    trainer_kwargs : dict, optional
        Optional keyword arguments passed to ``lightning.Trainer``.
    early_stopping_patience : int or None, default = 20
        Early stop training if the validation loss doesn't improve for this many epochs.
    """

    supports_known_covariates = True
    supports_past_covariates = True

    @property
    def default_context_length(self) -> int:
        return max(64, 2 * self.prediction_length)

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.tft import TemporalFusionTransformerEstimator

        return TemporalFusionTransformerEstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        if self.num_feat_dynamic_real > 0:
            init_kwargs["dynamic_dims"] = [self.num_feat_dynamic_real]
        if self.num_past_feat_dynamic_real > 0:
            init_kwargs["past_dynamic_dims"] = [self.num_past_feat_dynamic_real]
        if self.num_feat_static_real > 0:
            init_kwargs["static_dims"] = [self.num_feat_static_real]
        if len(self.feat_static_cat_cardinality):
            init_kwargs["static_cardinalities"] = self.feat_static_cat_cardinality
        init_kwargs.setdefault("time_features", get_time_features_for_frequency(self.freq))
        return init_kwargs


class DLinearModel(AbstractGluonTSModel):
    """Simple feedforward neural network that subtracts trend before forecasting [Zeng2023]_.

    Based on `gluonts.torch.model.d_linear.DLinearEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.d_linear.html>`_.
    See GluonTS documentation for additional hyperparameters.

    References
    ----------
    .. [Zeng2023] Zeng, Ailing, et al.
        "Are transformers effective for time series forecasting?"
        AAAI Conference on Artificial Intelligence. 2023.

    Other Parameters
    ----------------
    context_length : int, default = 96
        Number of time units that condition the predictions
    hidden_dimension: int, default = 20
        Size of hidden layers in the feedforward network
    distr_output : gluonts.torch.distributions.DistributionOutput, default = StudentTOutput()
        Distribution to fit.
    scaling : {"mean", "std", None}, default = "mean"
        Scaling applied to the inputs. One of ``"mean"`` (mean absolute scaling), ``"std"`` (standardization), ``None`` (no scaling).
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    predict_batch_size : int, default = 500
        Size of batches used during prediction.
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    trainer_kwargs : dict, optional
        Optional keyword arguments passed to ``lightning.Trainer``.
    early_stopping_patience : int or None, default = 20
        Early stop training if the validation loss doesn't improve for this many epochs.
    weight_decay : float, default = 1e-8
        Weight decay regularization parameter.
    """

    @property
    def default_context_length(self) -> int:
        return 96

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.d_linear import DLinearEstimator

        return DLinearEstimator


class PatchTSTModel(AbstractGluonTSModel):
    """Transformer-based forecaster that segments each time series into patches [Nie2023]_.

    Based on `gluonts.torch.model.d_linear.PatchTSTEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.patch_tst.html>`_.
    See GluonTS documentation for additional hyperparameters.

    References
    ----------
    .. [Nie2023] Nie, Yuqi, et al.
        "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
        International Conference on Learning Representations. 2023.

    Other Parameters
    ----------------
    context_length : int, default = 96
        Number of time units that condition the predictions
    patch_len : int, default = 16
        Length of the patch.
    stride : int, default = 8
        Stride of the patch.
    d_model : int, default = 32
        Size of hidden layers in the Transformer encoder.
    nhead : int, default = 4
        Number of attention heads in the Transformer encoder which must divide d_model.
    num_encoder_layers : int, default = 2
        Number of layers in the Transformer encoder.
    distr_output : gluonts.torch.distributions.DistributionOutput, default = StudentTOutput()
        Distribution to fit.
    scaling : {"mean", "std", None}, default = "mean"
        Scaling applied to the inputs. One of ``"mean"`` (mean absolute scaling), ``"std"`` (standardization), ``None`` (no scaling).
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    weight_decay : float, default = 1e-8
        Weight decay regularization parameter.
    """

    @property
    def default_context_length(self) -> int:
        return 96

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.patch_tst import PatchTSTEstimator

        return PatchTSTEstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs.setdefault("patch_len", 16)
        return init_kwargs


class WaveNetModel(AbstractGluonTSModel):
    """WaveNet estimator that uses the architecture proposed in [Oord2016] with quantized targets.

    The model is based on a CNN architecture with dilated convolutions. Time series values are quantized into buckets
    and the model is trained using the cross-entropy loss.

    Based on `gluonts.torch.model.wavenet.WaveNetEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.wavenet.html>`_.
    See GluonTS documentation for additional hyperparameters.

    References
    ----------
    .. [Oord2016] Oord, Aaron van den, et al.
        "Wavenet: A generative model for raw audio"
        arXiv preprint arXiv:1609.03499. 2016.

    Other Parameters
    ----------------
    num_bins : int, default = 1024
        Number of bins used for quantization of the time series.
    num_residual_channels : int, default = 24
        Number of residual channels in WaveNet architecture.
    num_skip_channels : int, default = 32
        Number of skip channels in WaveNet architecture, by default 32
    dilation_depth : int or None, default = None
        Number of dilation layers in WaveNet architecture. If set to None (default), dilation_depth is set such that
        the receptive length is at least as long as the ``seasonality`` and at least ``2 * prediction_length``.
    num_stacks : int, default = 1
        Number of dilation stacks in WaveNet architecture.
    temperature : float, default = 1.0
        Temperature used for sampling from softmax distribution.
    seasonality : int, optional
        The seasonality of the time series. By default is set based on the ``freq`` of the data.
    embedding_dimension : int, default = 5
        The dimension of the embeddings for categorical features.
    use_log_scale_feature : bool, default = True
        If True, logarithm of the scale of the past data will be used as an additional static feature.
    negative_data : bool, default = True
        Flag indicating whether the time series take negative values.
    max_epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    predict_batch_size : int, default = 500
        Size of batches used during prediction.
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    lr : float, default = 1e-3,
        Learning rate used during training
    trainer_kwargs : dict, optional
        Optional keyword arguments passed to ``lightning.Trainer``.
    early_stopping_patience : int or None, default = 20
        Early stop training if the validation loss doesn't improve for this many epochs.
    weight_decay : float, default = 1e-8
        Weight decay regularization parameter.
    """

    supports_known_covariates = True

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        from gluonts.torch.model.wavenet import WaveNetEstimator

        return WaveNetEstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs["num_feat_static_cat"] = self.num_feat_static_cat
        init_kwargs["num_feat_static_real"] = self.num_feat_static_real
        init_kwargs["cardinality"] = [1] if self.num_feat_static_cat == 0 else self.feat_static_cat_cardinality
        init_kwargs["num_feat_dynamic_real"] = self.num_feat_dynamic_real
        init_kwargs.setdefault("negative_data", self.negative_data)
        init_kwargs.setdefault("seasonality", get_seasonality(self.freq))
        init_kwargs.setdefault("time_features", get_time_features_for_frequency(self.freq))
        # WaveNet model fails if an unsupported frequency such as "SM" is provided. We provide a dummy freq instead
        init_kwargs["freq"] = "H"
        return init_kwargs
