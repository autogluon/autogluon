import logging
import re
from typing import Callable, List, Type

import gluonts
import mxnet as mx

from autogluon.core.utils import warning_filter
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract.abstract_timeseries_model import AbstractTimeSeriesModelFactory
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel

with warning_filter():
    from gluonts.model.estimator import Estimator as GluonTSEstimator
    from gluonts.dataset.field_names import FieldName
    from gluonts.mx.context import get_mxnet_context
    from gluonts.mx.model.deepar import DeepAREstimator
    from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.mx.model.transformer import TransformerEstimator
    from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
    from gluonts.mx.model.seq2seq import MQCNNEstimator, MQRNNEstimator

from .callback import GluonTSEarlyStoppingCallback, TimeLimitCallback

logger = logging.getLogger(__name__)
gts_logger = logging.getLogger(gluonts.__name__)


class AbstractGluonTSMXNetModel(AbstractGluonTSModel):
    def _get_callbacks(self, time_limit: int, *args, **kwargs) -> List[Callable]:
        callbacks = [TimeLimitCallback(time_limit)]

        early_stopping_patience = self._get_model_params().get("early_stopping_patience", None)
        if early_stopping_patience:
            callbacks.append(GluonTSEarlyStoppingCallback(early_stopping_patience))

        return callbacks


class DeepARMXNetModel(AbstractGluonTSMXNetModel):
    """DeepAR model from GluonTS based on the MXNet backend.

    The model consists of an RNN encoder (LSTM or GRU) and a decoder that outputs the
    distribution of the next target value. Close to the model described in [Salinas2020]_.


    Based on `gluonts.mx.model.deepar.DeepAREstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.deepar.html>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Salinas2020] Salinas, David, et al.
        "DeepAR: Probabilistic forecasting with autoregressive recurrent networks."
        International Journal of Forecasting. 2020.


    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    disable_static_features : bool, default = False
        If True, static features won't be used by the model even if they are present in the dataset.
        If False, static features will be used by the model if they are present in the dataset.
    disable_known_covariates : bool, default = False
        If True, known covariates won't be used by the model even if they are present in the dataset.
        If False, known covariates will be used by the model if they are present in the dataset.
    num_layers : int, default = 2
        Number of RNN layers
    num_cells : int, default = 40
        Number of RNN cells for each layer
    cell_type : str, default = "lstm"
        Type of recurrent cells to use (available: 'lstm' or 'gru')
    dropoutcell_type : str, default = 'ZoneoutCell'
        Type of dropout cells to use
        (available: 'ZoneoutCell', 'RNNZoneoutCell', 'VariationalDropoutCell' or
        'VariationalZoneoutCell')
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    embedding_dimension : int, optional
        Dimension of the embeddings for categorical features
        (if None, defaults to [min(50, (cat+1)//2) for cat in cardinality])
    distr_output : gluonts.mx.DistributionOutput, default = StudentTOutput()
        Distribution to use to evaluate observations and sample predictions
    scaling: bool, default = True
        Whether to automatically scale the target values
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = DeepAREstimator
    default_num_samples: int = 250
    supports_known_covariates = True

    def _get_estimator_init_args(self) -> dict:
        init_kwargs = super()._get_estimator_init_args()
        # Our API hides these model kwargs from the user. They can only be controlled through disable_static_features
        # and disable_known_covariates
        init_kwargs["use_feat_static_cat"] = self.num_feat_static_cat > 0
        init_kwargs["use_feat_static_real"] = self.num_feat_static_real > 0
        init_kwargs["cardinality"] = self.feat_static_cat_cardinality
        init_kwargs["use_feat_dynamic_real"] = self.num_feat_dynamic_real > 0
        return init_kwargs


class AbstractGluonTSSeq2SeqModel(AbstractGluonTSMXNetModel):
    """Abstract class for MQCNN and MQRNN which require hybridization to be turned off
    when fitting on the GPU.
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = None

    def _get_estimator_init_args(self):
        init_kwargs = super()._get_estimator_init_args()
        if get_mxnet_context() != mx.context.cpu():
            init_kwargs["hybridize"] = False
        return init_kwargs


class MQCNNMXNetModel(AbstractGluonTSSeq2SeqModel):
    """MQCNN model from GluonTS.

    The model consists of a CNN encoder and a decoder that directly predicts the
    quantiles of the future target values' distribution. As described in [Wen2017]_.

    Based on `gluonts.mx.model.seq2seq.MQCNNEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.seq2seq.html#gluonts.mx.model.seq2seq.MQCNNEstimator>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Wen2017] Wen, Ruofeng, et al.
        "A multi-horizon quantile recurrent forecaster."
        arXiv preprint arXiv:1711.11053 (2017)


    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    disable_static_features : bool, default = False
        If True, static features won't be used by the model even if they are present in the dataset.
        If False, static features will be used by the model if they are present in the dataset.
    disable_known_covariates : bool, default = False
        If True, known covariates won't be used by the model even if they are present in the dataset.
        If False, known covariates will be used by the model if they are present in the dataset.
    embedding_dimension : int, optional
        Dimension of the embeddings for categorical features. (default: [min(50, (cat+1)//2) for cat in cardinality])
    add_time_feature : bool, default = True
        Adds a set of time features.
    add_age_feature : bool, default = False
        Adds an age feature.
        The age feature starts with a small value at the start of the time series and grows over time.
    decoder_mlp_dim_seq : List[int], default = [30]
        The dimensionalities of the Multi Layer Perceptron layers of the decoder.
    channels_seq : List[int], default = [30, 30, 30]
        The number of channels (i.e. filters or convolutions) for each layer of the HierarchicalCausalConv1DEncoder.
        More channels usually correspond to better performance and larger network size.
    dilation_seq : List[int], default = [1, 3, 5]
        The dilation of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        Greater numbers correspond to a greater receptive field of the network, which is usually
        better with longer context_length. (Same length as channels_seq)
    kernel_size_seq : List[int], default = [7, 3, 3]
        The kernel sizes (i.e. window size) of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        (Same length as channels_seq)
    use_residual : bool, default = True
        Whether the hierarchical encoder should additionally pass the unaltered
        past target to the decoder.
    quantiles : List[float], default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        The list of quantiles that will be optimized for, and predicted by, the model.
        Optimizing for more quantiles than are of direct interest to you can result
        in improved performance due to a regularizing effect.
    distr_output : gluonts.mx.DistributionOutput, optional
        DistributionOutput to use. Only one between `quantile` and `distr_output`
        can be set.
    scaling : bool, optional
        Whether to automatically scale the target values. (default: False if quantile_output is used,
        True otherwise)
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = MQCNNEstimator
    supports_known_covariates = True
    supports_past_covariates = True

    def _get_estimator_init_args(self) -> dict:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs["use_feat_static_cat"] = self.num_feat_static_cat > 0
        init_kwargs["use_feat_static_real"] = self.num_feat_static_real > 0
        init_kwargs["cardinality"] = self.feat_static_cat_cardinality
        init_kwargs["use_feat_dynamic_real"] = self.num_feat_dynamic_real > 0
        init_kwargs["use_past_feat_dynamic_real"] = self.num_past_feat_dynamic_real > 0
        return init_kwargs


class MQRNNMXNetModel(AbstractGluonTSSeq2SeqModel):
    """MQRNN model from GluonTS.

    The model consists of an RNN encoder and a decoder that directly predicts the
    quantiles of the future target values' distribution. As described in [Wen2017]_.

    Based on `gluonts.mx.model.seq2seq.MQRNNEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.seq2seq.html#gluonts.mx.model.seq2seq.MQRNNEstimator>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Wen2017] Wen, Ruofeng, et al.
        "A multi-horizon quantile recurrent forecaster."
        arXiv preprint arXiv:1711.11053 (2017)


    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    embedding_dimension : int, optional
        Dimension of the embeddings for categorical features. (default: [min(50, (cat+1)//2) for cat in cardinality])
    decoder_mlp_dim_seq : List[int], default = [30]
        The dimensionalities of the Multi Layer Perceptron layers of the decoder.
    quantiles : List[float], default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        The list of quantiles that will be optimized for, and predicted by, the model.
        Optimizing for more quantiles than are of direct interest to you can result
        in improved performance due to a regularizing effect.
    distr_output : gluonts.mx.DistributionOutput, optional
        DistributionOutput to use. Only one between `quantile` and `distr_output`
        can be set.
    scaling : bool, optional
        Whether to automatically scale the target values. (default: False if quantile_output is used,
        True otherwise)
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = MQRNNEstimator
    supports_known_covariates = True


class SimpleFeedForwardMXNetModel(AbstractGluonTSMXNetModel):
    """SimpleFeedForward model from GluonTS based on the MXNet backend.

    The model consists of a multilayer perceptron (MLP) that predicts the distribution
    of the next target value.

    Based on `gluonts.mx.model.simple_feedforward.SimpleFeedForwardEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.simple_feedforward.html>`_.
    See GluonTS documentation for additional hyperparameters.

    Note that AutoGluon uses hyperparameters ``hidden_dim`` and ``num_layers`` instead of ``num_hidden_dimensions``
    used in GluonTS. This is done to ensure compatibility with Ray Tune.


    Other Parameters
    ----------------
    context_length : int, optional
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    hidden_dim: int, default = 40
        Number of hidden units in each layer of the MLP
    num_layers : int, default = 2
        Number of hidden layers in the MLP
    distr_output : gluonts.mx.DistributionOutput, default = StudentTOutput()
        Distribution to fit
    batch_normalization : bool, default = False
        Whether to use batch normalization
    mean_scaling : bool, default = True
        Scale the network input by the data mean and the network output by
        its inverse
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = SimpleFeedForwardEstimator

    def _get_estimator_init_args(self):
        init_kwargs = super()._get_estimator_init_args()
        # Workaround: Ray Tune doesn't support lists as hyperparameters, so we build `num_hidden_dimensions`
        # from `hidden_dim` and `num_layers`
        if "num_hidden_dimensions" in init_kwargs:
            logger.warning(
                f"Hyperparameter 'num_hidden_dimensions' is ignored by {self.name}. "
                f"Please use hyperparameters 'hidden_dim' and 'num_layers' instead."
            )
        hidden_dim = init_kwargs.pop("hidden_dim", 40)
        num_layers = init_kwargs.pop("num_layers", 2)
        init_kwargs["num_hidden_dimensions"] = [hidden_dim] * num_layers
        return init_kwargs


class TemporalFusionTransformerMXNetModel(AbstractGluonTSMXNetModel):
    """TemporalFusionTransformer model from GluonTS.

    The model combines an LSTM encoder, a transformer decoder, and directly predicts
    the quantiles of future target values. As described in [Lim2021]_.

    Based on `gluonts.mx.model.tft.TemporalFusionTransformerEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.tft.html>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Lim2021] Lim, Bryan, et al.
        "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting."
        International Journal of Forecasting. 2021.


    Other Parameters
    ----------------
    context_length : int or None, default = None
        Number of past values used for prediction.
        (default: None, in which case context_length = prediction_length)
    hidden_dim : int, default = 32
        Size of the hidden layer.
    num_heads : int, default = 4
        Number of attention heads in multi-head attention.
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = TemporalFusionTransformerEstimator
    supported_quantiles: set = set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    supports_known_covariates = True
    supports_past_covariates = True

    def _get_estimator_init_args(self) -> dict:
        init_kwargs = super()._get_estimator_init_args()
        if self.num_feat_static_real > 0:
            init_kwargs["static_feature_dims"] = {FieldName.FEAT_STATIC_REAL: self.num_feat_static_real}
        if self.num_feat_dynamic_real > 0 or self.num_past_feat_dynamic_real > 0:
            init_kwargs["dynamic_feature_dims"] = {}
            if self.num_feat_dynamic_real > 0:
                init_kwargs["dynamic_feature_dims"][FieldName.FEAT_DYNAMIC_REAL] = self.num_feat_dynamic_real
            if self.num_past_feat_dynamic_real > 0:
                init_kwargs["dynamic_feature_dims"][FieldName.PAST_FEAT_DYNAMIC_REAL] = self.num_past_feat_dynamic_real
                init_kwargs["past_dynamic_features"] = [FieldName.PAST_FEAT_DYNAMIC_REAL]

        # Turning off hybridization prevents MXNet errors when training on GPU
        init_kwargs["hybridize"] = False
        # TFT cannot handle arbitrary quantiles, this is a workaround
        init_kwargs["num_outputs"] = 9

        if not set(self.quantile_levels).issubset(self.supported_quantiles):
            raise ValueError(
                f"{self.name} requires that quantile_levels are a subset of "
                f"{self.supported_quantiles} (received quantile_levels = {self.quantile_levels})"
            )
        return init_kwargs

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        if quantile_levels is not None and not set(quantile_levels).issubset(self.supported_quantiles):
            raise ValueError(
                f"{self.name} requires that quantile_levels are a subset of "
                f"{self.supported_quantiles} (received quantile_levels = {self.quantile_levels})"
            )
        return super().predict(data=data, quantile_levels=quantile_levels, **kwargs)


class TransformerMXNetModel(AbstractGluonTSMXNetModel):
    """Autoregressive transformer forecasting model from GluonTS.

    The model consists of an Transformer encoder and a decoder that outputs the
    distribution of the next target value. The transformer architecture is close to the
    one described in [Vaswani2017]_.

    Based on `gluonts.mx.model.transformer.TransformerEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.mx.model.transformer.html>`_.
    See GluonTS documentation for additional hyperparameters.


    References
    ----------
    .. [Vaswani2017] Vaswani, Ashish, et al. "Attention is all you need."
        Advances in neural information processing systems. 2017.


    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    model_dim : int, default = 32
        Dimension of the transformer network, i.e., embedding dimension of the
        input
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    distr_output : gluonts.mx.DistributionOutput, default = StudentTOutput()
        Distribution to use to evaluate observations and sample predictions
    inner_ff_dim_scale : int, default = 4
        Dimension scale of the inner hidden layer of the transformer's
        feedforward network
    pre_seq : str, default = "dn"
        Sequence that defined operations of the processing block before the
        main transformer network. Available operations: 'd' for dropout, 'r'
        for residual connections and 'n' for normalization
    post_seq : str, default = "drn"
        Sequence that defined operations of the processing block in and after
        the main transformer network. Available operations: 'd' for
        dropout, 'r' for residual connections and 'n' for normalization
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    # TODO: Enable static and dynamic features
    gluonts_estimator_class: Type[GluonTSEstimator] = TransformerEstimator


class GenericGluonTSMXNetModel(AbstractGluonTSMXNetModel):
    """Generic wrapper model class for GluonTS models (in GluonTS terminology---
    Estimators). While this class is meant to generally enable fast use of GluonTS
    models in autogluon, specific GluonTS models accessed through this wrapper may
    not have been tested and should be used at the user's own risk.

    Please refer to each GluonTS estimator's individual documentation for
    initialization parameters of each model.

    Parameters
    ----------
    gluonts_estimator_class : Type[gluonts.model.estimator.Estimator]
        The class object of the GluonTS estimator to be used.
    """

    def __init__(self, gluonts_estimator_class: Type[GluonTSEstimator], **kwargs):
        self.gluonts_estimator_class = gluonts_estimator_class
        gluonts_model_name = re.sub(r"Estimator$", "", self.gluonts_estimator_class.__name__)

        super().__init__(name=kwargs.pop("name", gluonts_model_name), **kwargs)

    def get_params(self) -> dict:
        params_dict = super().get_params()
        params_dict["gluonts_estimator_class"] = self.gluonts_estimator_class
        return params_dict

    def _get_estimator_init_args(self):
        init_kwargs = super()._get_estimator_init_args()
        if get_mxnet_context() != mx.context.cpu():
            init_kwargs["hybridize"] = False
        return init_kwargs


class GenericGluonTSMXNetModelFactory(AbstractTimeSeriesModelFactory):
    """Factory class for GenericGluonTSModel for convenience of use"""

    def __init__(self, gluonts_estimator_class: Type[GluonTSEstimator], **kwargs):
        self.gluonts_estimator_class = gluonts_estimator_class
        self.init_kwargs = kwargs

    def __call__(self, **kwargs):
        model_init_kwargs = self.init_kwargs.copy()
        model_init_kwargs.update(kwargs)
        return GenericGluonTSMXNetModel(
            gluonts_estimator_class=self.gluonts_estimator_class,
            **model_init_kwargs,
        )
