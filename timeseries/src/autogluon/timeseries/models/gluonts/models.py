import re
from typing import Type

import mxnet as mx

from autogluon.core.utils import warning_filter

from ..abstract.abstract_timeseries_model import AbstractTimeSeriesModelFactory

with warning_filter():
    import gluonts.model.deepar
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.estimator import Estimator as GluonTSEstimator, DummyEstimator
    from gluonts.model.prophet import ProphetPredictor
    from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.model.transformer import TransformerEstimator
    from gluonts.mx.context import get_mxnet_context
    from gluonts.nursery.autogluon_tabular import TabularEstimator

from .abstract_gluonts import AbstractGluonTSModel


# HACK: DeepAR currently raises an exception when it finds a frequency it doesn't like.
#  we monkey-patch the get_lags and features functions here to return a default
#  instead of failing. We can remove this after porting to pytorch + patching GluonTS
def get_lags_for_frequency_safe(*args, **kwargs):
    from gluonts.time_feature import get_lags_for_frequency

    try:
        return get_lags_for_frequency(*args, **kwargs)
    except Exception as e:
        if "invalid frequency" not in str(e):
            raise
        return get_lags_for_frequency(freq_str="A")


def time_features_from_frequency_str_safe(*args, **kwargs):
    from gluonts.time_feature import time_features_from_frequency_str

    try:
        return time_features_from_frequency_str(*args, **kwargs)
    except Exception as e:
        if "Unsupported frequency" not in str(e):
            raise
        return []


gluonts.model.deepar._estimator.get_lags_for_frequency = get_lags_for_frequency_safe
gluonts.model.deepar._estimator.time_features_from_frequency_str = time_features_from_frequency_str_safe


class DeepARModel(AbstractGluonTSModel):
    """DeepAR model from Gluon-TS.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers : int, default = 2
        Number of RNN layers
    num_cells : int, default = 40
        Number of RNN cells for each layer
    epochs : int, default = 100
        Number of epochs the model will be trained for
    cell_type : str, default = "lstm"
        Type of recurrent cells to use (available: 'lstm' or 'gru')
    dropoutcell_type : typing.Type, default = ZoneoutCell
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
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = DeepAREstimator


class AbstractGluonTSSeq2SeqModel(AbstractGluonTSModel):
    """Abstract class for MQCNN and MQRNN which require hybridization to be turned off
    when fitting on the GPU.
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = None

    def _get_estimator(self):
        if get_mxnet_context() != mx.context.cpu():
            self.params["hybridize"] = False

        with warning_filter():
            return self.gluonts_estimator_class.from_hyperparameters(**self._get_estimator_init_args())


class MQCNNModel(AbstractGluonTSSeq2SeqModel):
    """MQCNN model from Gluon-TS. MQCNN is an encoder-decoder model where the encoder is a
    1D convolutional neural network and the decoder is a multilayer perceptron.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    embedding_dimension : int, optional
        Dimension of the embeddings for categorical features. (default: [min(50, (cat+1)//2) for cat in cardinality])
    add_time_feature : bool, default = True
        Adds a set of time features.
    add_age_feature : bool, default = False
        Adds an age feature.
        The age feature starts with a small value at the start of the time series and grows over time.
    epochs : int, default = 100
        Number of epochs the model will be trained for
    seed : int, optional
        Will set the specified int seed for numpy and MXNet if specified.
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
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = MQCNNEstimator


class MQRNNModel(AbstractGluonTSSeq2SeqModel):
    """MQRNN model from Gluon-TS. MQRNN is an encoder-decoder model where the encoder is a
    recurrent neural network and the decoder is a multilayer perceptron.

    See `AbstractGluonTSModel` for common parameters.
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = MQRNNEstimator


class SimpleFeedForwardModel(AbstractGluonTSModel):
    """SimpleFeedForward model, i.e. a simple multilayer perceptron for
     probabilistic forecasts, from GluonTS.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    num_hidden_dimensions : int, default = [40, 40]
        Number of hidden nodes in each layer
    context_length : int, optional
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    distr_output : gluonts.mx.DistributionOutput, default = StudentTOutput()
        Distribution to fit
    batch_normalization : bool, default = False
        Whether to use batch normalization
    mean_scaling : bool, default = True
        Scale the network input by the data mean and the network output by
        its inverse
    epochs : int, default = 100
        Number of epochs the model will be trained for
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = SimpleFeedForwardEstimator


class TransformerModel(AbstractGluonTSModel):
    """GluonTS Transformer model for forecasting, close to the one described in
    [Vaswani2017]_.

    .. [Vaswani2017] Vaswani, Ashish, et al. "Attention is all you need."
        Advances in neural information processing systems. 2017.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    context_length : int, optional
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    trainer : Trainer, default = Trainer()
        Trainer object to be used
    dropout_rate : float, default = 0.1
        Dropout regularization parameter
    distr_output : gluonts.mx.DistributionOutput, default = StudentTOutput()
        Distribution to use to evaluate observations and sample predictions
    model_dim : int, default = 32
        Dimension of the transformer network, i.e., embedding dimension of the
        input
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
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = TransformerEstimator


class GenericGluonTSModel(AbstractGluonTSModel):
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

    def _get_estimator(self):
        # TODO: temporarily disabling hybridization on GPU due to mxnet issue
        # TODO: fixed in mxnet v2.0
        if get_mxnet_context() != mx.context.cpu():
            self.params["hybridize"] = False

        with warning_filter():
            return self.gluonts_estimator_class.from_hyperparameters(**self._get_estimator_init_args())


class GenericGluonTSModelFactory(AbstractTimeSeriesModelFactory):
    """Factory class for GenericGluonTSModel for convenience of use"""

    def __init__(self, gluonts_estimator_class: Type[GluonTSEstimator], **kwargs):
        self.gluonts_estimator_class = gluonts_estimator_class
        self.init_kwargs = kwargs

    def __call__(self, **kwargs):
        model_init_kwargs = self.init_kwargs.copy()
        model_init_kwargs.update(kwargs)
        return GenericGluonTSModel(
            gluonts_estimator_class=self.gluonts_estimator_class,
            **model_init_kwargs,
        )


class _ProphetDummyEstimator(DummyEstimator):
    def train(self, train_data, validation_data=None, **kwargs):
        return self.predictor


class ProphetModel(AbstractGluonTSModel):
    """Wrapper around `Prophet <https://github.com/facebook/prophet>`_, which wraps the
    library through GluonTS's wrapper.

    In order to use it you need to install the package::

        pip install fbprophet

    Other Parameters
    ----------------
    hyperparameters : Dict[str, Any]
        Model hyperparameters that will be passed directly to the `Prophet`
        class. See Prophet documentation for available parameters.
    """

    gluonts_estimator_class = _ProphetDummyEstimator
    allowed_prophet_parameters = [
        "growth",
        "changepoints",
        "n_changepoints",
        "changepoint_range",
        "yearly_seasonality",
        "weekly_seasonality",
        "daily_seasonality",
        "holidays",
        "seasonality_mode",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "changepoint_prior_scale",
        "mcmc_samples",
        "interval_width",
        "uncertainty_samples",
    ]

    def _get_estimator(self) -> GluonTSEstimator:
        model_init_params = self._get_estimator_init_args()

        return _ProphetDummyEstimator(
            predictor_cls=ProphetPredictor,
            freq=model_init_params["freq"],
            prediction_length=model_init_params["prediction_length"],
            prophet_params={k: v for k, v in model_init_params.items() if k in self.allowed_prophet_parameters},
        )


# TODO: AutoGluon Tabular will be removed from GluonTS to avoid circular dependencies
class AutoTabularModel(AbstractGluonTSModel):
    """Autotabular model from Gluon-TS, which in turn uses autogluon.tabular
    predictors for fitting a forecast model.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    lag_indices : List[int], optional
        List of indices of the lagged observations to use as features. If
        None, this will be set automatically based on the frequency.
    scaling : Callable[[pd.Series], Tuple[pd.Series, float]], optional
        Function to be used to scale time series. This should take a pd.Series object
        as input, and return a scaled pd.Series and the scale (float). By default,
        this divides a series by the mean of its absolute value.
    disable_auto_regression : bool, default = False
        Weather to forcefully disable auto-regression in the model. If ``True``,
        this will remove any lag index which is smaller than ``prediction_length``.
        This will make predictions more efficient, but may impact their accuracy.
    """

    # TODO: AutoTabular model is experimental, may need its own logic for
    # TODO: handling time limit and training data. See PR #1538.
    def _get_estimator(self):
        return TabularEstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            time_limit=self.params_aux["time_limit"],
            last_k_for_val=self.prediction_length,
        )
