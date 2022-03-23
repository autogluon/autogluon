import re
from typing import Type

import mxnet as mx

from autogluon.core.utils import warning_filter

with warning_filter():
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.estimator import Estimator as GluonTSEstimator, DummyEstimator
    from gluonts.model.prophet import ProphetPredictor
    from gluonts.model.seq2seq import MQCNNEstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.mx.context import get_mxnet_context
    from gluonts.nursery.autogluon_tabular import TabularEstimator

from .abstract_gluonts import AbstractGluonTSModel


class DeepARModel(AbstractGluonTSModel):
    """DeepAR model from Gluon-TS.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers: int
        Number of RNN layers (default: 2)
    num_cells: int
        Number of RNN cells for each layer (default: 40)
    cell_type: str
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    dropoutcell_type: typing.Type
        Type of dropout cells to use
        (available: 'ZoneoutCell', 'RNNZoneoutCell', 'VariationalDropoutCell' or
        'VariationalZoneoutCell', default: 'ZoneoutCell')
    dropout_rate: float
        Dropout regularization parameter (default: 0.1)
    embedding_dimension: int
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    distr_output: gluonts.mx.DistributionOutput()
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput())
    scaling: bool
        Whether to automatically scale the target values (default: true)
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = DeepAREstimator


class MQCNNModel(AbstractGluonTSModel):
    """MQCNN model from Gluon-TS. MQCNN is an encoder-decoder model where the encoder is a
    1D convolutional neural network and the decoder is a multilayer perceptron.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    embedding_dimension: int
        Dimension of the embeddings for categorical features. (default: [min(50, (cat+1)//2) for cat in cardinality])
    add_time_feature: bool
        Adds a set of time features. (default: True)
    add_age_feature: bool
        Adds an age feature. (default: False)
        The age feature starts with a small value at the start of the time series and grows over time.
    seed: int
        Will set the specified int seed for numpy and MXNet if specified. (default: None)
    decoder_mlp_dim_seq: List[int]
        The dimensionalities of the Multi Layer Perceptron layers of the decoder.
        (default: [30])
    channels_seq: List[int]
        The number of channels (i.e. filters or convolutions) for each layer of the HierarchicalCausalConv1DEncoder.
        More channels usually correspond to better performance and larger network size.
        (default: [30, 30, 30])
    dilation_seq: List[int]
        The dilation of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        Greater numbers correspond to a greater receptive field of the network, which is usually
        better with longer context_length. (Same length as channels_seq) (default: [1, 3, 5])
    kernel_size_seq: List[int]
        The kernel sizes (i.e. window size) of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        (Same length as channels_seq) (default: [7, 3, 3])
    use_residual: bool
        Whether the hierarchical encoder should additionally pass the unaltered
        past target to the decoder. (default: True)
    quantiles: List[float]
        The list of quantiles that will be optimized for, and predicted by, the model.
        Optimizing for more quantiles than are of direct interest to you can result
        in improved performance due to a regularizing effect.
        (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    distr_output: gluonts.mx.DistributionOutput()
        DistributionOutput to use. Only one between `quantile` and `distr_output`
        can be set. (Default: None)
    scaling: bool
        Whether to automatically scale the target values. (default: False if quantile_output is used, True otherwise)
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = MQCNNEstimator

    def _get_estimator(self):
        if get_mxnet_context() != mx.context.cpu():
            self.params["hybridize"] = False

        with warning_filter():
            return self.gluonts_estimator_class.from_hyperparameters(
                **self._get_estimator_init_args()
            )


class SimpleFeedForwardModel(AbstractGluonTSModel):
    """SimpleFeedForward model, i.e. a simple multilayer perceptron for
     probabilistic forecasts, from GluonTS.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    num_hidden_dimensions: int
        Number of hidden nodes in each layer (default: [40, 40])
    context_length: int
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    distr_output: gluonts.mx.DistributionOutput
        Distribution to fit (default: StudentTOutput())
    batch_normalization: bool
        Whether to use batch normalization (default: False)
    mean_scaling: bool
        Scale the network input by the data mean and the network output by
        its inverse (default: True)
    """

    gluonts_estimator_class: Type[GluonTSEstimator] = SimpleFeedForwardEstimator


class GenericGluonTSModel(AbstractGluonTSModel):
    """Generic wrapper model class for GluonTS models (in GluonTS terminology---
    Estimators). While this class is meant to generally enable fast use of GluonTS
    models in autogluon, specific GluonTS models accessed through this wrapper may
    not have been tested and should be used at the user's own risk.

    Please refer to each GluonTS estimator's individual documentation for
    initialization parameters of each model.

    Parameters
    ----------
    gluonts_estimator_class:
        The class object of the GluonTS estimator to be used.
    """

    def __init__(self, gluonts_estimator_class: Type[GluonTSEstimator], **kwargs):
        self.gluonts_estimator_class = gluonts_estimator_class
        gluonts_model_name = re.sub(
            r"Estimator$", "", self.gluonts_estimator_class.__name__
        )

        super().__init__(name=kwargs.pop("name", gluonts_model_name), **kwargs)

    def get_params(self) -> dict:
        params_dict = super().get_params()
        params_dict["gluonts_estimator_class"] = self.gluonts_estimator_class
        return params_dict


class _ProphetDummyEstimator(DummyEstimator):
    def train(self, train_data, validation_data=None, **kwargs):
        return self.predictor


class ProphetModel(AbstractGluonTSModel):
    """Wrapper around `Prophet <https://github.com/facebook/prophet>`_, which wraps the
    library through GluonTS's wrapper.

    In order to use it you need to install the package::

        # you can either install Prophet directly
        pip install fbprophet

        # or install gluonts with the Prophet extras
        pip install gluonts[Prophet]

    Other Parameters
    ----------------
    hyperparameters
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
            prophet_params={
                k: v
                for k, v in model_init_params.items()
                if k in self.allowed_prophet_parameters
            },
        )


# TODO: AutoGluon Tabular will be removed from GluonTS to avoid circular dependencies
class AutoTabularModel(AbstractGluonTSModel):
    """Autotabular model from Gluon-TS, which in turn uses autogluon.tabular
    predictors for fitting a forecast model.

    See `AbstractGluonTSModel` for common parameters.

    Other Parameters
    ----------------
    lag_indices: List[int]
        List of indices of the lagged observations to use as features. If
        None, this will be set automatically based on the frequency.
    scaling: bool
        Function to be used to scale time series. This should take a pd.Series object
        as input, and return a scaled pd.Series and the scale (float). By default,
        this divides a series by the mean of its absolute value.
    disable_auto_regression: bool
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
