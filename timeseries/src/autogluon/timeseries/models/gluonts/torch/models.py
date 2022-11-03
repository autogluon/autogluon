"""
Module including wrappers for PyTorch implementations of models in GluonTS
"""
import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import gluonts
import numpy as np
import pandas as pd
import torch
from gluonts.core.component import from_hyperparameters
from gluonts.torch.distributions import AffineTransformed, NormalOutput
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.estimator import PyTorchLightningEstimator as GluonTSPyTorchLightningEstimator
from gluonts.torch.model.forecast import DistributionForecast, Forecast
from gluonts.torch.model.predictor import PyTorchPredictor as GluonTSPyTorchPredictor
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from pytorch_lightning.callbacks import Timer

from autogluon.core.hpo.constants import CUSTOM_BACKEND
from autogluon.core.utils.loaders import load_pkl
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.utils.warning_filters import torch_warning_filter

# FIXME: introduces cpflows dependency. We exclude this model until a future release.
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

# FIXME: DeepNPTS does not implement the GluonTS PyTorch API, and does not use
# PyTorch Lightning. We exclude this model until a future release.
# from gluonts.torch.model.deep_npts import DeepNPTSEstimator

# TODO: add docstrings for models

logger = logging.getLogger(__name__)
gts_logger = logging.getLogger(gluonts.__name__)
pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if "pytorch_lightning" in name]


class AbstractGluonTSPyTorchModel(AbstractGluonTSModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator]
    float_dtype: Type = np.float32

    def _get_hpo_backend(self):
        return CUSTOM_BACKEND

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to
        `self._get_model_params` for better readability."""
        init_kwargs = self._get_model_params()

        # GluonTS does not handle context_length=1 well, and sets the context to only prediction_length
        # we set it to a minimum of 10 here.
        init_kwargs["context_length"] = max(10, init_kwargs.get("context_length", self.prediction_length))
        init_kwargs.setdefault("lr", init_kwargs.get("learning_rate", 1e-3))

        return init_kwargs

    def _get_estimator(self) -> GluonTSPyTorchLightningEstimator:
        """Return the GluonTS Estimator object for the model"""

        # As GluonTSPyTorchLightningEstimator objects do not implement `from_hyperparameters` convenience
        # constructors, we re-implement the logic here.
        # we translate the "epochs" parameter to "max_epochs" for consistency in the AbstractGluonTSModel
        # interface

        init_args = self._get_estimator_init_args()

        trainer_kwargs = {}
        epochs = init_args.get("max_epochs", init_args.get("epochs"))
        callbacks = init_args.get("callbacks", [])

        if epochs is not None:
            trainer_kwargs.update({"max_epochs": epochs})
        trainer_kwargs.update({"callbacks": callbacks, "enable_progress_bar": False})

        return from_hyperparameters(
            self.gluonts_estimator_class,
            trainer_kwargs=trainer_kwargs,
            **init_args,
        )

    def _get_callbacks(self, time_limit: int, *args, **kwargs) -> List[Callable]:
        return [Timer(timedelta(seconds=time_limit))] if time_limit is not None else []

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        verbosity = kwargs.get("verbosity", 2)
        for pl_logger in pl_loggers:
            pl_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)
        super()._fit(train_data=train_data, val_data=val_data, time_limit=time_limit, **kwargs)

    def save(self, path: str = None, **kwargs) -> str:
        # we flush callbacks instance variable if it has been set. it can keep weak references
        # which breaks training
        self.callbacks = []
        return super().save(path, **kwargs)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "AbstractGluonTSModel":
        with torch_warning_filter():
            model = load_pkl.load(path=path + cls.model_file_name, verbose=verbose)
            if reset_paths:
                model.set_contexts(path)
            model.gts_predictor = GluonTSPyTorchPredictor.deserialize(Path(path) / cls.gluonts_model_path)
        return model


class DeepARModel(AbstractGluonTSPyTorchModel):
    """DeepAR model from GluonTS based on the PyTorch backend.

    The model consists of an RNN encoder (LSTM or GRU) and a decoder that outputs the
    distribution of the next target value. Close to the model described in [Salinas2020]_.

    .. [Salinas2020] Salinas, David, et al.
        "DeepAR: Probabilistic forecasting with autoregressive recurrent networks."
        International Journal of Forecasting. 2020.

    Based on `gluonts.torch.model.deepar.DeepAREstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html>`_.


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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 32
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DeepAREstimator
    default_num_samples: int = 250

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs["num_feat_static_cat"] = self.num_feat_static_cat
        init_kwargs["num_feat_static_real"] = self.num_feat_static_real
        init_kwargs["cardinality"] = self.feat_static_cat_cardinality
        init_kwargs["num_feat_dynamic_real"] = self.num_feat_dynamic_real
        return init_kwargs


class SimpleFeedForwardModel(AbstractGluonTSPyTorchModel):
    """SimpleFeedForward model from GluonTS based on the PyTorch backend.

    The model consists of a multilayer perceptron (MLP) that predicts the distribution of all the target value in the
    forecast horizon.

    Based on `gluonts.torch.model.simple_feedforward.SimpleFeedForwardEstimator <https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.simple_feedforward.html>`_.
    See GluonTS documentation for additional hyperparameters.


    Other Parameters
    ----------------
    context_length : int, optional
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    hidden_dimensions: List[int], default = [20, 20]
        Size of hidden layers in the feedforward network
    distr_output : gluonts.torch.distributions.DistributionOutput, default = NormalOutput()
        Distribution to fit.
    batch_normalization : bool, default = False
        Whether to use batch normalization
    mean_scaling : bool, default = True
        Scale the network input by the data mean and the network output by its inverse
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 32
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = SimpleFeedForwardEstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()

        # FIXME: PyTorch StudentT does not implement quantile functions
        if "distr_output" in init_kwargs:
            warnings.warn(
                f"distr_output {init_kwargs['distr_output']} specified for SimpleFeedForward, however training"
                "will default to the Gaussian distribution."
            )
        init_kwargs["distr_output"] = NormalOutput()
        return init_kwargs

    def _gluonts_forecasts_to_data_frame(
        self, forecasts: List[Forecast], quantile_levels: List[float]
    ) -> TimeSeriesDataFrame:
        assert isinstance(forecasts[0], DistributionForecast)

        result_dfs = []
        for i, forecast in enumerate(forecasts):
            item_forecast_dict = dict(mean=forecast.mean)
            if isinstance(forecast.distribution, AffineTransformed):
                # FIXME: this is a hack to get around GluonTS not implementing quantiles for
                # torch AffineTransformed. We hence force PyTorch SFF to always use Gaussian error.
                # However, this leads to a ~2x regression in error compared to MXNet SFF.
                fdist = forecast.distribution
                quantiles_tensor = torch.tensor(quantile_levels, device=fdist.scale.device).unsqueeze(1)
                q_transformed = (
                    (fdist.scale * fdist.base_dist.icdf(quantiles_tensor) + fdist.loc).cpu().numpy().tolist()
                )
                for ix, quantile in enumerate(quantile_levels):
                    item_forecast_dict[str(quantile)] = q_transformed[ix]
            else:
                for quantile in quantile_levels:
                    item_forecast_dict[str(quantile)] = forecast.quantile(str(quantile))

            df = pd.DataFrame(item_forecast_dict)
            df[ITEMID] = forecast.item_id
            # TODO: replace with get_forecast_horizon_index_single_time_series
            df[TIMESTAMP] = pd.date_range(
                start=forecasts[i].start_date.to_timestamp(how="S"),
                periods=self.prediction_length,
                freq=self.freq,
            )
            result_dfs.append(df)

        return TimeSeriesDataFrame.from_data_frame(pd.concat(result_dfs))
