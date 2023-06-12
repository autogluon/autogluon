"""
Module including wrappers for PyTorch implementations of models in GluonTS
"""
import logging
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import gluonts
import numpy as np
import torch
from gluonts.core.component import from_hyperparameters
from gluonts.model.forecast import QuantileForecast
from gluonts.torch.model.d_linear import DLinearEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.estimator import PyTorchLightningEstimator as GluonTSPyTorchLightningEstimator
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.predictor import PyTorchPredictor as GluonTSPyTorchPredictor
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from pytorch_lightning.callbacks import Timer

from autogluon.core.hpo.constants import CUSTOM_BACKEND
from autogluon.core.utils.loaders import load_pkl
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.utils.warning_filters import torch_warning_filter

# FIXME: introduces cpflows dependency. We exclude this model until a future release.
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

# FIXME: DeepNPTS does not implement the GluonTS PyTorch API, and does not use
# PyTorch Lightning. We exclude this model until a future release.
# from gluonts.torch.model.deep_npts import DeepNPTSEstimator


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
        init_kwargs = super()._get_estimator_init_args()
        # Map MXNet kwarg names to PyTorch Lightning kwarg names
        init_kwargs.setdefault("lr", init_kwargs.get("learning_rate", 1e-3))
        init_kwargs.setdefault("max_epochs", init_kwargs.get("epochs"))
        return init_kwargs

    def _get_estimator(self) -> GluonTSPyTorchLightningEstimator:
        """Return the GluonTS Estimator object for the model"""

        # As GluonTSPyTorchLightningEstimator objects do not implement `from_hyperparameters` convenience
        # constructors, we re-implement the logic here.
        # we translate the "epochs" parameter to "max_epochs" for consistency in the AbstractGluonTSModel
        # interface

        init_args = self._get_estimator_init_args()

        trainer_kwargs = {}
        epochs = init_args.get("max_epochs")
        callbacks = init_args.get("callbacks", [])

        # TODO: Provide trainer_kwargs outside the function (e.g., to specify # of GPUs)?
        if epochs is not None:
            trainer_kwargs.update({"max_epochs": epochs})
        trainer_kwargs.update({"callbacks": callbacks, "enable_progress_bar": False})
        trainer_kwargs["default_root_dir"] = self.path

        if torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1

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
        lightning_logs_dir = Path(self.path) / "lightning_logs"
        if lightning_logs_dir.exists() and lightning_logs_dir.is_dir():
            logger.debug(f"Removing lightning_logs directory {lightning_logs_dir}")
            shutil.rmtree(lightning_logs_dir)

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

    @staticmethod
    def _distribution_to_quantile_forecast(
        forecast: DistributionForecast, quantile_levels: List[float]
    ) -> QuantileForecast:
        # Compute all quantiles in parallel instead of a for-loop
        quantiles = torch.tensor(quantile_levels, device=forecast.distribution.mean.device).reshape(-1, 1)
        quantile_predictions = forecast.distribution.icdf(quantiles).cpu().detach().numpy()
        forecast_arrays = np.vstack([forecast.mean, quantile_predictions])
        forecast_keys = ["mean"] + [str(q) for q in quantile_levels]

        forecast_init_args = dict(
            forecast_arrays=forecast_arrays,
            start_date=forecast.start_date,
            forecast_keys=forecast_keys,
            item_id=str(forecast.item_id),
        )
        return QuantileForecast(**forecast_init_args)


class DeepARModel(AbstractGluonTSPyTorchModel):
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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DeepAREstimator
    default_num_samples: int = 250
    supports_known_covariates = True

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs["num_feat_static_cat"] = self.num_feat_static_cat
        init_kwargs["num_feat_static_real"] = self.num_feat_static_real
        init_kwargs["cardinality"] = self.feat_static_cat_cardinality
        init_kwargs["num_feat_dynamic_real"] = self.num_feat_dynamic_real
        return init_kwargs


class SimpleFeedForwardModel(AbstractGluonTSPyTorchModel):
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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = SimpleFeedForwardEstimator


class TemporalFusionTransformerModel(AbstractGluonTSPyTorchModel):
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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = TemporalFusionTransformerEstimator
    supports_known_covariates = True
    supports_past_covariates = True

    @property
    def default_context_length(self) -> int:
        return max(64, 2 * self.prediction_length)

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
        return init_kwargs


class DLinearModel(AbstractGluonTSPyTorchModel):
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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    weight_decay : float, default = 1e-8
        Weight decay regularization parameter.
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DLinearEstimator

    @property
    def default_context_length(self) -> int:
        return 96


class PatchTSTModel(AbstractGluonTSPyTorchModel):
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
    epochs : int, default = 100
        Number of epochs the model will be trained for
    batch_size : int, default = 64
        Size of batches used during training
    num_batches_per_epoch : int, default = 50
        Number of batches processed every epoch
    learning_rate : float, default = 1e-3,
        Learning rate used during training
    weight_decay : float, default = 1e-8
        Weight decay regularization parameter.
    """

    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = PatchTSTEstimator

    @property
    def default_context_length(self) -> int:
        return 96

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        init_kwargs = super()._get_estimator_init_args()
        init_kwargs.setdefault("patch_len", 16)
        return init_kwargs
