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
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.torch.distributions import AffineTransformed
from gluonts.torch.distributions.distribution_output import NormalOutput
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.estimator import PyTorchLightningEstimator as GluonTSPyTorchLightningEstimator
from gluonts.torch.model.forecast import DistributionForecast, Forecast
from gluonts.torch.model.predictor import PyTorchPredictor as GluonTSPyTorchPredictor
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from pytorch_lightning.callbacks import Timer

from autogluon.core.hpo.constants import CUSTOM_BACKEND
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel, SimpleGluonTSDataset

# FIXME: introduces cpflows dependency. We exclude this model until a future release.
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

# FIXME: DeepNPTS does not implement the GluonTS PyTorch API, and does not use
# PyTorch Ligthning. We exclude this model until a future release.
# from gluonts.torch.model.deep_npts import DeepNPTSEstimator

logger = logging.getLogger(__name__)
gts_logger = logging.getLogger(gluonts.__name__)
pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if "pytorch_lightning" in name]


class AbstractGluonTSPyTorchModel(AbstractGluonTSModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator]

    def _get_hpo_backend(self):
        return CUSTOM_BACKEND

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to
        `self._get_model_params` for better readability."""
        init_kwargs = self._get_model_params()

        # GluonTS does not handle context_length=1 well, and sets the context to only prediction_length
        # we set it to a minimum of 10 here.
        init_kwargs["context_length"] = max(10, init_kwargs.get("context_length", self.prediction_length))

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

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "AbstractGluonTSModel":
        model = super().load(path, reset_paths, verbose)
        model.gts_predictor = GluonTSPyTorchPredictor.deserialize(Path(path) / cls.gluonts_model_path)
        return model


class DeepARModel(AbstractGluonTSPyTorchModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DeepAREstimator


class SimpleFeedForwardModel(AbstractGluonTSPyTorchModel):
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

    def _to_gluonts_dataset(self, time_series_df: Optional[TimeSeriesDataFrame]) -> Optional[GluonTSDataset]:
        return (
            SimpleGluonTSDataset(time_series_df, target_field_name=self.target, float_dtype=np.float32)
            if time_series_df is not None
            else None
        )

    def _gluonts_forecasts_to_data_frame(
        self, forecasts: List[Forecast], quantile_levels: List[float]
    ) -> TimeSeriesDataFrame:
        assert isinstance(forecasts[0], DistributionForecast)

        result_dfs = []
        for i, forecast in enumerate(forecasts):
            item_forecast_dict = dict(mean=forecast.mean)
            if isinstance(forecast.distribution, AffineTransformed):
                # FIXME: this is a hack to get around GluonTS not implementing quantiles for
                # torch AffineTransformed
                fdist = forecast.distribution
                quantiles_tensor = torch.tensor(quantile_levels, device=fdist.scale.device).unsqueeze(1)
                q_transformed = (fdist.scale * fdist.base_dist.icdf(quantiles_tensor) + fdist.loc).cpu().numpy().tolist()
                for ix, quantile in enumerate(quantile_levels):
                    item_forecast_dict[str(quantile)] = q_transformed[ix]
            else:
                for quantile in quantile_levels:
                    item_forecast_dict[str(quantile)] = forecast.quantile(str(quantile))

            df = pd.DataFrame(item_forecast_dict)
            df[ITEMID] = forecast.item_id
            df[TIMESTAMP] = pd.date_range(
                start=forecasts[i].start_date.to_timestamp(how="S"),
                periods=self.prediction_length,
                freq=self.freq,
            )
            result_dfs.append(df)

        return TimeSeriesDataFrame.from_data_frame(pd.concat(result_dfs))
