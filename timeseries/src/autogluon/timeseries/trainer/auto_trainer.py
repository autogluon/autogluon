import logging
from typing import Any, Dict, Optional, Union

from ..models.presets import get_preset_models
from .abstract_trainer import AbstractTimeSeriesTrainer, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class AutoTimeSeriesTrainer(AbstractTimeSeriesTrainer):
    def construct_model_templates(self, hyperparameters, **kwargs):
        path = kwargs.pop("path", self.path)
        eval_metric = kwargs.pop("eval_metric", self.eval_metric)
        quantile_levels = kwargs.pop("quantile_levels", self.quantile_levels)
        hyperparameter_tune = kwargs.get("hyperparameter_tune", False)
        return get_preset_models(
            path=path,
            eval_metric=eval_metric,
            prediction_length=self.prediction_length,
            freq=kwargs.get("freq"),
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            quantiles=quantile_levels,
            invalid_model_names=self._get_banned_model_names(),
            target=self.target,
            metadata=self.metadata,
        )

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Union[str, Dict[Any, Dict]],
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        time_limit: float = None,
    ):
        """
        Fit a set of timeseries models specified by the `hyperparameters`
        dictionary that maps model names to their specified hyperparameters.

        Parameters
        ----------
        train_data: TimeSeriesDataFrame
            Training data for fitting time series timeseries models.
        hyperparameters: str or Dict
            A dictionary mapping selected model names, model classes or model factory to hyperparameter
            settings. Model names should be present in `trainer.presets.DEFAULT_MODEL_NAMES`. Optionally,
            the user may provide one of "local_only", "default", "default_hpo" to specify presets.
        val_data: TimeSeriesDataFrame
            Optional validation data set to report validation scores on.
        hyperparameter_tune_kwargs
            Args for hyperparameter tuning
        time_limit
            Time limit for training
        """
        self._train_multi(
            train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            time_limit=time_limit,
        )
