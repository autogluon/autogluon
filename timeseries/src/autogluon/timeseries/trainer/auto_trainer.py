import logging
from typing import Any, Dict, List, Optional, Union

from ..models.presets import get_preset_models
from .abstract_trainer import AbstractTimeSeriesTrainer, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class AutoTimeSeriesTrainer(AbstractTimeSeriesTrainer):
    def construct_model_templates(self, hyperparameters, multi_window: bool = False, **kwargs):
        path = kwargs.pop("path", self.path)
        eval_metric = kwargs.pop("eval_metric", self.eval_metric)
        eval_metric_seasonal_period = kwargs.pop("eval_metric", self.eval_metric_seasonal_period)
        quantile_levels = kwargs.pop("quantile_levels", self.quantile_levels)
        hyperparameter_tune = kwargs.get("hyperparameter_tune", False)
        return get_preset_models(
            path=path,
            eval_metric=eval_metric,
            eval_metric_seasonal_period=eval_metric_seasonal_period,
            prediction_length=self.prediction_length,
            freq=kwargs.get("freq"),
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            quantile_levels=quantile_levels,
            all_assigned_names=self._get_banned_model_names(),
            target=self.target,
            metadata=self.metadata,
            excluded_model_types=kwargs.get("excluded_model_types"),
            # if skip_model_selection = True, we skip backtesting
            multi_window=multi_window and not self.skip_model_selection,
        )

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Union[str, Dict[Any, Dict]],
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        excluded_model_types: Optional[List[str]] = None,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
        **kwargs,
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
            the user may provide one of "default", "light" and "very_light" to specify presets.
        val_data: TimeSeriesDataFrame
            Optional validation data set to report validation scores on.
        hyperparameter_tune_kwargs
            Args for hyperparameter tuning
        excluded_model_types
            Names of models that should not be trained, even if listed in `hyperparameters`.
        time_limit
            Time limit for training
        random_seed
            Random seed that will be set to each model during training
        """
        self._train_multi(
            train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            time_limit=time_limit,
            random_seed=random_seed,
        )
