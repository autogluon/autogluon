import copy
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Type

import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.estimator import Estimator as GluonTSEstimator
from gluonts.model.forecast import SampleForecast, QuantileForecast, Forecast
from gluonts.model.predictor import Predictor as GluonTSPredictor

import autogluon.core as ag
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils import warning_filter
from ...utils.metric_utils import METRIC_COEFFICIENTS

from ...utils.warning_filters import evaluator_warning_filter, serialize_warning_filter
from ..abstract import AbstractForecastingModel
from .callback import TimeLimitCallback

logger = logging.getLogger(__name__)


class AbstractGluonTSModel(AbstractForecastingModel):
    """Abstract class wrapping GluonTS estimators for use in autogluon.forecasting.

    Parameters
    ----------
    path: str
        directory to store model artifacts.
    freq: str
        string representation (compatible with GluonTS frequency strings) for the data provided.
        For example, "1D" for daily data, "1H" for hourly data, etc.
    prediction_length: int
        Number of time steps ahead (length of the forecast horizon) the model will be optimized
        to predict. At inference time, this will be the number of time steps the model will
        predict.
    name: str
        Name of the model. Also, name of subdirectory inside path where model will be saved.
    eval_metric: str
        objective function the model intends to optimize, will use mean_wQuantileLoss by default.
    hyperparameters:
        various hyperparameters that will be used by model (can be search spaces instead of
        fixed values). See *Other Parameters* in each inheriting model's documentation for
        possible values.
    """

    gluonts_model_path = "gluon_ts"
    gluonts_estimator_class: Type[GluonTSEstimator] = None

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        name = name or re.sub(
            r"Model$", "", self.__class__.__name__
        )  # TODO: look name up from presets
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.gts_predictor: Optional[GluonTSPredictor] = None
        self.callbacks = []

    def save(self, path: str = None, **kwargs) -> str:
        if path is None:
            path = self.path
        path = Path(path)
        path.mkdir(exist_ok=True)

        predictor = self.gts_predictor
        self.gts_predictor = None

        with serialize_warning_filter():
            if predictor:
                Path.mkdir(path / self.gluonts_model_path, exist_ok=True)
                predictor.serialize(path / self.gluonts_model_path)

        save_pkl.save(path=str(path / self.model_file_name), object=self)
        self.gts_predictor = predictor

        return str(path)

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, verbose: bool = True
    ) -> "AbstractGluonTSModel":
        model = super().load(path, reset_paths, verbose)
        model.gts_predictor = GluonTSPredictor.deserialize(
            Path(path) / cls.gluonts_model_path
        )
        return model

    def _deferred_init_params_aux(self, **kwargs) -> None:
        """Update GluonTS specific parameters with information available
        only at training time.
        """
        if "dataset" in kwargs:
            ds = kwargs.get("dataset")
            top_item = next(iter(ds))
            self.freq = top_item.get("freq", kwargs.get("freq") or self.freq)
            if not self.freq:
                raise ValueError(
                    "Dataset frequency not provided in the dataset, fit arguments or "
                    "during initialization. Please provide a `freq` string to `fit`."
                )

        if "callback" in kwargs:
            self.callbacks.append(kwargs["callback"])

    def _get_model_params(self) -> dict:
        """Gets params that are passed to the inner model."""
        args = super()._get_model_params()
        args.update(
            dict(
                freq=self.freq,
                prediction_length=self.prediction_length,
                quantiles=self.quantile_levels,
                callbacks=self.callbacks,
            )
        )

        return args

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to
        `self._get_model_params` for better readability."""
        return self._get_model_params()

    def _get_estimator(self) -> GluonTSEstimator:
        """Return the GluonTS Estimator object for the model"""
        with warning_filter():
            return self.gluonts_estimator_class.from_hyperparameters(
                **self._get_estimator_init_args()
            )

    def _fit(
        self,
        train_data: Dataset,
        val_data: Dataset = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        logger.log(30, f"Training forecasting model {self.name}...")

        # gracefully handle hyperparameter specifications if they are provided to fit instead
        if any(isinstance(v, ag.Space) for v in self.params.values()):
            raise ValueError(
                "Hyperparameter spaces provided to `fit`. Please provide concrete values "
                "as hyperparameters when initializing or use `hyperparameter_tune` instead."
            )

        # update auxiliary parameters
        self._deferred_init_params_aux(
            dataset=train_data, callback=TimeLimitCallback(time_limit), **kwargs
        )

        estimator = self._get_estimator()
        with warning_filter():
            self.gts_predictor = estimator.train(train_data, validation_data=val_data)

    # TODO: predict should also accept number of samples
    def predict(
        self, data: Dataset, quantile_levels: List[float] = None, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        logger.log(30, f"Predicting with forecasting model {self.name}")
        with warning_filter():
            quantiles = [str(q) for q in (quantile_levels or self.quantile_levels)]
            result_dict = {}
            predicted_targets = list(self.gts_predictor.predict(data))

            if not all(0 < float(q) < 1 for q in quantiles):
                raise ValueError(
                    "Invalid quantile value specified. Quantiles must be between 0 and 1 (exclusive)."
                )

            if not isinstance(predicted_targets[0], (QuantileForecast, SampleForecast)):
                raise TypeError("DistributionForecast is not yet supported.")

            # if predictions are gluonts SampleForecasts, convert them to quantile forecasts
            # but save the means
            forecast_means = []

            if isinstance(predicted_targets[0], SampleForecast):
                transformed_targets = []
                for forecast in predicted_targets:
                    tmp = []
                    for quantile in quantiles:
                        tmp.append(forecast.quantile(quantile))
                    transformed_targets.append(
                        QuantileForecast(
                            forecast_arrays=np.array(tmp),
                            start_date=forecast.start_date,
                            freq=forecast.freq,
                            forecast_keys=quantiles,
                            item_id=forecast.item_id,
                        )
                    )

                    forecast_means.append(forecast.mean)

                predicted_targets = copy.deepcopy(transformed_targets)

            # sanity check to ensure all quantiles are accounted for
            assert all(q in predicted_targets[0].forecast_keys for q in quantiles), (
                "Some forecast quantiles are missing from GluonTS forecast outputs. Was"
                " the model trained to forecast all quantiles?"
            )

            # get index of item_ids in the data set and check how many times each occurs
            index = [i["item_id"] for i in data]
            index_count = {}
            for idx in index:
                index_count[idx] = index_count.get(idx, 0) + 1

            for i, item_id in enumerate(index):
                item_forecast_dict = dict(
                    mean=forecast_means[i]
                    if forecast_means
                    else (
                        predicted_targets[i].quantile(
                            0.5
                        )  # assign P50 to mean if mean is missing
                    )
                )
                for quantile in quantiles:
                    item_forecast_dict[quantile] = predicted_targets[i].quantile(
                        str(quantile)
                    )

                # TODO: can be optimized: avoid redundant data frame constructions
                df = pd.DataFrame(item_forecast_dict)
                df.index = pd.date_range(
                    start=predicted_targets[i].start_date,
                    periods=self.prediction_length,
                    freq=self.freq,
                )

                # if item ids are redundant, index forecasts by itemid-forecast start date
                if index_count[item_id] > 1:
                    result_dict[f"{item_id}_{predicted_targets[i].start_date}"] = df
                else:
                    result_dict[item_id] = df

        return result_dict

    def _predict_for_scoring(
        self, data: Dataset, num_samples: int = 100
    ) -> Tuple[List[Forecast], List[Any]]:
        """Generate forecasts for the trailing `prediction_length` time steps of the
        data set, and return two iterators, one for the predictions and one for the
        ground truth time series.

        Differently from the `predict` function, this function returns predictions in
        GluonTS format for easier evaluation, and does not necessarily compute quantiles.
        """
        with warning_filter():
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=data, predictor=self.gts_predictor, num_samples=num_samples
            )
            return list(forecast_it), list(ts_it)

    def score(
        self, data: Dataset, metric: Optional[str] = None, num_samples: int = 100
    ):
        """Return the evaluation scores for given metric and dataset. The last
        `self.prediction_length` time steps of each time series in the input data set
        will be held out and used for computing the evaluation score.

        Parameters
        ----------
        data: gluonts.dataset.common.Dataset
            Dataset used for scoring.
        metric: str
            String identifier of evaluation metric to use, from one of
            `autogluon.forecasting.utils.metric_utils.AVAILABLE_METRICS`.
        num_samples: int
            Number of samples to use for making evaluation predictions if the probabilistic
            forecasts are generated by forward sampling from the fitted model. Otherwise, this
            parameter will be ignored.

        Returns
        -------
        score: float
            The computed forecast evaluation score on the last `self.prediction_length`
            time steps of each time series.
        """
        evaluator = (
            Evaluator(quantiles=self.quantile_levels)
            if self.quantile_levels is not None
            else Evaluator()
        )
        forecasts, tss = self._predict_for_scoring(data, num_samples=num_samples)
        num_series = len(tss)

        with evaluator_warning_filter():
            agg_metrics, item_metrics = evaluator(
                iter(tss), iter(forecasts), num_series=num_series
            )
        return agg_metrics[self.eval_metric] * METRIC_COEFFICIENTS[self.eval_metric]

    def __repr__(self) -> str:
        return self.name
