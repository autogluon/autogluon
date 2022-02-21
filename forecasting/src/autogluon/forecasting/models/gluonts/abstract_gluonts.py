import copy
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.estimator import Estimator as GluonTSEstimator
from gluonts.model.forecast import SampleForecast, QuantileForecast, Forecast
from gluonts.model.predictor import Predictor as GluonTSPredictor

from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils import warning_filter

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

    def __init__(
        self,
        path: str,
        freq: str,
        prediction_length: int,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
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
        )
        self.gts_predictor: Optional[GluonTSPredictor] = None

        # TODO: handle gluonts constructor parameters
        self.params["callbacks"] = []
        self.params["freq"] = freq
        self.params["prediction_length"] = prediction_length

    def save(self, path: str = None) -> str:
        if path is None:
            path = self.path
        path = Path(path)
        path.mkdir(exist_ok=True)

        predictor = self.gts_predictor
        self.gts_predictor = None

        # TODO: filtering the serializing warning out until gluonts fixes it
        with serialize_warning_filter():
            if predictor:
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

    def _get_estimator(self) -> GluonTSEstimator:
        raise NotImplementedError

    def _fit(
        self,
        train_data: Dataset,
        val_data: Dataset = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        logger.log(30, f"Training forecasting model {self.name}...")
        self.params["callbacks"].append(TimeLimitCallback(time_limit))
        estimator = self._get_estimator()
        with warning_filter():
            self.gts_predictor = estimator.train(train_data, validation_data=val_data)

    def predict(
        self, data: Dataset, quantile_levels: List[float] = None
    ) -> Dict[str, pd.DataFrame]:
        logger.log(30, f"Predicting with forecasting model {self.name}")
        with warning_filter():
            quantiles = [str(q) for q in (quantile_levels or self.quantile_levels)]
            result_dict = {}
            predicted_targets = list(self.gts_predictor.predict(data))

            if isinstance(predicted_targets[0], QuantileForecast):
                status = [
                    0 < float(quantiles[i]) < 1
                    and str(quantiles[i]) in predicted_targets[0].forecast_keys
                    for i in range(len(quantiles))
                ]
            elif isinstance(predicted_targets[0], SampleForecast):
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
                predicted_targets = copy.deepcopy(transformed_targets)
                status = [
                    0 < float(quantiles[i]) < 1
                    and str(quantiles[i]) in predicted_targets[0].forecast_keys
                    for i in range(len(quantiles))
                ]
            else:
                raise TypeError("DistributionForecast is not yet supported.")

            if not all(status):  # TODO: CHECK LOGIC
                raise ValueError("Invalid quantile value.")

            index = [i["item_id"] for i in data]

            index_count = {}
            for idx in index:
                index_count[idx] = index_count.get(idx, 0) + 1

            for i in range(len(index)):
                tmp_dict = {}
                for quantile in quantiles:
                    tmp_dict[quantile] = predicted_targets[i].quantile(str(quantile))
                df = pd.DataFrame(tmp_dict)
                df.index = pd.date_range(
                    start=predicted_targets[i].start_date,
                    periods=self.params["prediction_length"],
                    freq=self.params["freq"],
                )
                if index_count[index[i]] > 1:
                    result_dict[f"{index[i]}_{predicted_targets[i].start_date}"] = df
                else:
                    result_dict[index[i]] = df
        return result_dict

    def _predict_for_scoring(
        self, data: Dataset, num_samples: int = 100
    ) -> Tuple[List[Forecast], List[Any]]:
        with warning_filter():
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=data, predictor=self.gts_predictor, num_samples=num_samples
            )
            return list(tqdm(forecast_it, total=len(data))), list(
                tqdm(ts_it, total=len(data))
            )

    def score(
        self, data: Dataset, metric: Optional[str] = None, num_samples: int = 100
    ):
        evaluator = (
            Evaluator(quantiles=self.quantile_levels)
            if self.quantile_levels is not None
            else Evaluator()
        )
        forecasts, tss = self._predict_for_scoring(data, num_samples=num_samples)
        num_series = len(tss)

        # TODO: filtering the warnings out until gluonts perfects it.
        with evaluator_warning_filter():
            agg_metrics, item_metrics = evaluator(
                iter(tss), iter(forecasts), num_series=num_series
            )
        return agg_metrics[self.eval_metric]

    def __repr__(self) -> str:
        return self.name
