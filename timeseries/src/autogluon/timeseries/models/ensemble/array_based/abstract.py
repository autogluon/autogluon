from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.utils.features import CovariateMetadata

from ..abstract import AbstractTimeSeriesEnsembleModel
from .regressor import EnsembleRegressor


class ArrayBasedTimeSeriesEnsembleModel(AbstractTimeSeriesEnsembleModel, ABC):
    """Abstract base class for time series ensemble models which operate on arrays of base model
    predictions for training and inference.

    Other Parameters
    ----------------
    isotonization: str, default = "sort"
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported.
    detect_and_ignore_failures: bool, default = True
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.
    """

    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        freq: str | None = None,
        prediction_length: int = 1,
        covariate_metadata: CovariateMetadata | None = None,
        target: str = "target",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        eval_metric: str | TimeSeriesScorer | None = None,
    ):
        super().__init__(
            path=path,
            name=name,
            hyperparameters=hyperparameters,
            freq=freq,
            prediction_length=prediction_length,
            covariate_metadata=covariate_metadata,
            target=target,
            quantile_levels=quantile_levels,
            eval_metric=eval_metric,
        )
        self.ensemble_regressor: EnsembleRegressor | None = None
        self._model_names: list[str] = []

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {
            "isotonization": "sort",
            "detect_and_ignore_failures": True,
        }

    @staticmethod
    def to_array(df: TimeSeriesDataFrame) -> np.ndarray:
        """Given a TimeSeriesDataFrame object, return a single array composing the values contained
        in the data frame.

        Parameters
        ----------
        df
            TimeSeriesDataFrame to convert to an array. Must contain exactly `prediction_length`
            values for each item. The columns of `df` can correspond to ground truth values
            or predictions (in which case, these will be the mean or quantile forecasts).

        Returns
        -------
        array
            of shape (num_items, prediction_length, num_outputs).
        """
        assert df.index.is_monotonic_increasing
        array = df.to_numpy()
        num_items = df.num_items
        shape = (
            num_items,
            df.shape[0] // num_items,  # timesteps per item
            df.shape[1],  # num_outputs
        )
        return array.reshape(shape)

    def _get_base_model_predictions(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]] | dict[str, TimeSeriesDataFrame],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Given a mapping from model names to a list of data frames representing
        their predictions per window, return a multidimensional array representation.

        Parameters
        ----------
        predictions_per_window
            A dictionary with list[TimeSeriesDataFrame] values, where each TimeSeriesDataFrame
            contains predictions for the window in question. If the dictionary values are
            TimeSeriesDataFrame, they will be treated like a single window.

        Returns
        -------
        base_model_mean_predictions
            Array of shape (num_windows, num_items, prediction_length, 1, num_models)
        base_model_quantile_predictions
            Array of shape (num_windows, num_items, prediction_length, num_quantiles, num_models)
        """

        if not predictions_per_window:
            raise ValueError("No base model predictions are provided.")

        first_prediction = list(predictions_per_window.values())[0]
        if isinstance(first_prediction, TimeSeriesDataFrame):
            predictions_per_window = {k: [v] for k, v in predictions_per_window.items()}  # type: ignore

        predictions = {
            model_name: [self.to_array(window) for window in windows]  # type: ignore
            for model_name, windows in predictions_per_window.items()
        }
        base_model_predictions = np.stack([x for x in predictions.values()], axis=-1)

        return base_model_predictions[:, :, :, :1, :], base_model_predictions[:, :, :, 1:, :]

    def _isotonize(self, prediction_array: np.ndarray) -> np.ndarray:
        """Apply isotonization to ensure quantile non-crossing.

        Parameters
        ----------
        prediction_array
            Array of shape (num_windows, num_items, prediction_length, num_quantiles)

        Returns
        -------
        isotonized_array
            Array with same shape but quantiles sorted along last dimension
        """
        isotonization = self.get_hyperparameter("isotonization")
        if isotonization == "sort":
            return np.sort(prediction_array, axis=-1)
        return prediction_array

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ) -> None:
        # process inputs
        filtered_predictions = self._filter_failed_models(predictions_per_window, model_scores)
        base_model_mean_predictions, base_model_quantile_predictions = self._get_base_model_predictions(
            filtered_predictions
        )

        # process labels
        ground_truth_per_window = [y.slice_by_timestep(-self.prediction_length, None) for y in data_per_window]
        labels = np.stack(
            [self.to_array(gt) for gt in ground_truth_per_window], axis=0
        )  # (num_windows, num_items, prediction_length, 1)

        self._model_names = list(filtered_predictions.keys())
        self.ensemble_regressor = self._get_ensemble_regressor()
        self.ensemble_regressor.fit(
            base_model_mean_predictions=base_model_mean_predictions,
            base_model_quantile_predictions=base_model_quantile_predictions,
            labels=labels,
            time_limit=time_limit,
        )

    @abstractmethod
    def _get_ensemble_regressor(self) -> EnsembleRegressor:
        pass

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if self.ensemble_regressor is None:
            if not self._model_names:
                raise ValueError("Ensemble model has not been fitted yet.")
            # Try to recreate the regressor (for loaded models)
            self.ensemble_regressor = self._get_ensemble_regressor()

        input_data = {}
        for m in self.model_names:
            assert m in data, f"Predictions for model {m} not provided during ensemble prediction."
            input_data[m] = data[m]

        base_model_mean_predictions, base_model_quantile_predictions = self._get_base_model_predictions(input_data)

        mean_predictions, quantile_predictions = self.ensemble_regressor.predict(
            base_model_mean_predictions=base_model_mean_predictions,
            base_model_quantile_predictions=base_model_quantile_predictions,
        )

        quantile_predictions = self._isotonize(quantile_predictions)
        prediction_array = np.concatenate([mean_predictions, quantile_predictions], axis=-1)

        output = list(input_data.values())[0].copy()
        num_folds, num_items, num_timesteps, num_outputs = prediction_array.shape
        assert (num_folds, num_timesteps) == (1, self.prediction_length)
        assert len(output.columns) == num_outputs

        output[output.columns] = prediction_array.reshape((num_items * num_timesteps, num_outputs))

        return output

    @property
    def model_names(self) -> list[str]:
        return self._model_names

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        """Update names of the base models based on the mapping in model_refit_map."""
        self._model_names = [model_refit_map.get(name, name) for name in self._model_names]

    def _filter_failed_models(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        model_scores: dict[str, float] | None,
    ) -> dict[str, list[TimeSeriesDataFrame]]:
        """Filter out failed models based on detect_and_ignore_failures setting."""
        if not self.get_hyperparameter("detect_and_ignore_failures"):
            return predictions_per_window

        if model_scores is None or len(model_scores) == 0:
            return predictions_per_window

        valid_scores = {k: v for k, v in model_scores.items() if np.isfinite(v)}
        if len(valid_scores) == 0:
            raise ValueError("All models have NaN scores. At least one model must run successfully to fit an ensemble")

        losses = {k: -v for k, v in valid_scores.items()}
        median_loss = np.nanmedian(list(losses.values()))
        threshold = 10 * median_loss
        good_models = {k for k, loss in losses.items() if loss <= threshold}

        return {k: v for k, v in predictions_per_window.items() if k in good_models}
