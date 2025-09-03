from abc import ABC
from typing import Any, Optional, Sequence, Type, Union

import numpy as np

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.utils.features import CovariateMetadata

from ..abstract import AbstractTimeSeriesEnsembleModel
from .regressor import EnsembleRegressor


class TensorBasedTimeSeriesEnsembleModel(AbstractTimeSeriesEnsembleModel, ABC):
    """Abstract base class for time series ensemble models which operate on tensors of base model predictions
    for training and inference.

    Other Parameters
    ----------------
    isotonization: str
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported. Default is "sort".
    ignore_models: List[str]
        A list of models to ignore. Default is None.
    detect_and_ignore_failures: bool
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.  Default is True.
    sparsify: bool
        Whether to (attempt to) sparsify the model after training it.
        Default is False.
        See also `prune_below`.
    prune_below: float
        Specifies at which level of importance the model should be sparsified.
        Default is 0.05.
    """

    _regressor_type: Type[EnsembleRegressor]

    def __init__(
        self,
        path: Optional[str] = None,
        name: Optional[str] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        covariate_metadata: Optional[CovariateMetadata] = None,
        target: str = "target",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
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
        self.ensemble_regressor: Optional[EnsembleRegressor] = None

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {
            "isotonization": True,
        }

    @staticmethod
    def to_tensor(df: TimeSeriesDataFrame) -> np.ndarray:
        """Given a TimeSeriesDataFrame object, or a list or dict of such objects, return
        a single tensor composing the values contained in the data frames.

        Parameters
        ----------
        tsdf
            TimeSeriesDataFrame to convert to a tensor.

        Returns
        -------
        tensor
            of shape (items, prediction_length, quantiles).
        """
        df = df.sort_index()
        tensor = df.values
        shape = (
            len(df.index.get_level_values(0).unique()),  # nr_items
            df.shape[0] // df.shape[1],  # implied prediction length
            df.shape[1],  # number of quantiles
        )
        return tensor.reshape(shape)

    def _split_data_per_window(
        self,
        data_per_window: list[TimeSeriesDataFrame],
    ):
        """Split the given `data_per_window` into ground truth for that window (fold) and the past data."""
        past_data_per_window = [y.slice_by_timestep(None, -self.prediction_length) for y in data_per_window]
        ground_truth_per_window = [y.slice_by_timestep(-self.prediction_length, None) for y in data_per_window]
        return ground_truth_per_window, past_data_per_window

    def _get_base_model_predictions_tensor(
        self,
        predictions_per_window: Union[dict[str, list[TimeSeriesDataFrame]], dict[str, TimeSeriesDataFrame]],
    ) -> np.ndarray:
        if not predictions_per_window:
            raise ValueError("No base model predictions are provided.")

        first_prediction = list(predictions_per_window.values())[0]
        if isinstance(first_prediction, TimeSeriesDataFrame):
            predictions_per_window = {k: [v] for k, v in predictions_per_window.items()}  # type: ignore

        predictions = {
            model_name: [self.to_tensor(window) for window in windows]  # type: ignore
            for model_name, windows in predictions_per_window.items()
        }
        return np.stack([x for x in predictions.values()], axis=-1)

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: Optional[dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ) -> None:
        ground_truth_per_window, _ = self._split_data_per_window(data_per_window=data_per_window)
        labels = np.concatenate(
            [self.to_tensor(gt) for gt in ground_truth_per_window], axis=0
        )  # (nr_windows, *ground_truth_dims)

        self.ensemble_regressor = self._regressor_type(**self.get_hyperparameters())
        self.ensemble_regressor.fit(
            base_model_predictions=self._get_base_model_predictions_tensor(predictions_per_window),
            labels=labels,
        )

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if self.ensemble_regressor is None:
            raise ValueError("Ensemble model has not been fitted yet.")

        prediction_tensor = self.ensemble_regressor.predict(
            self._get_base_model_predictions_tensor(data),
        )
        assert prediction_tensor.shape[0] == 1

        output = list(data.values())[0].copy()

        n_folds, n_items, n_timesteps, n_outputs = prediction_tensor.shape
        assert n_folds, n_timesteps == (1, self.prediction_length)
        assert len(output.columns) == n_outputs

        output[output.columns] = prediction_tensor.reshape((n_items * n_timesteps, -1))

        return output
