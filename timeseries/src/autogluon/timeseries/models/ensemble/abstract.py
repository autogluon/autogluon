import logging
from abc import ABC, abstractmethod

from typing_extensions import final

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import TimeSeriesModelBase

logger = logging.getLogger(__name__)


class AbstractTimeSeriesEnsembleModel(TimeSeriesModelBase, ABC):
    """Abstract base class for time series ensemble models that combine predictions from multiple base models.

    Ensemble training process operates on validation predictions from base models rather than raw time series
    data. This allows the ensemble to learn optimal combination strategies based on each model's performance
    across different validation windows and time series patterns.
    """

    @property
    @abstractmethod
    def model_names(self) -> list[str]:
        """Names of base models included in the ensemble."""
        pass

    @final
    def fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ):
        """Fit ensemble model given predictions of candidate base models and the true data.

        Parameters
        ----------
        predictions_per_window
            Dictionary that maps the names of component models to their respective predictions for each validation
            window.
        data_per_window
            Observed ground truth data used to train the ensemble for each validation window. Each entry in the list
            includes both the forecast horizon (for which the predictions are given in ``predictions``), as well as the
            "history".
        model_scores
            Scores (higher is better) for the models that will constitute the ensemble.
        time_limit
            Maximum allowed time for training in seconds.
        """
        if time_limit is not None and time_limit <= 0:
            logger.warning(
                f"\tWarning: Model has no time left to train, skipping model... (Time Left = {round(time_limit, 1)}s)"
            )
            raise TimeLimitExceeded
        if isinstance(data_per_window, TimeSeriesDataFrame):
            raise ValueError("When fitting ensemble, `data` should contain ground truth for each validation window")
        num_val_windows = len(data_per_window)
        for model, preds in predictions_per_window.items():
            if len(preds) != num_val_windows:
                raise ValueError(f"For model {model} predictions are unavailable for some validation windows")
        self._fit(
            predictions_per_window=predictions_per_window,
            data_per_window=data_per_window,
            model_scores=model_scores,
            time_limit=time_limit,
        )
        return self

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ) -> None:
        """Private method for `fit`. See `fit` for documentation of arguments. Apart from the model
        training logic, `fit` additionally implements other logic such as keeping track of the time limit.
        """
        raise NotImplementedError

    @final
    def predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if not set(self.model_names).issubset(set(data.keys())):
            raise ValueError(
                f"Set of models given for prediction in {self.name} differ from those provided during initialization."
            )
        for model_name, model_pred in data.items():
            if model_pred is None:
                raise RuntimeError(f"{self.name} cannot predict because base model {model_name} failed.")

        # Make sure that all predictions have same shape
        assert len(set(pred.shape for pred in data.values())) == 1

        return self._predict(data=data, **kwargs)

    @abstractmethod
    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        pass

    @abstractmethod
    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        """Update names of the base models based on the mapping in model_refit_map.

        This method should be called after performing refit_full to point to the refitted base models, if necessary.
        """
        pass
