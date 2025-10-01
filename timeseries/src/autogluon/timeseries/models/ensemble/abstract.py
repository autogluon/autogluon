import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

from typing_extensions import final

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.models.abstract import TimeSeriesModelBase
from autogluon.timeseries.utils.features import CovariateMetadata

logger = logging.getLogger(__name__)


class AbstractTimeSeriesEnsembleModel(TimeSeriesModelBase, ABC):
    """Abstract class for time series ensemble models."""

    _default_model_name = None

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
        if name is None:
            name = self._default_model_name
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
        model_scores: Optional[dict[str, float]] = None,
        time_limit: Optional[float] = None,
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
        model_scores: Optional[dict[str, float]] = None,
        time_limit: Optional[float] = None,
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
