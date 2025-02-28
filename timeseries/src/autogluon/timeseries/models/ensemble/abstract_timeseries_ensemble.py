import logging
from typing import Dict, List, Optional

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class AbstractTimeSeriesEnsembleModel(AbstractTimeSeriesModel):
    """Abstract class for time series ensemble models."""

    @property
    def model_names(self) -> List[str]:
        """Names of base models included in the ensemble."""
        raise NotImplementedError

    def fit_ensemble(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        time_limit: Optional[float] = None,
        **kwargs,
    ):
        """Fit ensemble model given predictions of candidate base models and the true data.

        Parameters
        ----------
        predictions_per_window : Dict[str, List[TimeSeriesDataFrame]]
            Dictionary that maps the names of component models to their respective predictions for each validation
            window.
        data_per_window : List[TimeSeriesDataFrame]
            Observed ground truth data used to train the ensemble for each validation window. Each entry in the list
            includes both the forecast horizon (for which the predictions are given in ``predictions``), as well as the
            "history".
        time_limit : Optional[int]
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
        self._fit_ensemble(
            predictions_per_window=predictions_per_window,
            data_per_window=data_per_window,
            time_limit=time_limit,
        )
        return self

    def _fit_ensemble(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        """Private method for `fit_ensemble`. See `fit_ensemble` for documentation of arguments. Apart from the model
        training logic, `fit_ensemble` additionally implements other logic such as keeping track of the time limit.
        """
        raise NotImplementedError

    def predict(self, data: Dict[str, Optional[TimeSeriesDataFrame]], **kwargs) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def remap_base_models(self, model_refit_map: Dict[str, str]) -> None:
        """Update names of the base models based on the mapping in model_refit_map.

        This method should be called after performing refit_full to point to the refitted base models, if necessary.
        """
        raise NotImplementedError

    # TODO: remove
    def _fit(*args, **kwargs):
        pass

    # TODO: remove
    def _predict(*args, **kwargs):
        pass
