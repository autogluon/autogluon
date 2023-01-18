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
        predictions: Dict[str, TimeSeriesDataFrame],
        data: TimeSeriesDataFrame,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        """Fit ensemble model given predictions of candidate base models and the true data.

        Parameters
        ----------
        predictions : Dict[str, TimeSeriesDataFrame]
            Dictionary that maps the names of component models to their respective predictions as TimeSeriesDataFrames.
        data : TimeSeriesDataFrame
            Observed ground truth data used to train the ensemble. This includes both the forecast horizon (for which
            the predictions are given in ``predictions``), as well as the "history".
        time_limit : Optional[int]
            Maximum allowed time for training in seconds.
        """
        if time_limit is not None and time_limit <= 0:
            logger.warning(
                f"\tWarning: Model has no time left to train, skipping model... (Time Left = {round(time_limit, 1)}s)"
            )
            raise TimeLimitExceeded
        self._fit_ensemble(
            predictions=predictions,
            data=data,
            time_limit=time_limit,
        )
        return self

    def _fit_ensemble(
        self,
        predictions: Dict[str, TimeSeriesDataFrame],
        data: TimeSeriesDataFrame,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        """Private method for `fit_ensemble`. See `fit_ensemble` for documentation of arguments. Apart from the model
        training logic, `fit_ensemble` additionally implements other logic such as keeping track of the time limit.
        """
        raise NotImplementedError

    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        raise NotImplementedError
