from typing import List, Optional
import numpy as np
import logging

from ..datatypes.tuning_job_state import TuningJobState
from ..tuning_algorithms.base_classes import SurrogateModel
from ..utils.debug_log import DebugLogPrinter

logger = logging.getLogger(__name__)


class BaseSurrogateModel(SurrogateModel):
    """
    Base class for (most) SurrogateModel implementations, provides common code

    """
    def __init__(
            self, state: TuningJobState, active_metric: str, random_seed: int,
            debug_log: Optional[DebugLogPrinter] = None):
        super().__init__(state, active_metric, random_seed)
        self._current_best = None
        self._debug_log = debug_log

    def _current_best_filter_candidates(self, candidates):
        """
        In some subclasses, 'current_best' is not computed over all (observed
        and pending) candidates: they need to implement this filter.

        """
        return candidates  # Default: No filtering

    def current_best(self) -> List[np.ndarray]:
        if self._current_best is None:
            candidates = [
                x.candidate for x in self.state.candidate_evaluations] + \
                         self.state.pending_candidates
            candidates = self._current_best_filter_candidates(candidates)
            assert len(candidates) > 0, \
                "Cannot determine incumbent (current_best) with no candidates at all"

            inputs = self.state.hp_ranges.to_ndarray_matrix(candidates)
            result = []
            for prediction in self.predict(inputs):
                means = prediction['mean']
                if means.ndim == 1:
                    means = means.reshape((-1, 1))
                result.append(np.min(means, axis=0))
            self._current_best = result
        return self._current_best
