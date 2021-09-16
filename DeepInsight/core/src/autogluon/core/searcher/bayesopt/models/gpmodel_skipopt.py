from abc import ABC, abstractmethod
import ConfigSpace as CS

from ..datatypes.common import Candidate
from ..datatypes.tuning_job_state import TuningJobState


class SkipOptimizationPredicate(ABC):
    """
    Interface for skip_optimization predicate in GPMXNetModel

    """
    def reset(self):
        """
        If there is an internal state, reset it to its initial value
        """
        pass

    @abstractmethod
    def __call__(self, state: TuningJobState) -> bool:
        """
        :param state: Current TuningJobState
        :return: Skip hyperparameter optimization in GPMXNetModel.update?
        """
        pass


class NeverSkipPredicate(SkipOptimizationPredicate):
    """
    Hyperparameter optimization is never skipped.

    """
    def __call__(self, state: TuningJobState) -> bool:
        return False


class AlwaysSkipPredicate(SkipOptimizationPredicate):
    """
    Hyperparameter optimization is always skipped.

    """
    def __call__(self, state: TuningJobState) -> bool:
        return True


class SkipPeriodicallyPredicate(SkipOptimizationPredicate):
    """
    Let N = len(state.candidate_evaluations) be the number of labeled
    points. Optimizations are not skipped if N < init_length. Afterwards,
    we increase a counter whenever N is larger than in the previous
    call. With respect to this counter, optimizations are done every
    period times, in between they are skipped.

    """
    def __init__(self, init_length: int, period: int):
        assert init_length >= 0
        assert period > 1
        self.init_length = init_length
        self.period = period
        self.reset()

    def reset(self):
        self._counter = 0
        # Need to make sure that if called several times with the same state,
        # we return the same value
        self._last_size = None
        self._last_retval = None

    def __call__(self, state: TuningJobState) -> bool:
        num_labeled = len(state.candidate_evaluations)
        if num_labeled == self._last_size:
            return self._last_retval
        if self._last_size is not None:
            assert num_labeled > self._last_size, \
                "num_labeled = {} < {} = _last_size".format(
                    num_labeled, self._last_size)
        if num_labeled < self.init_length:
            ret_value = False
        else:
            ret_value = (self._counter % self.period != 0)
            self._counter += 1
        self._last_size = num_labeled
        self._last_size = ret_value
        return ret_value


class SkipNoMaxResourcePredicate(SkipOptimizationPredicate):
    """
    This predicate works for multi-fidelity HPO, see for example
    GPMultiFidelitySearcher.

    We track the number of labeled datapoints at resource level max_resource.
    HP optimization is skipped if the total number of labeled cases is >=
    init_length, and if the number of max_resource cases has not increased
    since the last recent optimization.

    This means that as long as the dataset only grows w.r.t. cases at lower
    resources than max_resource, this does not trigger HP optimization.

    """
    def __init__(self, init_length: int, resource_attr_name: str,
                 max_resource: int):
        assert init_length >= 0
        self.init_length = init_length
        self.resource_attr_name = resource_attr_name
        self.max_resource = max_resource
        self.reset()

    def reset(self):
        self.lastrec_max_resource_cases = None

    def _num_max_resource_cases(self, state: TuningJobState):
        def is_max_resource(config: Candidate) -> int:
            if isinstance(config, CS.Configuration) and \
                    (config.get_dictionary()[self.resource_attr_name] ==
                     self.max_resource):
                return 1
            else:
                return 0

        return sum(is_max_resource(x.candidate)
                   for x in state.candidate_evaluations)

    def __call__(self, state: TuningJobState) -> bool:
        if len(state.candidate_evaluations) < self.init_length:
            return False
        num_max_resource_cases = self._num_max_resource_cases(state)
        if self.lastrec_max_resource_cases is None or \
                num_max_resource_cases > self.lastrec_max_resource_cases:
            self.lastrec_max_resource_cases = num_max_resource_cases
            return False
        else:
            return True
