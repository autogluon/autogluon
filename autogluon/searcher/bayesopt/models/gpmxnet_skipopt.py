from abc import ABC, abstractmethod
import ConfigSpace as CS

from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.searcher.bayesopt.datatypes.common import Candidate


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
    points. Optimizations are not skipped if N < init_length, and
    afterwards optimizations are done every period times.

    NOTE: This predicate cannot be a state-less function, because
    __call__ may not be called for every value of N in sequence!

    """
    def __init__(self, init_length: int, period: int):
        assert init_length >= 0
        assert period > 1
        self.init_length = init_length
        self.period = period
        self.reset()

    def reset(self):
        self.current_bound = self.init_length
        self.lastrec_key = None

    def __call__(self, state: TuningJobState) -> bool:
        num_labeled = len(state.candidate_evaluations)
        assert self.lastrec_key is None or \
            num_labeled >= self.lastrec_key, \
            "num_labeled = {} < {} = lastrec_key".format(
                num_labeled, self.lastrec_key)
        # self.lastrec_XYZ is needed in order to allow __call__
        # to be called several times with the same num_labeled
        if num_labeled == self.lastrec_key:
            return self.lastrec_value
        if num_labeled < self.current_bound:
            ret_value = True
        else:
            # At this point, we passed current_bound, so should do the
            # optimization
            # Smallest init_length + k*period > num_labeled:
            self.current_bound = num_labeled + self.period -\
                ((num_labeled - self.init_length) % self.period)
            ret_value = False
        self.lastrec_key = num_labeled
        self.lastrec_value = ret_value
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
