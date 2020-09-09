# Could eventually remove this code: Is this needed in unit tests?

"""
Object definitions that are used for testing.
"""

from typing import Iterator
import numpy as np

from ..datatypes.common import StateIdAndCandidate
from ..datatypes.hp_ranges import HyperparameterRanges_Impl, HyperparameterRangeContinuous, HyperparameterRangeInteger, HyperparameterRangeCategorical
from ..datatypes.scaling import LogScaling, LinearScaling
from ..tuning_algorithms.base_classes import CandidateGenerator
from ..tuning_algorithms.default_algorithm import dictionarize_objective


class RepeatedCandidateGenerator(CandidateGenerator):
    """Generates candidates from a fixed set. Used to test the deduplication logic."""
    def __init__(self, n_unique_candidates: int):
        self.all_unique_candidates = [
            (1.0*j, j, "value_" + str(j))
            for j in range(n_unique_candidates)
        ]

    def generate_candidates(self) -> Iterator[StateIdAndCandidate]:
        i = 0
        while True:
            i += 1
            yield self.all_unique_candidates[i % len(self.all_unique_candidates)]


# Example black box function, with adjustable location of global minimum.
# Potentially could catch issues with optimizer, e.g. if the optimizer
# ignoring somehow candidates on the edge of search space.
# A simple quadratic function is used.
class Quadratic3d:
    def __init__(self, local_minima, active_metric, metric_names):
        # local_minima: point where local_minima is located
        self.local_minima = np.array(local_minima).astype('float')
        self.local_minima[0] = np.log10(self.local_minima[0])
        self.active_metric = active_metric
        self.metric_names = metric_names

    @property
    def search_space(self):
        return HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('x', 1.0, 100.0, scaling=LogScaling()),
            HyperparameterRangeInteger('y', 0, 2, scaling=LinearScaling()),
            HyperparameterRangeCategorical('z', ('0.0', '1.0', '2.0'))
        )

    @property
    def f_min(self):
        return 0.0

    def __call__(self, candidate):
        p = np.array([float(hp) for hp in candidate])
        p[0] = np.log10(p[0])
        return dictionarize_objective(np.sum((self.local_minima - p) ** 2))
