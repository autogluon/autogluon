# Could eventually remove this code: Is this needed in unit tests?

"""
Object definitions that are used for testing.
"""

from typing import Iterator, Tuple, Dict
import numpy as np

from ..datatypes.common import StateIdAndCandidate
from ..datatypes.hp_ranges import HyperparameterRanges_Impl, \
    HyperparameterRangeContinuous, HyperparameterRangeInteger, \
    HyperparameterRangeCategorical, HyperparameterRanges
from ..datatypes.scaling import LogScaling, LinearScaling
from ..datatypes.tuning_job_state import TuningJobState
from ..gpautograd.constants import MCMCConfig, OptimizationConfig
from ..gpautograd.gp_regression import GaussianProcessRegression
from ..gpautograd.gpr_mcmc import GPRegressionMCMC
from ..gpautograd.kernel import Matern52, KernelFunction
from ..gpautograd.warping import WarpedKernel, Warping
from ..tuning_algorithms.base_classes import CandidateGenerator
from ..tuning_algorithms.defaults import dictionarize_objective


def build_kernel(state: TuningJobState,
                 do_warping: bool = False) -> KernelFunction:
    dims, warping_ranges = dimensionality_and_warping_ranges(state.hp_ranges)
    kernel = Matern52(dims, ARD=True)
    if do_warping:
        return WarpedKernel(
            kernel=kernel, warping=Warping(dims, warping_ranges))
    else:
        return kernel


def default_gpmodel(
        state: TuningJobState, random_seed: int,
        optimization_config: OptimizationConfig) -> GaussianProcessRegression:
    return GaussianProcessRegression(
        kernel=build_kernel(state),
        optimization_config=optimization_config,
        random_seed=random_seed
    )


def default_gpmodel_mcmc(
        state: TuningJobState, random_seed: int,
        mcmc_config: MCMCConfig) -> GPRegressionMCMC:
    return GPRegressionMCMC(
        build_kernel=lambda: build_kernel(state),
        mcmc_config=mcmc_config,
        random_seed=random_seed
    )


def dimensionality_and_warping_ranges(hp_ranges: HyperparameterRanges) -> \
        Tuple[int, Dict[int, Tuple[float, float]]]:
    dims = 0
    warping_ranges = dict()
    # NOTE: This explicit loop over hp_ranges will fail if
    # HyperparameterRanges.hp_ranges is not implemented! Needs to be fixed if
    # it becomes an issue, either by moving the functionality here into
    # HyperparameterRanges, or by converting hp_ranges to
    # HyperparameterRanges_Impl, which supports the hp_ranges property.
    for hp_range in hp_ranges.hp_ranges:
        if not isinstance(hp_range, HyperparameterRangeCategorical):
            if isinstance(hp_range, HyperparameterRangeInteger):
                lower = int(round(hp_range.lower_bound))
                upper = int(round(hp_range.upper_bound))
            else:
                assert isinstance(hp_range, HyperparameterRangeContinuous)
                lower = float(hp_range.lower_bound)
                upper = float(hp_range.upper_bound)
            lower_internal = hp_range.to_ndarray(lower).item()
            upper_internal = hp_range.to_ndarray(upper).item()
            if upper_internal > lower_internal:  # exclude cases where max equal to min
                warping_ranges[dims] = (lower_internal, upper_internal)
            else:
                assert upper_internal == lower_internal
        dims += hp_range.ndarray_size()
    return dims, warping_ranges


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
