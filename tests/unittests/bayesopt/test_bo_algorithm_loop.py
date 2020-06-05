from typing import List
import numpy as np
import pytest
import mxnet as mx
import random

from autogluon.searcher.bayesopt.datatypes.common import CandidateEvaluation, \
    PendingEvaluation, Candidate
from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.searcher.bayesopt.models.nphead_acqfunc import \
    EIAcquisitionFunction
from autogluon.searcher.bayesopt.models.gpmxnet import GPMXNetModel, \
    default_gpmodel, default_gpmodel_mcmc
from autogluon.searcher.bayesopt.utils.duplicate_detector import \
    DuplicateDetectorEpsilon
from autogluon.searcher.bayesopt.utils.test_objects import Quadratic3d
from autogluon.searcher.bayesopt.tuning_algorithms.bo_algorithm import \
    BayesianOptimizationAlgorithm
from autogluon.searcher.bayesopt.tuning_algorithms.bo_algorithm_components import \
    IndependentThompsonSampling, LBFGSOptimizeAcquisition
from autogluon.searcher.bayesopt.tuning_algorithms.common import \
    RandomCandidateGenerator
from autogluon.searcher.bayesopt.tuning_algorithms.default_algorithm import \
    DEFAULT_METRIC
from autogluon.searcher.bayesopt.gpmxnet.constants import \
    DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG


@pytest.mark.slow
@pytest.mark.parametrize('batch_size, seed, global_minimum', [
    (1, 0, (10.0, 1, '1.0')),
    (2, 0, (100.0, 1, '1.0')),
    (2, 0, (100.0, 2, '2.0')),
])
def test_canonical_loop_bo(batch_size, seed, global_minimum):
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    random_iterations = 30
    gp_loop_iterations = 10

    objective = Quadratic3d(
        global_minimum, active_metric=DEFAULT_METRIC, metric_names=[DEFAULT_METRIC])
    hp_ranges = objective.search_space
    f_min = objective.f_min
    active_metric = objective.active_metric

    candidate_evaluations = []
    failed_candidates = []
    pending_candidates = []
    best_so_far = {
        'best_objective': np.inf,
        'best_candidate': np.inf,
    }

    def _get_current_optim_state() -> TuningJobState:
        return TuningJobState(
                hp_ranges=hp_ranges,
                candidate_evaluations=candidate_evaluations,
                failed_candidates=failed_candidates,
                pending_evaluations=[PendingEvaluation(x) for x in pending_candidates]
        )

    def _evaluate_blackbox_with(candidates: List[Candidate]):

        # contains logic that puts some candidates into pending or failed
        evaluations = [CandidateEvaluation(candidate, objective(candidate)) for candidate in sorted(candidates)]
        for candidate_evaluation in evaluations:
            candidate, evaluation = candidate_evaluation.candidate, candidate_evaluation.metrics
            total_number_of_candidates = len(candidate_evaluations) + \
                len(failed_candidates) + len(pending_candidates)
            if total_number_of_candidates % 5 == 0:  # candidate "slow" to compute
                pending_candidates.append(candidate)
            elif total_number_of_candidates % 7 == 0:  # candidate that "failed"
                failed_candidates.append(candidate)
            else:  # normal candidate
                if best_so_far['best_objective'] > evaluation[active_metric]:
                    best_so_far['best_objective'] = evaluation[active_metric]
                    best_so_far['best_candidate'] = candidate
                candidate_evaluations.append(CandidateEvaluation(candidate, evaluation))

    def random_generator():
        return RandomCandidateGenerator(
            _get_current_optim_state().hp_ranges, random_seed=seed)

    random_candidate_generator = random_generator()
    init_candidates = []
    for candidate in random_candidate_generator.generate_candidates():
        init_candidates.append(candidate)
        if len(init_candidates) > random_iterations:
            break

    _evaluate_blackbox_with(init_candidates)

    def _get_next_candidates_with_bo():
        state = _get_current_optim_state()
        gpmodel = default_gpmodel(state, 0, DEFAULT_OPTIMIZATION_CONFIG)
        model = GPMXNetModel(
            state, active_metric, 0, gpmodel, fit_parameters=True,
            num_fantasy_samples=20)
        done_candidates = [candidate_evaluation.candidate for candidate_evaluation in candidate_evaluations]
        blacklisted_candidates = set(pending_candidates + failed_candidates + done_candidates)
        bo_algo = BayesianOptimizationAlgorithm(
            initial_candidates_generator=random_generator(),
            num_requested_candidates=batch_size,
            initial_candidates_scorer=IndependentThompsonSampling(model=model),
            num_initial_candidates=1000,
            local_optimizer=LBFGSOptimizeAcquisition(state, model, EIAcquisitionFunction),
            pending_candidate_state_transformer=None,
            blacklisted_candidates=blacklisted_candidates,
            greedy_batch_selection=False,
            duplicate_detector=DuplicateDetectorEpsilon(state.hp_ranges)
        )

        candidates = bo_algo.next_candidates()
        candidates = list(sorted(candidates))
        return candidates

    for _ in range(gp_loop_iterations):
        bo_candidates = _get_next_candidates_with_bo()
        _evaluate_blackbox_with(bo_candidates)

    # is the best found candidate reasonable?
    assert abs(f_min - best_so_far['best_objective']) < 0.4

    # are there duplicate candidates suggested?
    total_candidates = candidate_evaluations + failed_candidates + pending_candidates
    unique_candidate_evaluations = []
    for candidate_evaluation in candidate_evaluations:
        if candidate_evaluation not in unique_candidate_evaluations:
            unique_candidate_evaluations.append(candidate_evaluation)
    assert len(unique_candidate_evaluations) + len(set(failed_candidates + pending_candidates)) == len(total_candidates)

    # are any pending or excluded candidates evaluated multiple times?
    normal_evaluations = set(ce.candidate for ce in candidate_evaluations)
    failed_candidates = set(failed_candidates)

    pending_candidates = set(pending_candidates)
    assert len(failed_candidates.intersection(normal_evaluations)) == 0
    assert len(pending_candidates.intersection(normal_evaluations)) == 0
