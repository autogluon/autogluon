from typing import List, Set
import pytest

from autogluon.core.searcher.bayesopt.datatypes.common import Candidate, \
    CandidateEvaluation, PendingEvaluation
from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges_Impl, HyperparameterRangeInteger, \
    HyperparameterRangeCategorical, HyperparameterRangeContinuous
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.core.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.core.searcher.bayesopt.tuning_algorithms.common import \
    compute_blacklisted_candidates, generate_unique_candidates, \
    RandomCandidateGenerator
from autogluon.core.searcher.bayesopt.tuning_algorithms.default_algorithm \
    import dictionarize_objective
from autogluon.core.searcher.bayesopt.utils.test_objects import \
    RepeatedCandidateGenerator


@pytest.fixture(scope='function')
def hp_ranges():
    return HyperparameterRanges_Impl(
        HyperparameterRangeInteger('hp1', 0, 200, LinearScaling()),
        HyperparameterRangeCategorical('hp2', ('a', 'b', 'c')))


@pytest.fixture(scope='function')
def multi_algo_state():
    def _candidate_evaluations(num):
        return [
            CandidateEvaluation(
                candidate=(i,),
                metrics=dictionarize_objective(float(i)))
            for i in range(num)]

    return {
        '0': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(
                HyperparameterRangeContinuous('a1_hp_1', -5.0, 5.0, LinearScaling(), -5.0, 5.0)),
            candidate_evaluations=_candidate_evaluations(2),
            failed_candidates=[(i,) for i in range(3)],
            pending_evaluations=[PendingEvaluation((i,)) for i in range(100)]
        ),
        '1': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(),
            candidate_evaluations=_candidate_evaluations(5),
            failed_candidates=[],
            pending_evaluations=[]
        ),
        '2': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(),
            candidate_evaluations=_candidate_evaluations(3),
            failed_candidates=[(i,) for i in range(10)],
            pending_evaluations=[PendingEvaluation((i,)) for i in range(1)]
        ),
        '3': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(),
            candidate_evaluations=_candidate_evaluations(6),
            failed_candidates=[],
            pending_evaluations=[]
        ),
        '4': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(),
            candidate_evaluations=_candidate_evaluations(120),
            failed_candidates=[],
            pending_evaluations=[]
        ),
    }


@pytest.mark.parametrize('candidate_evaluations,failed_candidates,pending_evaluations,expected', [
    ([], [], [], set()),
    ([CandidateEvaluation((123, 'a'), {'': 9.87})], [], [], {(123, 'a')}),
    ([], [(123, 'a')], [], {(123, 'a')}),
    ([], [], [PendingEvaluation((123, 'a'))], {(123, 'a')}),
    ([CandidateEvaluation((1, 'a'), {'': 9.87})], [(2, 'b')],
     [PendingEvaluation((3, 'c'))], {(1, 'a'), (2, 'b'), (3, 'c')})
])
def test_compute_blacklisted_candidates(
        hp_ranges: HyperparameterRanges_Impl,
        candidate_evaluations: List[CandidateEvaluation],
        failed_candidates: List[Candidate],
        pending_evaluations: List[PendingEvaluation],
        expected: Set[Candidate]):
    state = TuningJobState(
        hp_ranges, candidate_evaluations,
        failed_candidates, pending_evaluations
    )
    actual = compute_blacklisted_candidates(state)
    assert set(expected) == set(actual)


@pytest.mark.parametrize('num_unique_candidates,num_requested_candidates', [
    (5, 10),
    (15, 10)
])
def test_generate_unique_candidates(num_unique_candidates, num_requested_candidates):
    candidates = generate_unique_candidates(RepeatedCandidateGenerator(num_unique_candidates),
                                            num_requested_candidates, set())
    assert len(candidates) == min(num_unique_candidates, num_requested_candidates)
    assert len(candidates) == len(set(candidates)) # make sure they are unique

    # introduce excluded candidates, simply take a few already unique
    size_excluded = len(candidates) // 2
    excluded = list(candidates)[:size_excluded]
    excluded = set(excluded)

    candidates = generate_unique_candidates(RepeatedCandidateGenerator(num_unique_candidates),
                                            num_requested_candidates, excluded)

    # total unique candidates are adjusted by the number of excluded candidates which are unique too due to set()
    assert len(candidates) == min(num_unique_candidates - len(excluded), num_requested_candidates)
    assert len(candidates) == len(set(candidates)) # make sure they are unique


def test_generate_unique_candidates_fixed_choice(multi_algo_state):
    random_generator = RandomCandidateGenerator(
        multi_algo_state['0'].hp_ranges, 0)
    candidates = []

    for candidate in random_generator.generate_candidates():
        candidates.append(candidate)
        if len(candidates) > 9:
            break

    random_generator = RandomCandidateGenerator(
        multi_algo_state['0'].hp_ranges, 0)
    candidates_2 = generate_unique_candidates(random_generator, 10, set())
    assert candidates == candidates_2
