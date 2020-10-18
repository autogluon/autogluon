import pytest

from autogluon.core.searcher.bayesopt.datatypes.common import CandidateEvaluation
from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRangeContinuous, HyperparameterRangeInteger, \
    HyperparameterRangeCategorical, HyperparameterRanges_Impl
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.core.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState


@pytest.fixture(scope='function')
def tuning_job_state():
    return {'algo-1': TuningJobState(
        hp_ranges=HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('a1_hp_1', -5.0, 5.0, LinearScaling()),
            HyperparameterRangeCategorical('a1_hp_2', ('a', 'b', 'c'))),
        candidate_evaluations=[CandidateEvaluation(candidate=(-3.0, 'a'), value=1.0),
                               CandidateEvaluation(candidate=(-1.9, 'c'), value=2.0),
                               CandidateEvaluation(candidate=(-3.5, 'a'), value=0.3)],
        failed_candidates=[],
        pending_evaluations=[]
    ),
        'algo-2': TuningJobState(
            hp_ranges=HyperparameterRanges_Impl(
                HyperparameterRangeContinuous('a2_hp_1', -5.0, 5.0, LinearScaling()),
                HyperparameterRangeInteger('a2_hp_2', -5, 5, LinearScaling(), -5, 5)),
            candidate_evaluations=[CandidateEvaluation(candidate=(-1.9, -1), value=0.0),
                                   CandidateEvaluation(candidate=(-3.5, 3), value=2.0)],
            failed_candidates=[],
            pending_evaluations=[]
        )
    }


@pytest.fixture(scope='function')
def tuning_job_sub_state():
    return TuningJobState(
        hp_ranges=HyperparameterRanges_Impl(),
        candidate_evaluations=[],
        failed_candidates=[],
        pending_evaluations=[])
