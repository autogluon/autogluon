import numpy as np

from autogluon.core.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState, CandidateEvaluation
from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRangeCategorical, HyperparameterRangeInteger, \
    HyperparameterRangeContinuous, HyperparameterRanges_Impl
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling, \
    LogScaling
from autogluon.core.searcher.bayesopt.models.gp_model import \
    get_internal_candidate_evaluations
from autogluon.core.searcher.bayesopt.tuning_algorithms.default_algorithm import \
    dictionarize_objective, DEFAULT_METRIC
from autogluon.core.searcher.bayesopt.utils.test_objects import \
    dimensionality_and_warping_ranges


def test_get_internal_candidate_evaluations():
    """we do not test the case with no evaluations, as it is assumed
    that there will be always some evaluations generated in the beginning
    of the BO loop."""

    candidates = [
        CandidateEvaluation((2, 3.3, 'X'), dictionarize_objective(5.3)),
        CandidateEvaluation((1, 9.9, 'Y'), dictionarize_objective(10.9)),
        CandidateEvaluation((7, 6.1, 'X'), dictionarize_objective(13.1)),
    ]

    state = TuningJobState(
        hp_ranges=HyperparameterRanges_Impl(
            HyperparameterRangeInteger('integer', 0, 10, LinearScaling()),
            HyperparameterRangeContinuous('real', 0, 10, LinearScaling()),
            HyperparameterRangeCategorical('categorical', ('X', 'Y')),
        ),
        candidate_evaluations=candidates,
        failed_candidates=[candidates[0].candidate],  # these should be ignored by the model
        pending_evaluations=[]
    )

    result = get_internal_candidate_evaluations(
        state, DEFAULT_METRIC, normalize_targets=True,
        num_fantasize_samples=20)

    assert len(result.X.shape) == 2, "Input should be a matrix"
    assert len(result.y.shape) == 2, "Output should be a matrix"

    assert result.X.shape[0] == len(candidates)
    assert result.y.shape[-1] == 1, "Only single output value per row is suppored"

    assert np.abs(np.mean(result.y)) < 1e-8, "Mean of the normalized outputs is not 0.0"
    assert np.abs(np.std(result.y) - 1.0) < 1e-8, "Std. of the normalized outputs is not 1.0"

    np.testing.assert_almost_equal(result.mean, 9.766666666666666)
    np.testing.assert_almost_equal(result.std, 3.283629428273267)


def test_dimensionality_and_warping_ranges():
    hp_ranges = HyperparameterRanges_Impl(
        HyperparameterRangeCategorical('categorical1', ('X', 'Y')),
        HyperparameterRangeContinuous('integer', 0.1, 10.0, LogScaling()),
        HyperparameterRangeCategorical('categorical2', ('a', 'b', 'c')),
        HyperparameterRangeContinuous('real', 0.0, 10.0, LinearScaling(), 2.5, 5.0),
        HyperparameterRangeCategorical('categorical3', ('X', 'Y')),
    )

    dim, warping_ranges = dimensionality_and_warping_ranges(hp_ranges)
    assert dim == 9
    assert warping_ranges == {
        2: (0.0, 1.0),
        6: (0.0, 1.0)
    }
