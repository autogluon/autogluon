import numpy as np
import mxnet as mx

from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52
from autogluon.searcher.bayesopt.gpmxnet.warping import WarpedKernel
from autogluon.searcher.bayesopt.models.gpmxnet import \
    get_internal_candidate_evaluations, build_kernel, \
    dimensionality_and_warping_ranges
from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState, CandidateEvaluation
from autogluon.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRangeCategorical, HyperparameterRangeInteger, \
    HyperparameterRangeContinuous, HyperparameterRanges_Impl
from autogluon.searcher.bayesopt.datatypes.scaling import LinearScaling, \
    LogScaling
from autogluon.searcher.bayesopt.tuning_algorithms.default_algorithm import \
    dictionarize_objective, DEFAULT_METRIC
from autogluon.searcher.bayesopt.models.mxnet_base import \
    _compute_mean_across_samples
from autogluon.searcher.bayesopt.models.nphead_acqfunc import \
    _reshape_predictions


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


def test_compute_mean_across_samples():
    # Assume we predict on 2 test data points
    # GP without MCMC, no fantasizing, means and stds are all of shape (2,)
    means = mx.nd.array([1., 0.5])
    stds = mx.nd.array([0.5, 1.])
    expected_means = np.array([1., 0.5])
    prediction_list = [(means, stds)]
    pred_means = _compute_mean_across_samples(prediction_list)
    np.testing.assert_equal(expected_means, pred_means.asnumpy())

    # GP with 2 MCMC samples, no fantasizing, means and stds are all of shape (2,)
    means_2 = mx.nd.array([0.5, 1.])
    expected_means = np.array([0.75, 0.75])
    prediction_list = [(means, stds), (means_2, stds)]
    pred_means = _compute_mean_across_samples(prediction_list)
    np.testing.assert_equal(expected_means, pred_means.asnumpy())

    # GP with 2 MCMC samples, fantasizing with 3 samples, means has the shape (2, 3)
    means = mx.nd.array([[1., 0.8, 1.2], [0.5, 0.3, 0.7]])
    stds = mx.nd.array([0.5, 1.])
    means_2 = mx.nd.array([[0, 0.6, 0.8], [0.3, 0.5, 0.5]])
    expected_means = np.array([[0.5, 0.7, 1.0], [0.4, 0.4, 0.6]])
    prediction_list = [(means, stds), (means_2, stds)]
    pred_means = _compute_mean_across_samples(prediction_list)
    np.testing.assert_array_almost_equal(expected_means, pred_means.asnumpy())


def test_reshape_predictions():
    # without MCMC samples, without fantasizing
    predictions = [(mx.nd.array([1,]), mx.nd.array([0.1,]))]
    expected = (mx.nd.array([1,]), mx.nd.array([0.1,]))
    result = _reshape_predictions(predictions)
    np.testing.assert_equal(expected[0].asnumpy(), result[0].asnumpy())
    np.testing.assert_equal(expected[1].asnumpy(), result[1].asnumpy())

    # 3 MCMC samples, without fantasizing
    predictions = [(mx.nd.array([1,]), mx.nd.array([0.1,])),
                   (mx.nd.array([2,]), mx.nd.array([0.2,])),
                   (mx.nd.array([3,]), mx.nd.array([0.3,]))]
    expected = (mx.nd.array([1,2,3]), mx.nd.array([0.1,0.2,0.3]))
    result = _reshape_predictions(predictions)
    np.testing.assert_equal(expected[0].asnumpy(), result[0].asnumpy())
    np.testing.assert_equal(expected[1].asnumpy(), result[1].asnumpy())

    # 3 MCMC samples, with fantasizing of 2 samples
    predictions = [(mx.nd.array([1,2]), mx.nd.array([0.1,])),
                   (mx.nd.array([2,3]), mx.nd.array([0.2,])),
                   (mx.nd.array([3,4]), mx.nd.array([0.3,]))]
    expected = (mx.nd.array([1,2,2,3,3,4]), mx.nd.array([0.1,0.1,0.2,0.2,0.3,0.3]))
    result = _reshape_predictions(predictions)
    np.testing.assert_equal(expected[0].asnumpy(), result[0].asnumpy())
    np.testing.assert_equal(expected[1].asnumpy(), result[1].asnumpy())


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
