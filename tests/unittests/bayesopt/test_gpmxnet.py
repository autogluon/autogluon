from typing import List
import numpy as np
import mxnet
import pytest

from autogluon.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.searcher.bayesopt.datatypes.common import CandidateEvaluation, \
    PendingEvaluation
from autogluon.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRangeContinuous, HyperparameterRanges_Impl
from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.searcher.bayesopt.models.gpmxnet import default_gpmodel, \
    default_gpmodel_mcmc, GPMXNetModel
from autogluon.searcher.bayesopt.models.nphead_acqfunc import \
    EIAcquisitionFunction
from autogluon.searcher.bayesopt.tuning_algorithms.default_algorithm import \
    DEFAULT_METRIC, dictionarize_objective
from autogluon.searcher.bayesopt.gpmxnet.constants import DEFAULT_MCMC_CONFIG, \
    DEFAULT_OPTIMIZATION_CONFIG


@pytest.fixture(scope='function')
def tuning_job_state() -> TuningJobState:
    X = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]

    return TuningJobState(
        HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('x', 0.0, 1.0, LinearScaling()),
            HyperparameterRangeContinuous('y', 0.0, 1.0, LinearScaling()),
        ),
        [
            CandidateEvaluation(x, y) for x, y in zip(X, Y)
        ],
        [], []
    )


def _set_seeds(seed=0):
    mxnet.random.seed(seed)
    np.random.seed(seed)


def test_gp_fit(tuning_job_state):
    _set_seeds(0)
    X = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    Y = [np.sum(x) * 10.0 for x in X]

    # checks if fitting is running
    random_seed = 0
    gpmodel = default_gpmodel(tuning_job_state, random_seed,
        optimization_config=DEFAULT_OPTIMIZATION_CONFIG)
    model = GPMXNetModel(
        tuning_job_state, DEFAULT_METRIC, random_seed, gpmodel,
        fit_parameters=True, num_fantasy_samples=20)

    X = [tuning_job_state.hp_ranges.to_ndarray(x) for x in X]
    Y_mean, Y_std = model.predict(np.array(X))[0]

    assert np.all(np.abs(Y_mean - Y) < 1e-1), \
        "in a noiseless setting, mean of GP should coincide closely with outputs at training points"

    X_test = [
        (0.2, 0.2),
        (0.4, 0.2),
        (0.1, 0.9),
        (0.5, 0.5),
    ]
    X_test = [tuning_job_state.hp_ranges.to_ndarray(x) for x in X_test]

    Y_mean_test, Y_std_test = model.predict(np.array(X_test))[0]
    assert np.min(Y_std) < np.min(Y_std_test), \
        "Standard deviation on un-observed points should be greater than at observed ones"


def test_gp_mcmc_fit(tuning_job_state):

    def tuning_job_state_mcmc(X, Y) -> TuningJobState:
        Y = [dictionarize_objective(y) for y in Y]

        return TuningJobState(
            HyperparameterRanges_Impl(HyperparameterRangeContinuous('x', -4., 4., LinearScaling())),
            [CandidateEvaluation(x, y) for x, y in zip(X, Y)],
            [], []
        )

    _set_seeds(0)

    def f(x):
        return 0.1 * np.power(x, 3)

    X = np.concatenate((np.random.uniform(-4., -1., 10), np.random.uniform(1., 4., 10)))
    Y = f(X)
    X_test = np.sort(np.random.uniform(-1., 1., 10))

    X = [(x,)for x in X]
    X_test = [(x,) for x in X_test]

    tuning_job_state = tuning_job_state_mcmc(X, Y)
    # checks if fitting is running
    random_seed = 0
    gpmodel = default_gpmodel_mcmc(tuning_job_state, random_seed, mcmc_config=DEFAULT_MCMC_CONFIG)
    model = GPMXNetModel(
        tuning_job_state, DEFAULT_METRIC, random_seed, gpmodel,
        fit_parameters=True, num_fantasy_samples=20)

    X = [tuning_job_state.hp_ranges.to_ndarray(x) for x in X]
    predictions = model.predict(np.array(X))

    Y_std_list = [stds for means, stds in predictions]
    Y_mean_list = [means for means, stds in predictions]
    Y_mean = np.mean(Y_mean_list, axis=0)
    Y_std = np.mean(Y_std_list, axis=0)

    assert np.all(np.abs(Y_mean - Y) < 1e-1), \
        "in a noiseless setting, mean of GP should coincide closely with outputs at training points"

    X_test = [tuning_job_state.hp_ranges.to_ndarray(x) for x in X_test]

    predictions_test = model.predict(np.array(X_test))
    Y_std_test_list = [stds for means, stds in predictions_test]
    Y_std_test = np.mean(Y_std_test_list, axis=0)
    assert np.max(Y_std) < np.min(Y_std_test), \
        "Standard deviation on un-observed points should be greater than at observed ones"


def test_gp_fantasizing():
    """
    Compare whether acquisition function evaluations (values, gradients) with
    fantasizing are the same as averaging them by hand.
    """
    random_seed = 4567
    _set_seeds(random_seed)
    num_fantasy_samples = 10
    num_pending = 5

    hp_ranges = HyperparameterRanges_Impl(
        HyperparameterRangeContinuous('x', 0.0, 1.0, LinearScaling()),
        HyperparameterRangeContinuous('y', 0.0, 1.0, LinearScaling()))
    X = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    num_data = len(X)
    Y = [dictionarize_objective(np.random.randn(1, 1)) for _ in range(num_data)]
    # Draw fantasies. This is done for a number of fixed pending candidates
    # The model parameters are fit in the first iteration, when there are
    # no pending candidates

    # Note: It is important to not normalize targets, because this would be
    # done on the observed targets only, not the fantasized ones, so it
    # would be hard to compare below.
    pending_evaluations = []
    for _ in range(num_pending):
        pending_cand = tuple(np.random.rand(2,))
        pending_evaluations.append(PendingEvaluation(pending_cand))
    state = TuningJobState(
        hp_ranges,
        [CandidateEvaluation(x, y) for x, y in zip(X, Y)],
        failed_candidates=[],
        pending_evaluations=pending_evaluations)
    gpmodel = default_gpmodel(
        state, random_seed, optimization_config=DEFAULT_OPTIMIZATION_CONFIG)
    model = GPMXNetModel(
        state, DEFAULT_METRIC, random_seed, gpmodel, fit_parameters=True,
        num_fantasy_samples=num_fantasy_samples, normalize_targets=False)
    fantasy_samples = model.fantasy_samples
    # Evaluate acquisition function and gradients with fantasizing
    num_test = 50
    X_test = np.vstack([hp_ranges.to_ndarray(
        tuple(np.random.rand(2,))) for _ in range(num_test)])
    acq_func = EIAcquisitionFunction(model)
    fvals, grads = acq_func.compute_acq_with_gradients(X_test)
    # Do the same computation by averaging by hand
    fvals_cmp = np.empty((num_fantasy_samples,) + fvals.shape)
    grads_cmp = np.empty((num_fantasy_samples,) + grads.shape)
    X_full = X + state.pending_candidates
    for it in range(num_fantasy_samples):
        Y_full = Y + [dictionarize_objective(eval.fantasies[DEFAULT_METRIC][:, it])
                      for eval in fantasy_samples]
        state2 = TuningJobState(
            hp_ranges,
            [CandidateEvaluation(x, y) for x, y in zip(X_full, Y_full)],
            failed_candidates=[],
            pending_evaluations=[])
        # We have to skip parameter optimization here
        model2 = GPMXNetModel(
            state2, DEFAULT_METRIC, random_seed, gpmodel,
            fit_parameters=False, num_fantasy_samples=num_fantasy_samples,
            normalize_targets=False)
        acq_func2 = EIAcquisitionFunction(model2)
        fvals_, grads_ = acq_func2.compute_acq_with_gradients(X_test)
        fvals_cmp[it, :] = fvals_
        grads_cmp[it, :] = grads_
    # Comparison
    fvals2 = np.mean(fvals_cmp, axis=0)
    grads2 = np.mean(grads_cmp, axis=0)
    assert np.allclose(fvals, fvals2)
    assert np.allclose(grads, grads2)


def default_models() -> List[GPMXNetModel]:
    X = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),  # same evals are added multiple times to force GP to unlearn prior
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]

    state = TuningJobState(
        HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('x', 0.0, 1.0, LinearScaling()),
            HyperparameterRangeContinuous('y', 0.0, 1.0, LinearScaling()),
        ),
        [
            CandidateEvaluation(x, y) for x, y in zip(X, Y)
        ],
        [], [],
    )
    random_seed = 0

    gpmodel = default_gpmodel(
        state, random_seed=random_seed,
        optimization_config=DEFAULT_OPTIMIZATION_CONFIG
    )

    gpmodel_mcmc = default_gpmodel_mcmc(
        state, random_seed=random_seed,
        mcmc_config=DEFAULT_MCMC_CONFIG
    )

    return [
        GPMXNetModel(state, DEFAULT_METRIC, random_seed, gpmodel,
                     fit_parameters=True, num_fantasy_samples=20),
        GPMXNetModel(state, DEFAULT_METRIC, random_seed, gpmodel_mcmc,
                     fit_parameters=True, num_fantasy_samples=20)]


def test_current_best():
    for model in default_models():
        current_best = model.current_best().item()
        print(current_best)
        assert -0.1 < current_best < 0.1
