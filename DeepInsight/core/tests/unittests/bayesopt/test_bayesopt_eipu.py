from typing import List
import numpy as np

from autogluon.core.searcher.bayesopt.datatypes.common import \
    CandidateEvaluation
from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges_Impl, HyperparameterRangeContinuous
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.core.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.core.searcher.bayesopt.gpautograd.constants import \
    DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG
from autogluon.core.searcher.bayesopt.models.meanstd_acqfunc_impl import \
    EIpuAcquisitionFunction
from autogluon.core.searcher.bayesopt.models.gp_model import \
    GaussProcSurrogateModel
from autogluon.core.searcher.bayesopt.tuning_algorithms.bo_algorithm_components import \
    LBFGSOptimizeAcquisition
from autogluon.core.searcher.bayesopt.tuning_algorithms.base_classes import \
    dictionarize_objective, DEFAULT_METRIC, DEFAULT_COST_METRIC
from autogluon.core.searcher.bayesopt.utils.test_objects import \
    default_gpmodel, default_gpmodel_mcmc


def default_models(metric, do_mcmc=True) -> List[GaussProcSurrogateModel]:
    X = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    if metric == DEFAULT_METRIC:
        Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]
    elif metric == DEFAULT_COST_METRIC:
        # Increasing the first hp increases cost
        Y = [{metric: 1.0 + x[0] * 2.0}
             for x in X]
    else:
        raise ValueError(f"{metric} is not a valid metric")

    state = TuningJobState(
        HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('x', 0.0, 1.0, LinearScaling()),
            HyperparameterRangeContinuous('y', 0.0, 1.0, LinearScaling()),
        ),
        [
            CandidateEvaluation(x, y) for x, y in zip(X, Y)
        ],
        [], []
    )
    random_seed = 0

    gpmodel = default_gpmodel(
        state, random_seed=random_seed,
        optimization_config=DEFAULT_OPTIMIZATION_CONFIG)
    result = [GaussProcSurrogateModel(
        state, metric, random_seed, gpmodel, fit_parameters=True,
        num_fantasy_samples=20)]
    if do_mcmc:
        gpmodel_mcmc = default_gpmodel_mcmc(
            state, random_seed=random_seed,
            mcmc_config=DEFAULT_MCMC_CONFIG)
        result.append(
            GaussProcSurrogateModel(
                state, metric, random_seed, gpmodel_mcmc,
                fit_parameters=True,num_fantasy_samples=20))
    return result


def plot_ei_mean_std(model, eipu, max_grid=1.0):
    import matplotlib.pyplot as plt

    grid = np.linspace(0, max_grid, 400)
    Xgrid, Ygrid = np.meshgrid(grid, grid)
    inputs = np.hstack([Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1)])
    Z_ei = eipu.compute_acq(inputs)[0]
    predictions = model.predict(inputs)[0]
    Z_means = predictions['mean']
    Z_std = predictions['std']
    titles = ['EIpu', 'mean', 'std']
    for i, (Z, title) in enumerate(zip([Z_ei, Z_means, Z_std], titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(
            Z.reshape(Xgrid.shape), extent=[0, max_grid, 0, max_grid],
            origin='lower')
        plt.colorbar()
        plt.title(title)
    plt.show()


# Note: This test fails when run with GP MCMC model. There, acq[5] > acq[7], and acq[8] > acq[5]
# ==> Need to look into GP MCMC model
def test_sanity_check():
    # - test that values are negative as we should be returning *minus* expected improvement
    # - test that values that are further from evaluated candidates have higher expected improvement
    #   given similar mean
    # - test that points closer to better points have higher expected improvement
    active_models = default_models(DEFAULT_METRIC, do_mcmc=False)
    cost_models = default_models(DEFAULT_COST_METRIC, do_mcmc=False)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {DEFAULT_METRIC: active_model, DEFAULT_COST_METRIC: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=DEFAULT_METRIC)
        X = np.array([
            (0.0, 0.0),  # 0
            (1.0, 0.0),  # 1
            (0.0, 1.0),  # 2
            (1.0, 1.0),  # 3
            (0.2, 0.0),  # 4
            (0.0, 0.2),  # 5
            (0.1, 0.0),  # 6
            (0.0, 0.1),  # 7
            (0.1, 0.1),  # 8
            (0.9, 0.9),  # 9
        ])
        acq = list(eipu.compute_acq(X).flatten())
        assert all(a <= 0 for a in acq), acq

        # lower evaluations with lower cost should correspond to better acquisition
        # second inequality is less equal because last two values are likely zero
        assert acq[0] < acq[1] <= acq[3], acq
        assert acq[8] < acq[9], acq

        # evaluations with same EI but lower cost give a better EIpu
        assert acq[5] < acq[4]
        assert acq[7] < acq[6]

        # further from an evaluated point should correspond to better acquisition
        assert acq[6] < acq[4] < acq[1], acq
        assert acq[7] < acq[5] < acq[2], acq


def test_best_value():
    # test that the best value affects the cost-aware expected improvement
    active_models = default_models(DEFAULT_METRIC)
    cost_models = default_models(DEFAULT_COST_METRIC)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {DEFAULT_METRIC: active_model, DEFAULT_COST_METRIC: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=DEFAULT_METRIC)

        random = np.random.RandomState(42)
        test_X = random.uniform(low=0.0, high=1.0, size=(10, 2))

        acq_best0 = list(eipu.compute_acq(test_X).flatten())
        zero_row = np.zeros((1, 2))
        acq0_best0 = eipu.compute_acq(zero_row)

        # override current best
        def new_current_best():
            return np.array([30])
        models[DEFAULT_METRIC].current_best = new_current_best

        acq_best2 = list(eipu.compute_acq(test_X).flatten())
        acq0_best2 = eipu.compute_acq(zero_row)

        # if the best is only 2 the acquisition function should be better (lower value)
        assert all(a2 < a0 for a2, a0 in zip(acq_best2, acq_best0))

        # there should be a considerable gap at the point of the best evaluation
        assert acq0_best2 < acq0_best0 - 1.0


def test_optimization_improves():
    debug_output = False
    # Pick a random point, optimize and the expected improvement should be better:
    # But only if the starting point is not too far from the origin
    random = np.random.RandomState(42)
    active_models = default_models(DEFAULT_METRIC)
    cost_models = default_models(DEFAULT_COST_METRIC)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {DEFAULT_METRIC: active_model, DEFAULT_COST_METRIC: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=DEFAULT_METRIC)
        opt = LBFGSOptimizeAcquisition(
            models[DEFAULT_METRIC].state, models, EIpuAcquisitionFunction, DEFAULT_METRIC)
        if debug_output:
            print('\n\nGP MCMC' if models.does_mcmc() else 'GP Opt')
            fzero = eipu.compute_acq(np.zeros((1, 2)))[0]
            print('f(0) = {}'.format(fzero))
        if debug_output and not models.does_mcmc():
            print('Hyperpars: {}'.format(models.get_params()))
            # Plot the thing!
            plot_ei_mean_std(models, eipu, max_grid=0.001)
            plot_ei_mean_std(models, eipu, max_grid=0.01)
            plot_ei_mean_std(models, eipu, max_grid=0.1)
            plot_ei_mean_std(models, eipu, max_grid=1.0)

        non_zero_acq_at_least_once = False
        for iter in range(10):
            initial_point = random.uniform(low=0.0, high=0.1, size=(2,))
            acq0, df0 = eipu.compute_acq_with_gradient(initial_point)
            if debug_output:
                print('\nInitial point: f(x0) = {}, x0 = {}'.format(
                    acq0, initial_point))
                print('grad0 = {}'.format(df0))
            if acq0 != 0:
                non_zero_acq_at_least_once = True
                optimized = np.array(opt.optimize(tuple(initial_point)))
                acq_opt = eipu.compute_acq(optimized)[0]
                if debug_output:
                    print('Final point: f(x1) = {}, x1 = {}'.format(
                        acq_opt, optimized))
                assert acq_opt < 0
                assert acq_opt < acq0

        assert non_zero_acq_at_least_once


def test_numerical_gradient():
    debug_output = False
    random = np.random.RandomState(42)
    eps = 1e-6

    active_models = default_models(DEFAULT_METRIC)
    cost_models = default_models(DEFAULT_COST_METRIC)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {DEFAULT_METRIC: active_model, DEFAULT_COST_METRIC: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=DEFAULT_METRIC)

        for iter in range(10):
            high = 1.0 if iter < 5 else 0.02
            x = random.uniform(low=0.0, high=high, size=(2,))
            f0, analytical_gradient = eipu.compute_acq_with_gradient(x)
            analytical_gradient = analytical_gradient.flatten()
            if debug_output:
                print('x0 = {}, f(x_0) = {}, grad(x_0) = {}'.format(
                    x, f0, analytical_gradient))

            for i in range(2):
                h = np.zeros_like(x)
                h[i] = eps
                fpeps = eipu.compute_acq(x+h)[0]
                fmeps = eipu.compute_acq(x-h)[0]
                numerical_derivative = (fpeps - fmeps) / (2 * eps)
                if debug_output:
                    print('f(x0+eps) = {}, f(x0-eps) = {}, findiff = {}, deriv = {}'.format(
                        fpeps[0], fmeps[0], numerical_derivative[0],
                        analytical_gradient[i]))
                np.testing.assert_almost_equal(
                    numerical_derivative.item(), analytical_gradient[i],
                    decimal=4)


def test_value_same_as_with_gradient():
    # test that compute_acq and compute_acq_with_gradients return the same acquisition values
    active_models = default_models(DEFAULT_METRIC)
    cost_models = default_models(DEFAULT_COST_METRIC)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {DEFAULT_METRIC: active_model, DEFAULT_COST_METRIC: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=DEFAULT_METRIC)

        random = np.random.RandomState(42)
        X = random.uniform(low=0.0, high=1.0, size=(10, 2))

        # assert same as computation with gradients
        vec1 = eipu.compute_acq(X).flatten()
        vec2 = np.array([eipu.compute_acq_with_gradient(x)[0] for x in X])
        np.testing.assert_almost_equal(vec1, vec2)


if __name__ == "__main__":
    test_optimization_improves()
    test_numerical_gradient()
