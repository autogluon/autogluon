import numpy as np
import mxnet as mx
import pytest

from autogluon.searcher.bayesopt.gpmxnet import SliceException
from autogluon.searcher.bayesopt.gpmxnet.slice import SliceSampler, \
    slice_sampler_step_out, slice_sampler_step_in
from autogluon.searcher.bayesopt.gpmxnet.distribution import Normal, \
    LogNormal, Horseshoe, Uniform
from autogluon.searcher.bayesopt.gpmxnet.warping import Warping, WarpedKernel
from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52
from autogluon.searcher.bayesopt.gpmxnet.mean import ScalarMeanFunction
from autogluon.searcher.bayesopt.gpmxnet.likelihood import MarginalLikelihood
from autogluon.searcher.bayesopt.gpmxnet.gpr_mcmc import GPRegressionMCMC, \
    _get_gp_hps, _set_gp_hps, _create_likelihood


# This is just to make the tests work. In GPRegressionMCMC, lower and upper
# bounds are dealt with through the encoding
def old_log_likelihood(
        x, distribution, lower=-float('inf'), upper=float('inf')):
    if any(x < lower) or any(x > upper):
        return -float("inf")
    return -distribution(mx.nd, mx.nd.array(x, dtype=np.float64)).asnumpy()


def test_uniform():
    uniform = Uniform(0.0, 1.0)
    lower, upper = 0.0, 1.0
    assert old_log_likelihood(np.array([0.2]), uniform, lower, upper) == \
           old_log_likelihood(np.array([0.3]), uniform, lower, upper)
    assert old_log_likelihood(np.array([2.0]), uniform, lower, upper) == -float("inf")
    assert old_log_likelihood(np.array([-1.0]), uniform, lower, upper) == -float("inf")


def test_normal():
    normal = Normal(0, 1)
    lower, upper =  -1e3, 1e3
    assert old_log_likelihood(np.array([0.0]), normal, lower, upper) > \
           old_log_likelihood(np.array([0.1]), normal, lower, upper)
    assert old_log_likelihood(np.array([0.0]), normal, lower, upper) > \
           old_log_likelihood(np.array([-0.1]), normal, lower, upper)
    assert old_log_likelihood(np.array([1e4]), normal, lower, upper) == -float("inf")
    assert old_log_likelihood(np.array([-1e4]), normal, lower, upper) == -float("inf")


def test_log_normal():
    log_normal = LogNormal(0.0, 1.0)
    lower, upper =  1e-6, 1e9
    assert old_log_likelihood(np.array([1.0]), log_normal, lower, upper) > \
           old_log_likelihood(np.array([1.1]), log_normal, lower, upper)
    assert old_log_likelihood(np.array([1.0]), log_normal, lower, upper) < \
           old_log_likelihood(np.array([0.9]), log_normal, lower, upper)
    assert old_log_likelihood(np.array([1e10]), log_normal, lower, upper) == -float("inf")
    assert old_log_likelihood(np.array([1e-8]), log_normal, lower, upper) == -float("inf")


def test_horse_shoe():
    horse_shoe = Horseshoe(0.1)
    lower, upper = 1e-6, 1e6
    assert old_log_likelihood(np.array([0.01]), horse_shoe, lower, upper) > \
           old_log_likelihood(np.array([0.1]), horse_shoe, lower, upper)
    assert old_log_likelihood(np.array([1e-7]), horse_shoe, lower, upper) == -float("inf")
    assert old_log_likelihood(np.array([1e7]), horse_shoe, lower, upper) == -float("inf")


def test_slice_normal():
    normal = Normal(0, 1)
    slice = SliceSampler(lambda x: old_log_likelihood(x, normal), 1.0, 0)
    samples = slice.sample(np.array([0.0]), 5000, 1, 1)
    np.testing.assert_almost_equal(np.mean(samples), 0.0, decimal=2)
    np.testing.assert_almost_equal(np.std(samples),  1.0, decimal=2)


def test_slice_step_out():
    normal = Normal(0, 1)

    def sliced_log_density(x):
        return old_log_likelihood(np.array([x]), normal)
    # the lower and upper bound should has log density smaller than this log_pivot
    log_pivot = sliced_log_density(1.0)
    random_state = np.random.RandomState(0)
    lower, upper = slice_sampler_step_out(
        log_pivot, 0.1, sliced_log_density, random_state)
    assert lower < -1.0 and upper > 1.0

    log_pivot = sliced_log_density(100)
    with pytest.raises(SliceException):  # the log_pivot is too small so need > 200 steps
        slice_sampler_step_out(log_pivot, 0.1, sliced_log_density, random_state)


def test_slice_step_in():
    normal = Normal(0., 1.)

    def sliced_log_density(x):
        return old_log_likelihood(np.array([x]), normal)
    log_pivot = sliced_log_density(1.)  # the movement should between [-1., 1.] after step in
    random_state = np.random.RandomState(0)
    movement = slice_sampler_step_in(-20.0, 20.0, log_pivot, sliced_log_density, random_state)
    assert -1.0 < movement < 1.0

    with pytest.raises(SliceException):  # when bound is off, should get SliceException
        slice_sampler_step_in(2.0, 10.0, log_pivot, sliced_log_density, random_state)


def test_get_gp_hps():
    mean = ScalarMeanFunction()
    kernel = Matern52(dimension=1)
    warping = Warping(dimension=1, index_to_range={0: (-4., 4.)})
    warped_kernel = WarpedKernel(kernel=kernel, warping=warping)
    likelihood = MarginalLikelihood(kernel=warped_kernel, mean=mean, initial_noise_variance=1e-6)
    likelihood.initialize(ctx=mx.cpu(), force_reinit=True)
    likelihood.hybridize()
    hp_values = _get_gp_hps(likelihood)
    # the oder of hps are noise, mean, covariance scale, bandwidth, warping a, warping b
    np.testing.assert_array_almost_equal(hp_values, np.array([1e-6, 0.0, 1.0, 1.0, 1.0, 1.0]))


def test_set_gp_hps():
    mean = ScalarMeanFunction()
    kernel = Matern52(dimension=1)
    warping = Warping(dimension=1, index_to_range={0: (-4., 4.)})
    warped_kernel = WarpedKernel(kernel=kernel, warping=warping)
    likelihood = MarginalLikelihood(kernel=warped_kernel, mean=mean, initial_noise_variance=1e-6)
    likelihood.initialize(ctx=mx.cpu(), force_reinit=True)
    likelihood.hybridize()
    hp_values = np.array([1e-2, 1.0, 0.5, 0.3, 0.2, 1.1])
    _set_gp_hps(hp_values, likelihood)
    np.testing.assert_array_almost_equal(hp_values, _get_gp_hps(likelihood))


def test_create_likelihood():
    def build_kernel():
        kernel = Matern52(dimension=1)
        warping = Warping(dimension=1, index_to_range={0: (-4., 4.)})
        return WarpedKernel(kernel=kernel, warping=warping)
    likelihood1 = _create_likelihood(build_kernel, ctx=mx.cpu())
    likelihood2 = _create_likelihood(build_kernel, ctx=mx.cpu())
    np.testing.assert_array_almost_equal(_get_gp_hps(likelihood1), _get_gp_hps(likelihood2))


@pytest.mark.skip(reason="Need manual inspection on the plots")
def test_mcmc():
    np.random.seed(7)

    def f_n(x):
        noise = np.random.normal(0.0, 0.25, x.shape[0])
        return 0.1 * np.power(x, 3) + noise

    def f(x):
        return 0.1 * np.power(x, 3)

    x_train = np.concatenate((np.random.uniform(-4., -1., 40), np.random.uniform(1., 4., 40)))
    y_train = f_n(x_train)
    x_test = np.sort(np.random.uniform(-4., 4., 200))

    y_train_mx_nd = mx.nd.array(y_train, dtype=np.float64)
    x_train_mx_nd = mx.nd.array(x_train, dtype=np.float64)
    x_test_mx_nd = mx.nd.array(x_test, dtype=np.float64)

    def build_kernel():
        return WarpedKernel(
            kernel=Matern52(dimension=1),
            warping=Warping(dimension=1, index_to_range={0: (-4., 4.)})
        )

    model_mcmc = GPRegressionMCMC(build_kernel=build_kernel, random_seed=1)
    model_mcmc.fit(x_train_mx_nd, y_train_mx_nd)
    mcmc_predictions = model_mcmc.predict(x_test_mx_nd)

    import matplotlib.pyplot as plt
    for mcmc_mean, mcmc_var in mcmc_predictions:
        mcmc_mean, mcmc_std = mcmc_mean.asnumpy(), np.sqrt(mcmc_var.asnumpy())
        plt.figure()
        plt.scatter(x_train, y_train, color="red", label="observations")
        plt.plot(x_test, f(x_test), color="black", label="ground truth")
        plt.plot(x_test, mcmc_mean, color="blue", label="mcmc prediction")
        plt.fill_between(x_test, mcmc_mean - 1.96*mcmc_std, mcmc_mean + 1.96*mcmc_std, alpha=.5)
        plt.legend()
    plt.show()
