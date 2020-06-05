import numpy as np
import mxnet as mx

from autogluon.searcher.bayesopt.gpmxnet.mean import ScalarMeanFunction
from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52
from autogluon.searcher.bayesopt.gpmxnet.likelihood import MarginalLikelihood
from autogluon.searcher.bayesopt.gpmxnet.gp_regression import \
    GaussianProcessRegression
from autogluon.searcher.bayesopt.gpmxnet.constants import \
    NOISE_VARIANCE_LOWER_BOUND, INVERSE_BANDWIDTHS_LOWER_BOUND
from autogluon.searcher.bayesopt.gpmxnet.gluon_blocks_helpers import \
    LogarithmScalarEncoding, PositiveScalarEncoding


def test_likelihood_encoding():
    mean = ScalarMeanFunction()
    kernel = Matern52(dimension=1)
    likelihood = MarginalLikelihood(mean=mean, kernel=kernel)
    assert isinstance(likelihood.encoding, LogarithmScalarEncoding)
    likelihood = MarginalLikelihood(mean=mean, kernel=kernel, encoding_type="positive")
    assert isinstance(likelihood.encoding, PositiveScalarEncoding)


def test_gp_regression_no_noise():

    def f(x):
        return np.sin(x)/x

    x_train = np.arange(-5, 5, 0.2)# [-5,-4.8,-4.6,...,4.8]
    x_test = np.arange(-4.9, 5, 0.2)# [-4.9, -4.7, -4.5,...,4.9], note that train and test points do not overlap
    y_train = f(x_train)
    y_test = f(x_test)

    # to mx.nd
    y_train_mx_nd = mx.nd.array(y_train)
    x_train_mx_nd = mx.nd.array(x_train)
    x_test_mx_nd = mx.nd.array(x_test)

    model = GaussianProcessRegression(kernel=Matern52(dimension=1))
    model.fit(x_train_mx_nd, y_train_mx_nd)

    # Check that the value of the residual noise variance learned by empirical Bayes is in the same order
    # as the smallest allowed value (since there is no noise)
    noise_variance = model.likelihood.get_noise_variance()
    np.testing.assert_almost_equal(noise_variance, NOISE_VARIANCE_LOWER_BOUND)

    mu_train, var_train = model.predict(x_train_mx_nd)[0]
    mu_test, var_test = model.predict(x_test_mx_nd)[0]

    # back to np.array
    mu_train = mu_train.asnumpy()
    mu_test = mu_test.asnumpy()
    var_train = var_train.asnumpy()
    var_test = var_test.asnumpy()

    np.testing.assert_almost_equal(mu_train, y_train, decimal=4)
    np.testing.assert_almost_equal(var_train, [0.0] * len(var_train), decimal=4)
    # Fewer decimals imposed for the test points
    np.testing.assert_almost_equal(mu_test, y_test, decimal=3)

    # If we wish plot
    # import matplotlib.pyplot as plt
    # plt.plot(x_train, y_train, "r")
    # plt.errorbar(x=x_train,
    #              y=mu_train,
    #              yerr=var_train)
    # plt.plot(x_test, y_test, "b")
    # plt.errorbar(x=x_test,
    #              y=mu_test,
    #              yerr=var_test)
    # plt.show()


def test_gp_regression_with_noise():

    def f(x):
        return np.sin(x)/x

    np.random.seed(7)

    x_train = np.arange(-5, 5, 0.2)# [-5, -4.8, -4.6,..., 4.8]
    x_test = np.arange(-4.9, 5, 0.2)# [-4.9, -4.7, -4.5,..., 4.9], note that train and test points do not overlap
    y_train = f(x_train)
    y_test = f(x_test)

    std_noise = 0.01
    noise_train = np.random.normal(0.0, std_noise,size=y_train.shape)

    # to mx.nd
    y_train_mx_nd = mx.nd.array(y_train)
    noise_train_mx_nd = mx.nd.array(noise_train)
    x_train_mx_nd = mx.nd.array(x_train)
    x_test_mx_nd = mx.nd.array(x_test)

    model = GaussianProcessRegression(kernel=Matern52(dimension=1))
    model.fit(x_train_mx_nd, y_train_mx_nd + noise_train_mx_nd)

    # Check that the value of the residual noise variance learned by empirical Bayes is in the same order as std_noise^2
    noise_variance = model.likelihood.get_noise_variance()
    np.testing.assert_almost_equal(noise_variance, std_noise**2, decimal=4)

    mu_train, _ = model.predict(x_train_mx_nd)[0]
    mu_test, _ = model.predict(x_test_mx_nd)[0]

    # back to np.array
    mu_train = mu_train.asnumpy()
    mu_test = mu_test.asnumpy()

    np.testing.assert_almost_equal(mu_train, y_train, decimal=2)
    np.testing.assert_almost_equal(mu_test, y_test, decimal=2)


def test_gp_regression_2d_with_ard():

    def f(x):
        # Only dependent on the first column of x
        return np.sin(x[:,0])/x[:,0]

    np.random.seed(7)

    dimension = 3

    # 30 train and test points in R^3
    x_train = np.random.uniform(-5, 5, size=(30,dimension))
    x_test = np.random.uniform(-5, 5, size=(30,dimension))
    y_train = f(x_train)
    y_test = f(x_test)

    # to mx.nd
    y_train_mx_nd = mx.nd.array(y_train)
    x_train_mx_nd = mx.nd.array(x_train)
    x_test_mx_nd = mx.nd.array(x_test)

    model = GaussianProcessRegression(kernel=Matern52(dimension=dimension, ARD=True))
    model.fit(x_train_mx_nd, y_train_mx_nd)

    # Check that the value of the residual noise variance learned by empirical Bayes is in the same order as the smallest allowed value (since there is no noise)
    noise_variance = model.likelihood.get_noise_variance()
    np.testing.assert_almost_equal(noise_variance, NOISE_VARIANCE_LOWER_BOUND)

    # Check that the bandwidths learned by empirical Bayes reflect the fact that only the first column is useful
    # In particular, for the useless dimensions indexed by {1,2}, the inverse bandwidths should be close to INVERSE_BANDWIDTHS_LOWER_BOUND
    # (or conversely, bandwidths should be close to their highest allowed values)
    sqd = model.likelihood.kernel.squared_distance
    inverse_bandwidths = sqd.encoding.get(mx.nd, sqd.inverse_bandwidths_internal.data()).asnumpy()

    assert inverse_bandwidths[0] > inverse_bandwidths[1] and inverse_bandwidths[0] > inverse_bandwidths[2]
    np.testing.assert_almost_equal(inverse_bandwidths[1], INVERSE_BANDWIDTHS_LOWER_BOUND)
    np.testing.assert_almost_equal(inverse_bandwidths[2], INVERSE_BANDWIDTHS_LOWER_BOUND)

    mu_train, _ = model.predict(x_train_mx_nd)[0]
    mu_test, _ = model.predict(x_test_mx_nd)[0]

    # back to np.array
    mu_train = mu_train.asnumpy()
    mu_test = mu_test.asnumpy()

    np.testing.assert_almost_equal(mu_train, y_train, decimal=2)
    # Fewer decimals imposed for the test points
    np.testing.assert_almost_equal(mu_test, y_test, decimal=1)
