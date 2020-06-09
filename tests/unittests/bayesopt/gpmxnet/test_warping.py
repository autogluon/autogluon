import numpy as np
import mxnet as mx

from autogluon.searcher.bayesopt.gpmxnet.warping import OneDimensionalWarping, \
    Warping, WarpedKernel
from autogluon.searcher.bayesopt.gpmxnet.constants import DATA_TYPE, \
    NUMERICAL_JITTER
from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52
from autogluon.searcher.bayesopt.gpmxnet.gp_regression import \
    GaussianProcessRegression
from autogluon.searcher.bayesopt.gpmxnet.gluon_blocks_helpers import \
    LogarithmScalarEncoding, PositiveScalarEncoding


def test_warping_encoding():
    input_range = (0., 2.)
    warping = OneDimensionalWarping(input_range)
    assert isinstance(warping.encoding, LogarithmScalarEncoding)
    assert warping.encoding.dimension == 2
    warping = OneDimensionalWarping(input_range, encoding_type="positive")
    assert isinstance(warping.encoding, PositiveScalarEncoding)


def test_warping_default_parameters():
    x = mx.nd.array([0., 1., 2.], dtype=DATA_TYPE)
    input_range = (0., 2.)
    warping = OneDimensionalWarping(input_range)
    warping.collect_params().initialize()

    warping_parameters = warping.encoding.get(mx.nd, warping.warping_internal.data())

    np.testing.assert_almost_equal(warping_parameters.asnumpy(), np.ones(2))
    np.testing.assert_almost_equal(warping(x).asnumpy(), np.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER]))


def test_warping_with_arbitrary_parameters():
    x = mx.nd.array([0., 1., 2.], dtype=DATA_TYPE)
    input_range = (0., 2.)

    warping = OneDimensionalWarping(input_range)
    warping.collect_params().initialize()

    warping.encoding.set(warping.warping_internal, [2., 0.5])
    warping_parameters = warping.encoding.get(mx.nd, warping.warping_internal.data())

    np.testing.assert_almost_equal(warping_parameters.asnumpy(), [2., 0.5])

    # In that case (with parameters [2., 0.5]), the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1. - np.sqrt(1. - x*x)

    np.testing.assert_almost_equal(warping(x).asnumpy(), expected_warping(np.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER])))


def test_warping_with_multidimension_and_arbitrary_parameters():
    X = mx.nd.array([[0., 1., 0.], [1.,2.,1.], [2., 0., 2.]], dtype=DATA_TYPE)

    dimension=3

    # We transform only the columns {0,2} of the 3-dimensional data X
    input_range = (0., 2.)
    warping = Warping(index_to_range={0:input_range, 2:input_range}, dimension=dimension)

    assert len(warping.transformations) == dimension

    warping.collect_params().initialize()

    # We change the warping parameters of the first dimension only
    w0 = warping.transformations[0]
    w0.encoding.set(w0.warping_internal, [2., 0.5])

    w2 = warping.transformations[2]
    w2_parameters = w2.encoding.get(mx.nd, w2.warping_internal.data())

    # The parameters of w2 should be the default ones (as there was no set operations)
    np.testing.assert_almost_equal(w2_parameters.asnumpy(), np.ones(2))

    # print(warping(X).asnumpy())
    # for name, p  in warping.collect_params().items():
    #     print(name, p.data().asnumpy())

    # With parameters [2., 0.5], the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1. - np.sqrt(1. - x*x)

    expected_column0 = expected_warping(np.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER])).reshape((-1,1))
    expected_column1 = np.array([1., 2., 0.]).reshape((-1,1))
    expected_column2 = np.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER]).reshape((-1,1))

    np.testing.assert_almost_equal(warping(X).asnumpy(), np.hstack([expected_column0, expected_column1, expected_column2]))


def test_gp_regression_with_warping():

    def f(x):
        return np.sin(3*np.log(x))

    np.random.seed(7)

    L, U = -5., 12.
    input_range = (2.**L, 2.**U)

    x_train = np.sort(2.**np.random.uniform(L, U, 250))
    x_test = np.sort(2.**np.random.uniform(L, U, 500))
    y_train = f(x_train)
    y_test = f(x_test)

    # to mx.nd
    y_train_mx_nd = mx.nd.array(y_train)
    x_train_mx_nd = mx.nd.array(x_train)
    x_test_mx_nd = mx.nd.array(x_test)

    kernels = [
        Matern52(dimension=1),
        WarpedKernel(
            kernel=Matern52(dimension=1),
            warping=Warping(dimension=1, index_to_range={0: input_range})
        )
    ]

    models = [GaussianProcessRegression(kernel=k, random_seed=0) for k in kernels]
    train_errors, test_errors = [], []

    for model in models:

        model.fit(x_train_mx_nd, y_train_mx_nd)

        mu_train, var_train = model.predict(x_train_mx_nd)[0]
        mu_test, var_test = model.predict(x_test_mx_nd)[0]

        # back to np.array
        mu_train = mu_train.asnumpy()
        mu_test = mu_test.asnumpy()
        # var_train = var_train.asnumpy()
        # var_test = var_test.asnumpy()

        train_errors.append(np.mean(np.abs((mu_train - y_train))))
        test_errors.append(np.mean(np.abs((mu_test - y_test))))

    # The two models have similar performance on training points
    np.testing.assert_almost_equal(train_errors[0], train_errors[1], decimal=4)

    # As expected, the model with warping largely outperforms the model without
    assert test_errors[1] < 0.1 * test_errors[0]

    # If we wish to plot things
    # import matplotlib.pyplot as plt
    # plt.plot(x_train, y_train, "r-")
    # plt.plot(x_train, mu_train, "b--")
    #
    # plt.plot(x_test, y_test, "y-")
    # plt.plot(x_test, mu_test, "m--")

    # plt.fill_between(x_train,
    #                  mu_train - np.sqrt(var_train),
    #                  mu_train + np.sqrt(var_train),
    #                  alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
    #
    # plt.fill_between(x_test,
    #                  mu_test - np.sqrt(var_test),
    #                  mu_test + np.sqrt(var_test),
    #                  alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
    #
    # plt.show()
