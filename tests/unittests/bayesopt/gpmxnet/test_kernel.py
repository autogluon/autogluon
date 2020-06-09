import numpy as np
import mxnet as mx
import pytest

from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52, \
    FabolasKernelFunction, ProductKernelFunction
from autogluon.searcher.bayesopt.gpmxnet.kernel.base import SquaredDistance
from autogluon.searcher.bayesopt.gpmxnet.constants import DATA_TYPE
from autogluon.searcher.bayesopt.gpmxnet.gluon_blocks_helpers import \
    LogarithmScalarEncoding, PositiveScalarEncoding


def test_square_distance_no_ard_unit_bandwidth():
    X = mx.nd.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    # test default ard=False
    sqd = SquaredDistance(dimension=2)
    assert sqd.ARD == False
    sqd.collect_params().initialize()
    D = sqd(X, X).asnumpy()
    expected_D = np.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_almost_equal(expected_D, D)


def test_square_distance_no_ard_non_unit_bandwidth():
    X = mx.nd.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    sqd = SquaredDistance(dimension=2)
    assert sqd.ARD == False
    sqd.collect_params().initialize()
    sqd.encoding.set(sqd.inverse_bandwidths_internal, 1./np.sqrt(2.))
    D = sqd(X, X).asnumpy()
    expected_D = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_almost_equal(expected_D, D)


def test_square_distance_with_ard():
    X = mx.nd.array([[2., 1.], [1., 2.], [0., 1.]], dtype=DATA_TYPE)
    sqd = SquaredDistance(dimension=2, ARD=True)
    assert sqd.ARD == True
    sqd.collect_params().initialize()
    sqd.encoding.set(sqd.inverse_bandwidths_internal, [1. / np.sqrt(2.), 1.])
    D = sqd(X, X).asnumpy()
    expected_D = np.array([[0., 3./2., 2.], [3./2., 0., 3./2.], [2.0, 3./2., 0.]])
    np.testing.assert_almost_equal(expected_D, D)


mater52 = lambda squared_dist: (1. + np.sqrt(5. * squared_dist) + 5. / 3. * squared_dist) * np.exp(-np.sqrt(5. * squared_dist))
freeze_thaw = lambda u, alpha, beta: beta**alpha / (u + beta)**alpha


def test_matern52_unit_scale():
    X = mx.nd.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2)
    assert kernel.ARD == False
    kernel.collect_params().initialize()
    K = kernel(X,X).asnumpy()
    expected_K = np.array([[mater52(0.0), mater52(2.0)], [mater52(2.0), mater52(0.0)]])
    np.testing.assert_almost_equal(expected_K, K)


def test_matern52_non_unit_scale():
    X = mx.nd.array([[1, 0], [0, 1]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2)
    assert kernel.ARD == False
    kernel.collect_params().initialize()
    kernel.encoding.set(kernel.covariance_scale_internal, 0.5)
    K = kernel(X,X).asnumpy()
    expected_K = 0.5 * np.array([[mater52(0.0), mater52(2.0)], [mater52(2.0), mater52(0.0)]])
    np.testing.assert_almost_equal(expected_K, K)


def test_matern52_ard():
    X = mx.nd.array([[2., 1.], [1., 2.], [0., 1.]], dtype=DATA_TYPE)
    kernel = Matern52(dimension=2, ARD=True)
    kernel.collect_params().initialize()
    sqd = kernel.squared_distance
    assert kernel.ARD == True
    assert sqd.ARD == True
    sqd.encoding.set(sqd.inverse_bandwidths_internal, [1. / np.sqrt(2.), 1.])
    K = kernel(X,X).asnumpy()
    # expected_D is taken from previous test about squared distances
    expected_D = np.array([[0., 3. / 2., 2.], [3. / 2., 0., 3. / 2.], [2.0, 3. / 2., 0.]])
    expected_K = mater52(expected_D)
    np.testing.assert_almost_equal(expected_K, K)


def test_matern52_encoding():
    kernel = Matern52(dimension=2, ARD=True)
    assert isinstance(kernel.encoding, LogarithmScalarEncoding)
    assert isinstance(kernel.squared_distance.encoding, LogarithmScalarEncoding)
    assert kernel.encoding.dimension == 1
    assert kernel.squared_distance.encoding.dimension == 2
    kernel = Matern52(dimension=2, ARD=True, encoding_type="positive")
    assert isinstance(kernel.encoding, PositiveScalarEncoding)
    assert isinstance(kernel.squared_distance.encoding, PositiveScalarEncoding)
    assert kernel.encoding.dimension == 1
    assert kernel.squared_distance.encoding.dimension == 2


def test_fabolas_encoding():
    kernel = FabolasKernelFunction()
    assert isinstance(kernel.encoding_u12, LogarithmScalarEncoding)
    assert kernel.encoding_u12.dimension == 1

    kernel = FabolasKernelFunction(encoding_type="positive")
    assert isinstance(kernel.encoding_u12, PositiveScalarEncoding)
    assert kernel.encoding_u12.dimension == 1


def test_matern52_wrongshape():
    kernel = Matern52(dimension=3)
    kernel.collect_params().initialize()
    X1 = mx.nd.random_normal(0.0, 1.0, shape=(5, 2))
    with pytest.raises(mx.base.MXNetError):
        kmat = kernel(X1, X1)
    with pytest.raises(mx.base.MXNetError):
        kdiag = kernel.diagonal(mx.nd, X1)
    X2 = mx.nd.random_normal(0.0, 1.0, shape=(3, 3))
    with pytest.raises(mx.base.MXNetError):
        kmat = kernel(X2, X1)


def test_product_wrongshape():
    kernel1 = Matern52(dimension=2)
    kernel1.collect_params().initialize()

    #  A better way to do this is using the `pytest.mark.parametrize`
    #  decorator. Keeping it simple for now.
    kernels = [Matern52(dimension=1),
               FabolasKernelFunction()]

    for kernel2 in kernels:

        kernel2.collect_params().initialize()
        kernel = ProductKernelFunction(kernel1, kernel2)
        X1 = mx.nd.random_normal(0.0, 1.0, shape=(5, 4))
        with pytest.raises(mx.base.MXNetError):
            kmat = kernel(X1, X1)
        with pytest.raises(mx.base.MXNetError):
            kdiag = kernel.diagonal(mx.nd, X1)
        X2 = mx.nd.random_normal(0.0, 1.0, shape=(3, 3))
        with pytest.raises(mx.base.MXNetError):
            kmat = kernel(X2, X1)
        X1 = mx.nd.random_normal(0.0, 1.0, shape=(5, 2))
        with pytest.raises(mx.base.MXNetError):
            kmat = kernel(X1, X1)
