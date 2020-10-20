import numpy
import autograd.numpy as anp

from autogluon.core.searcher.bayesopt.gpautograd.warping import \
    OneDimensionalWarping, Warping
from autogluon.core.searcher.bayesopt.gpautograd.constants import DATA_TYPE, \
    NUMERICAL_JITTER
from autogluon.core.searcher.bayesopt.gpautograd.gluon_blocks_helpers import \
    LogarithmScalarEncoding, PositiveScalarEncoding


def test_warping_encoding():
    input_range = (0., 2.)
    warping = OneDimensionalWarping(input_range)
    assert isinstance(warping.encoding, LogarithmScalarEncoding)
    assert warping.encoding.dimension == 2
    warping = OneDimensionalWarping(input_range, encoding_type="positive")
    assert isinstance(warping.encoding, PositiveScalarEncoding)


def test_warping_default_parameters():
    x = anp.array([0., 1., 2.], dtype=DATA_TYPE)
    input_range = (0., 2.)
    warping = OneDimensionalWarping(input_range)
    warping.collect_params().initialize()
    
    warping_parameters = warping.encoding.get(warping.warping_internal.data())
    
    numpy.testing.assert_almost_equal(warping_parameters, anp.ones(2))
    numpy.testing.assert_almost_equal(warping(x), anp.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER]))


def test_warping_with_arbitrary_parameters():
    x = anp.array([0., 1., 2.], dtype=DATA_TYPE)
    input_range = (0., 2.)
    warping = OneDimensionalWarping(input_range)
    warping.collect_params().initialize()
    warping.encoding.set(warping.warping_internal, [2., 0.5])
    warping_parameters = warping.encoding.get(warping.warping_internal.data())
    numpy.testing.assert_almost_equal(warping_parameters, [2., 0.5])   
    # In that case (with parameters [2., 0.5]), the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1. - anp.sqrt(1. - x*x)
    numpy.testing.assert_almost_equal(warping(x), expected_warping(anp.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER])))
    

def test_warping_with_multidimension_and_arbitrary_parameters():
    X = anp.array([[0., 1., 0.], [1.,2.,1.], [2., 0., 2.]], dtype=DATA_TYPE)
    
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
    w2_parameters = w2.encoding.get(w2.warping_internal.data())
    
    # The parameters of w2 should be the default ones (as there was no set operations)
    numpy.testing.assert_almost_equal(w2_parameters, anp.ones(2))
    
    # With parameters [2., 0.5], the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1. - anp.sqrt(1. - x*x)
        
    expected_column0 = expected_warping(anp.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER])).reshape((-1,1))
    expected_column1 = anp.array([1., 2., 0.]).reshape((-1,1))
    expected_column2 = anp.array([NUMERICAL_JITTER, 0.5, 1.-NUMERICAL_JITTER]).reshape((-1,1))
    
    numpy.testing.assert_almost_equal(warping(X), anp.hstack([expected_column0, expected_column1, expected_column2]))
