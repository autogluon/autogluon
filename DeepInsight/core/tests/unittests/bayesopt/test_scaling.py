# TODO: This code tests XYZScaling, which is only needed for HyperparameterRanges.
# If the latter code is removed, this test can go as well.

import pytest
from numpy.testing import assert_almost_equal

from autogluon.core.searcher.bayesopt.datatypes.scaling import \
    LinearScaling, LogScaling, ReverseLogScaling


@pytest.mark.parametrize('value, expected, scaling', [
    (0.0, 0.0, LinearScaling()),
    (0.5, 0.5, LinearScaling()),
    (5.0, 5.0, LinearScaling()),
    (-5.0, -5.0, LinearScaling()),
    (0.5, -0.69314718055994529, LogScaling()),
    (5.0, 1.6094379124341003, LogScaling()),
    (0.0, 0.0, ReverseLogScaling()),
    (0.5, 0.69314718055994529, ReverseLogScaling())
])
def test_to_internal(value, expected, scaling):
    assert_almost_equal(expected, scaling.to_internal(value))


@pytest.mark.parametrize('value, expected, scaling', [
    (0.0001, -9.210340371976182, LogScaling()),
    (0.000001, -13.815510557964274, LogScaling()),
    (0.0001, 0.00010000500033334732, ReverseLogScaling()),
    (0.000001, 1.000000500029089e-06, ReverseLogScaling()),
    (0.9999, 9.210340371976294, ReverseLogScaling()),
    (0.999999, 13.815510557935518, ReverseLogScaling())
])
def test_close_to_bounds_values(value, expected, scaling):
    assert_almost_equal(expected, scaling.to_internal(value))


@pytest.mark.parametrize('value, scaling', [
    (-5.0, LogScaling()),
    (0.0, LogScaling()),
    (5.0, ReverseLogScaling()),
    (-5.0, ReverseLogScaling())
])
def test_invalid_values(value, scaling):
    with pytest.raises(AssertionError):
        scaling.to_internal(value)


@pytest.mark.parametrize('value, expected, scaling', [
    (0.0, 0.0, LinearScaling()),
    (0.5, 0.5, LinearScaling()),
    (5.0, 5.0, LinearScaling()),
    (-5.0, -5.0, LinearScaling()),
    (0.0, 1.0, LogScaling()),
    (0.5, 1.6487212707001282, LogScaling()),
    (5.0, 148.4131591025766, LogScaling()),
    (-5.0, 0.006737946999085467, LogScaling()),
    (0.0, 0.0, ReverseLogScaling()),
    (0.5, 0.39346934028736658, ReverseLogScaling()),
    (5.0, 0.99326205300091452, ReverseLogScaling()),
    (-5.0, -147.4131591025766, ReverseLogScaling())
])
def test_from_internal(value, expected, scaling):
    assert_almost_equal(expected, scaling.from_internal(value))
