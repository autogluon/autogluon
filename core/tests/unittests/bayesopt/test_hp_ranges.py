# TODO: This code tests HyperparameterRanges and XYZScaling.
# If the latter code is removed, this test can go as well.

from collections import Counter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from pytest import approx

from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRangeContinuous, HyperparameterRangeInteger, \
    HyperparameterRangeCategorical, HyperparameterRanges_Impl
from autogluon.core.searcher.bayesopt.datatypes.scaling import \
    LinearScaling, LogScaling, ReverseLogScaling


@pytest.mark.parametrize('lower,upper,external_hp,internal_ndarray,scaling', [
    (0.0, 8.0, 0.0, 0.0, LinearScaling()),
    (0.0, 8.0, 8.0, 1.0, LinearScaling()),
    (0.0, 8.0, 2.0, 0.25, LinearScaling()),
    (100.2, 100.6, 100.4, 0.5, LinearScaling()),
    (-2.0, 8.0, 0.0, 0.2, LinearScaling()),
    (-11.0, -1.0, -10.0, 0.1, LinearScaling()),
    (1.0, 8.0, 1.0, 0.0, LogScaling()),
    (1.0, 8.0, 8.0, 1.0, LogScaling()),
    (1.0, 10000.0, 10.0, 0.25, LogScaling()),
    (1.0, 10000.0, 100.0, 0.5, LogScaling()),
    (1.0, 10000.0, 1000.0, 0.75, LogScaling()),
    (0.001, 0.1, 0.01, 0.5, LogScaling()),
    (0.1, 100, 1.0, 1.0/3, LogScaling()),
])
def test_continuous_to_and_from_ndarray(lower, upper, external_hp, internal_ndarray, scaling):
    hp_range = HyperparameterRangeContinuous('hp', lower, upper, scaling)
    assert_allclose(
        hp_range.to_ndarray(external_hp), np.array([internal_ndarray])
    )
    assert_allclose(
        hp_range.from_ndarray(np.array([internal_ndarray])), external_hp
    )


@pytest.mark.parametrize('lower,upper,external_hp,internal_ndarray,scaling', [
    (1, 10001, 5001, 0.5, LinearScaling()),
    (-10, 10, 0, 0.5, LinearScaling()),
])
def test_integer_to_and_from_ndarray(lower, upper, external_hp, internal_ndarray, scaling):
    hp_range = HyperparameterRangeInteger('hp', lower, upper, scaling)
    assert_allclose(
        hp_range.to_ndarray(external_hp), np.array([internal_ndarray])
    )
    assert_allclose(
        hp_range.from_ndarray(np.array([internal_ndarray])), external_hp
    )


@pytest.mark.parametrize('choices,external_hp,internal_ndarray', [
    (('a', 'b'), 'a', [1.0, 0.0]),
    (('a', 'b'), 'b', [0.0, 1.0]),
    (('a', 'b', 'c', 'd'), 'c', [0.0, 0.0, 1.0, 0.0]),
])
def test_categorical_to_and_from_ndarray(choices, external_hp, internal_ndarray):
    hp_range = HyperparameterRangeCategorical('hp', choices)
    assert_allclose(
        hp_range.to_ndarray(external_hp), np.array(internal_ndarray)
    )
    assert hp_range.from_ndarray(np.array(internal_ndarray)) == external_hp


# Going to internal representation and back should give back the original value
@pytest.mark.parametrize('lower,upper,scaling', [
    (0.0, 8.0, LinearScaling()),
    (0.01, 0.1, LinearScaling()),
    (-10.0, -5.1, LinearScaling()),
    (-1000000000000000.0, 100000000000000000.0, LinearScaling()),
    (10.0, 10000000000.0, LogScaling()),
    (-1000.0, 100.0, LinearScaling()),
    (1.0, 1000.0, LogScaling()),
    (10.0, 15.0, LogScaling()),
    (0.1, 20.0, LogScaling()),
])
def test_continuous_to_ndarray_and_back(lower, upper, scaling):
    # checks the lower bound upper bound and 5 random values
    _test_continuous_to_ndarray_and_back(lower, upper, lower, scaling)
    _test_continuous_to_ndarray_and_back(lower, upper, upper, scaling)
    rnd = np.random.RandomState(0)
    for random_hp in rnd.uniform(lower, upper, size=10):
        _test_continuous_to_ndarray_and_back(lower, upper, random_hp, scaling)
    _test_continuous_to_ndarray_and_back(lower, upper, lower, scaling)
    _test_continuous_to_ndarray_and_back(lower, upper, upper, scaling)


# helper for the previous test
def _test_continuous_to_ndarray_and_back(lower, upper, external_hp, scaling):
    hp_range = HyperparameterRangeContinuous('hp', lower, upper, scaling)
    assert hp_range.from_ndarray(hp_range.to_ndarray(external_hp)) == approx(external_hp)


@pytest.mark.parametrize('lower,upper,scaling', [
    (0, 8, LinearScaling()),
    (1, 20, LinearScaling()),
    (-10, -5, LinearScaling()),
    (-1000000000000000, 100000000000000000, LinearScaling()),
    (10, 10000000000, LogScaling()),
    (-1000, 100, LinearScaling()),
    (1, 1000, LogScaling()),
    (10, 15, LogScaling()),
])
def test_integer_to_ndarray_and_back(lower, upper, scaling):
    # checks the lower bound upper bound and 5 random values
    _test_integer_to_ndarray_and_back(lower, upper, lower, scaling)
    _test_integer_to_ndarray_and_back(lower, upper, upper, scaling)
    rnd = np.random.RandomState(0)
    for random_hp in rnd.randint(lower+1, upper, size=15):
        _test_integer_to_ndarray_and_back(lower, upper, int(random_hp), scaling)
    _test_integer_to_ndarray_and_back(lower, upper, lower, scaling)
    _test_integer_to_ndarray_and_back(lower, upper, upper, scaling)


# helper for the previous test
def _test_integer_to_ndarray_and_back(lower, upper, external_hp, scaling):
    hp_range = HyperparameterRangeInteger('hp', lower, upper, scaling)
    assert hp_range.from_ndarray(hp_range.to_ndarray(external_hp)) == approx(external_hp)


# this is more of a functional test testing of HP conversion and scaling
# it generates random candidates and checks the distribution is correct
# and also that they can be transformed to internal representation and back while still obtaining
# the same value
def test_distribution_of_random_candidates():
    random_state = np.random.RandomState(0)
    hp_ranges = HyperparameterRanges_Impl(
        HyperparameterRangeContinuous('0', 1.0, 1000.0, scaling=LinearScaling()),
        HyperparameterRangeContinuous('1', 1.0, 1000.0, scaling=LogScaling()),
        HyperparameterRangeContinuous('2', 0.9, 0.9999, scaling=ReverseLogScaling()),
        HyperparameterRangeInteger('3', 1, 1000, scaling=LinearScaling()),
        HyperparameterRangeInteger('4', 1, 1000, scaling=LogScaling()),
        HyperparameterRangeCategorical('5', ('a', 'b', 'c')),
    )
    num_random_candidates = 600
    random_candidates = [hp_ranges.random_candidate(random_state) for _ in range(num_random_candidates)]

    # check converting back gets to the same candidate
    for cand in random_candidates[2:]:
        ndarray_candidate = hp_ranges.to_ndarray(cand)
        converted_back = hp_ranges.from_ndarray(ndarray_candidate)
        for hp, hp_converted_back in zip(cand, converted_back):
            if isinstance(hp, str):
                assert hp == hp_converted_back
            else:
                assert_almost_equal(hp, hp_converted_back)

    hps0, hps1, hps2, hps3, hps4, hps5 = zip(*random_candidates)
    assert 200 < np.percentile(hps0, 25) < 300
    assert 450 < np.percentile(hps0, 50) < 550
    assert 700 < np.percentile(hps0, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps1, 25) < 10
    assert 20 < np.percentile(hps1, 50) < 40
    assert 100 < np.percentile(hps1, 75) < 200

    # reverse log
    assert 0.9 < np.percentile(hps2, 25) < 0.99
    assert 0.99 < np.percentile(hps2, 50) < 0.999
    assert 0.999 < np.percentile(hps2, 75) < 0.9999

    # integer
    assert 200 < np.percentile(hps3, 25) < 300
    assert 450 < np.percentile(hps3, 50) < 550
    assert 700 < np.percentile(hps3, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps4, 25) < 10
    assert 20 < np.percentile(hps4, 50) < 40
    assert 100 < np.percentile(hps4, 75) < 200

    counter = Counter(hps5)
    assert len(counter) == 3

    assert 150 < counter['a'] < 250  # should be about 200
    assert 150 < counter['b'] < 250  # should be about 200
    assert 150 < counter['c'] < 250  # should be about 200


@pytest.mark.parametrize('lower,upper,scaling,constructor', [
    (0, 8, LinearScaling(), HyperparameterRangeInteger),
    (1, 20, LinearScaling(), HyperparameterRangeInteger),
    (-10, -5, LinearScaling(), HyperparameterRangeInteger),
    (-1000000000000000, 100000000000000000, LinearScaling(), HyperparameterRangeInteger),
    (10, 10000000000, LogScaling(), HyperparameterRangeInteger),
    (-1000, 100, LinearScaling(), HyperparameterRangeInteger),
    (1, 1000, LogScaling(), HyperparameterRangeInteger),
    (10, 15, LogScaling(), HyperparameterRangeInteger),
    
    (0.0, 8.0, LinearScaling(), HyperparameterRangeContinuous),
    (0.01, 0.1, LinearScaling(), HyperparameterRangeContinuous),
    (-10.0, -5.1, LinearScaling(), HyperparameterRangeContinuous),
    (-1000000000000000.0, 100000000000000000.0, LinearScaling(), HyperparameterRangeContinuous),
    (-1000.0, 100.0, LinearScaling(), HyperparameterRangeContinuous),
    (10.0, 10000000000.0, LogScaling(), HyperparameterRangeContinuous),
    (1.0, 1000.0, LogScaling(), HyperparameterRangeContinuous),
    (10.0, 15.0, LogScaling(), HyperparameterRangeContinuous),
    (0.1, 20.0, LogScaling(), HyperparameterRangeContinuous),
])
def test_from_zero_one_extremes(lower, upper, scaling, constructor):
    hp_range = constructor('hp', lower, upper, scaling)
    assert hp_range.from_zero_one(0.0) == approx(lower)
    assert hp_range.from_zero_one(1.0) == approx(upper)
