import pytest

from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges_Impl, HyperparameterRangeInteger, \
    HyperparameterRangeContinuous, HyperparameterRangeCategorical
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.core.searcher.bayesopt.utils.duplicate_detector import \
    DuplicateDetectorEpsilon, DuplicateDetectorIdentical, \
    DuplicateDetectorNoDetection


hp_ranges = HyperparameterRanges_Impl(
    HyperparameterRangeInteger('hp1', 0, 1000000000, scaling=LinearScaling()),
    HyperparameterRangeContinuous('hp2', -10.0, 10.0, scaling=LinearScaling()),
    HyperparameterRangeCategorical('hp3', ('a', 'b', 'c')),
)

duplicate_detector_epsilon = DuplicateDetectorEpsilon(hp_ranges)


@pytest.mark.parametrize('existing, new, contained', [
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10000, 3.0, 'c'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.000001, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.000001, 'b'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (25, 1.0, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'a'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.0, 'b'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (19, 1.0, 'a'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0000001, 'a'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'c'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'b'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 1.0, 'b'), False),
])
def test_contains_epsilon(existing, new, contained):
    assert duplicate_detector_epsilon.contains(existing, new) == contained


@pytest.mark.parametrize('existing, new, contained', [
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10000, 3.0, 'c'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.000001, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.000001, 'b'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (25, 1.0, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'a'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.0, 'b'), True),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (19, 1.0, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0000001, 'a'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'c'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'b'), False),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 1.0, 'b'), False),
])
def test_contains_identical(existing, new, contained):
    assert DuplicateDetectorIdentical().contains(existing, new) == contained


@pytest.mark.parametrize('existing, new', [
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10000, 3.0, 'c')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.000001, 'a')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.000001, 'b')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (25, 1.0, 'a')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'a')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 2.0, 'b')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (19, 1.0, 'a')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0000001, 'a')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'c')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (10, 1.0, 'b')),
    ({(10, 1.0, 'a'), (20, 2.0, 'b')}, (20, 1.0, 'b')),
])
def test_contains_no_detection(existing, new):
    assert not DuplicateDetectorNoDetection().contains(existing, new)
