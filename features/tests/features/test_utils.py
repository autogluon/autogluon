import numpy as np
import pytest

from autogluon.features.utils import get_smallest_valid_dtype_int


@pytest.mark.parametrize(
    "min_val, max_val, expected_dtype",
    [
        # Purely non-negative values → unsigned branch
        (0, 0, np.uint8),
        (0, 10, np.uint8),
        (0, 255, np.uint8),  # max exactly at uint8 max
        (0, 256, np.uint16),  # just above uint8, needs uint16
        (0, np.iinfo(np.uint16).max, np.uint16),
        (0, np.iinfo(np.uint16).max + 1, np.uint32),
        # Large but still in uint32
        (0, np.iinfo(np.uint32).max, np.uint32),
    ],
)
def test_get_smallest_valid_dtype_int_unsigned(min_val, max_val, expected_dtype):
    result = get_smallest_valid_dtype_int(min_val=min_val, max_val=max_val)
    assert np.dtype(result) == np.dtype(expected_dtype)


@pytest.mark.parametrize(
    "min_val, max_val, expected_dtype",
    [
        # Negative min → signed branch
        (-1, 0, np.int8),
        (-1, 1, np.int8),
        (-127, 127, np.int8),
        (-128, 127, np.int8),  # full int8 range
        (-129, 127, np.int16),  # just below int8.min
        (np.iinfo(np.int16).min, np.iinfo(np.int16).max, np.int16),
        (np.iinfo(np.int16).min - 1, 0, np.int32),
        # Mixed negative/positive within int32
        (-10_000, 10_000, np.int16),
        (-100_000, 100_000, np.int32),
    ],
)
def test_get_smallest_valid_dtype_int_signed(min_val, max_val, expected_dtype):
    result = get_smallest_valid_dtype_int(min_val=min_val, max_val=max_val)
    assert np.dtype(result) == np.dtype(expected_dtype)


def test_min_zero_prefers_unsigned():
    """When min_val == 0, the function should use unsigned dtypes."""
    # Could fit in int8, but branch is unsigned so we expect uint8
    result = get_smallest_valid_dtype_int(min_val=0, max_val=np.iinfo(np.int8).max)
    assert np.dtype(result) == np.dtype(np.uint8)


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        # Out of range for all signed dtypes
        (np.iinfo(np.int64).min - 1, 0),
        (np.iinfo(np.int64).min - 10, np.iinfo(np.int64).max),
    ],
)
def test_get_smallest_valid_dtype_int_raises_for_signed_overflow(min_val, max_val):
    with pytest.raises(ValueError):
        get_smallest_valid_dtype_int(min_val=min_val, max_val=max_val)


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        # Out of range for all unsigned dtypes
        (0, np.iinfo(np.uint64).max + 1),
        (5, np.iinfo(np.uint64).max + 12345),
    ],
)
def test_get_smallest_valid_dtype_int_raises_for_unsigned_overflow(min_val, max_val):
    with pytest.raises(ValueError):
        get_smallest_valid_dtype_int(min_val=min_val, max_val=max_val)
