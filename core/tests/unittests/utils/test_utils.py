import unittest

import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY, MULTICLASS, MULTICLASS_UPPER_LIMIT, REGRESSION
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils.utils import generate_train_test_split


class TestInferProblemType(unittest.TestCase):
    def test_infer_problem_type_empty(self):
        with self.assertRaises(ValueError):
            infer_problem_type(pd.Series([], dtype="float"))

    def test_infer_problem_type_nan(self):
        with self.assertRaises(ValueError):
            infer_problem_type(pd.Series([np.nan]))

    def test_infer_problem_type_inf(self):
        with self.assertRaises(ValueError):
            infer_problem_type(pd.Series([np.inf]))

    def test_infer_problem_type_ninf(self):
        with self.assertRaises(ValueError):
            infer_problem_type(pd.Series([-np.inf]))

    def test_infer_problem_type_binary(self):
        inferred_problem_type = infer_problem_type(pd.Series([-1, -1, 99, -1, -1, 99]))
        assert inferred_problem_type == BINARY

    def test_infer_problem_type_binary_with_nan(self):
        inferred_problem_type = infer_problem_type(pd.Series([-1, -1, 99, -1, -1, 99, np.nan]))
        assert inferred_problem_type == BINARY

    def test_infer_problem_type_str(self):
        inferred_problem_type = infer_problem_type(pd.Series(["a", "b", "c"], dtype=str))
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_category(self):
        inferred_problem_type = infer_problem_type(pd.Series(["a", "b", "c"], dtype="category"))
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_object(self):
        inferred_problem_type = infer_problem_type(pd.Series(["a", "b", "c"], dtype="object"))
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_multiclass_with_nan(self):
        inferred_problem_type = infer_problem_type(pd.Series(["a", "b", "c", np.nan]))
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_big_float_data_regression(self):
        big_float_regression_series = pd.Series(np.repeat(np.linspace(0.0, 1.0, MULTICLASS_UPPER_LIMIT + 1), 2))
        inferred_problem_type = infer_problem_type(big_float_regression_series)
        assert inferred_problem_type == REGRESSION

    def test_infer_problem_type_small_float_data_multiclass(self):
        big_float_multiclass_series = pd.Series(np.repeat([1.0, 2.0, 3.0], MULTICLASS_UPPER_LIMIT - 1))
        inferred_problem_type = infer_problem_type(big_float_multiclass_series)
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_small_float_data_regression(self):
        small_float_regression_series = pd.Series(np.linspace(0.0, 1.0, MULTICLASS_UPPER_LIMIT - 1))
        inferred_problem_type = infer_problem_type(small_float_regression_series)
        assert inferred_problem_type == REGRESSION

    def test_infer_problem_type_big_integer_data_regression(self):
        big_integer_regression_series = pd.Series(np.repeat(np.arange(MULTICLASS_UPPER_LIMIT + 1), 2), dtype=np.int64)
        inferred_problem_type = infer_problem_type(big_integer_regression_series)
        assert inferred_problem_type == REGRESSION

    def test_infer_problem_type_small_integer_data_multiclass(self):
        small_integer_multiclass_series = pd.Series(
            np.repeat(np.arange(3), MULTICLASS_UPPER_LIMIT - 1), dtype=np.int64
        )
        inferred_problem_type = infer_problem_type(small_integer_multiclass_series)
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_small_integer_data_regression(self):
        small_integer_regression_series = pd.Series(np.arange(MULTICLASS_UPPER_LIMIT - 1), dtype=np.int64)
        inferred_problem_type = infer_problem_type(small_integer_regression_series)
        assert inferred_problem_type == REGRESSION


def _assert_equals_generate_train_test_split(X, y, test_size, problem_type=None, test_equals=True, train_size=None):
    X_train, X_test, y_train, y_test = generate_train_test_split(
        X=X, y=y, problem_type=problem_type, test_size=test_size, train_size=train_size
    )
    assert len(X_train) == len(y_train)
    assert list(X_train.index) == list(y_train.index)
    assert len(X_test) == len(y_test)
    assert list(X_test.index) == list(y_test.index)
    assert len(X_train.index.intersection(X_test.index)) == 0  # No shared indices
    if test_equals:
        if isinstance(test_size, int):
            assert len(X_test) == test_size
        else:
            test_size_int = round(len(X) * test_size)
            assert len(X_test) == test_size_int
    else:
        if isinstance(test_size, int):
            assert len(X_test) <= test_size
        else:
            test_size_int = round(len(X) * test_size)
            assert len(X_test) <= test_size_int
    if train_size is not None:
        if isinstance(train_size, int):
            assert len(X_train) == train_size
        else:
            train_size_int = round(len(X) * train_size)
            assert len(X_train) == train_size_int
    if train_size is None or test_size is None:
        assert len(X_train) == len(X) - len(X_test)

    if train_size is None:
        if isinstance(test_size, int):
            train_size = len(X) - test_size
        else:
            train_size = 1.0 - test_size
        X_train_v2, X_test_v2, y_train_v2, y_test_v2 = generate_train_test_split(
            X=X, y=y, problem_type=problem_type, train_size=train_size
        )
        assert X_train.equals(X_train_v2)
        assert y_train.equals(y_train_v2)
        assert X_test.equals(X_test_v2)
        assert y_test.equals(y_test_v2)
        train_size = None

    if problem_type is not None and problem_type in ["binary", "multiclass"]:
        X_train_v3, X_test_v3, y_train_v3, y_test_v3 = generate_train_test_split(
            X=X, y=y, test_size=test_size, train_size=train_size, stratify=y
        )
        assert X_train.loc[X_train_v3.index].equals(X_train_v3)
        assert y_train.loc[y_train_v3.index].equals(y_train_v3)
        assert X_test.equals(X_test_v3.loc[X_test.index])
        assert y_test.equals(y_test_v3.loc[y_test.index])


def test_generate_train_test_split_edgecase():
    """
    Test rare edge-cases when data has many classes or very few samples when doing train test splits.
    """
    data = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data["label"] = [0, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]

    for test_size in range(1, 12):
        _assert_equals_generate_train_test_split(X=data, y=data["label"], test_size=test_size)
        _assert_equals_generate_train_test_split(X=data, y=data["label"], test_size=test_size / len(data))
        for problem_type in ["regression", "softclass", "quantile"]:
            """
            Normal Case: Regression should always work
            """
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size
            )
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data)
            )
        for train_size in range(1, len(data) - test_size + 1):
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], test_size=test_size, train_size=train_size
            )
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], test_size=test_size / len(data), train_size=train_size / len(data)
            )

    for test_size in range(1, 12):
        _assert_equals_generate_train_test_split(X=data, y=data["label"], test_size=test_size)
        _assert_equals_generate_train_test_split(X=data, y=data["label"], test_size=test_size / len(data))
        for problem_type in ["regression", "softclass", "quantile"]:
            """
            Normal Case: Regression should always work
            """
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size
            )
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data)
            )

    for problem_type in ["binary", "multiclass"]:
        for test_size in range(1, 6):
            """
            Edge-case: There are fewer test rows than classes
             This only works because of special try/except logic in `generate_train_test_split`.
            """
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size, test_equals=False
            )
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data), test_equals=False
            )

        for test_size in range(6, 7):
            """
            Normal Case
            """
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size, test_equals=False
            )
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data), test_equals=False
            )

        for test_size in range(7, 12):
            """
            Edge-case: There are fewer train rows than classes
            Error due to not enough training data to have at least one instance of every class in train.
            Note: Ideally this shouldn't raise an exception, but writing the logic to avoid the error is tricky and the scenario should never occur in practice.
            """
            with pytest.raises(ValueError):
                X_train, X_test, y_train, y_test = generate_train_test_split(
                    X=data, y=data["label"], problem_type=problem_type, test_size=test_size
                )

        # FIXME: Different for fractional inputs, because there is an inconsistency between float test_size and integer test_size in the internal logic.
        #  We should fix this eventually. Once it is fixed, this test will fail.
        for test_size in range(7, 10):
            _assert_equals_generate_train_test_split(
                X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data), test_equals=False
            )
        for test_size in range(10, 12):
            with pytest.raises(ValueError):
                X_train, X_test, y_train, y_test = generate_train_test_split(
                    X=data, y=data["label"], problem_type=problem_type, test_size=test_size / len(data)
                )
