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
            infer_problem_type(pd.Series([np.NINF]))

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
        small_integer_multiclass_series = pd.Series(np.repeat(np.arange(3), MULTICLASS_UPPER_LIMIT - 1), dtype=np.int64)
        inferred_problem_type = infer_problem_type(small_integer_multiclass_series)
        assert inferred_problem_type == MULTICLASS

    def test_infer_problem_type_small_integer_data_regression(self):
        small_integer_regression_series = pd.Series(np.arange(MULTICLASS_UPPER_LIMIT - 1), dtype=np.int64)
        inferred_problem_type = infer_problem_type(small_integer_regression_series)
        assert inferred_problem_type == REGRESSION


def test_generate_train_test_split_edgecase():
    """
    Test rare edge-cases when data has many classes or very few samples when doing train test splits.
    """
    data = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data["label"] = [0, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]

    for test_size in range(1, 12):
        """
        Normal Case: Regression should always work
        """
        X_train, X_test, y_train, y_test = generate_train_test_split(X=data, y=data["label"], problem_type="regression", test_size=test_size)
        assert len(X_train) == len(y_train)
        assert list(X_train.index) == list(y_train.index)
        assert len(X_test) == len(y_test)
        assert list(X_test.index) == list(y_test.index)
        assert len(X_test) == test_size
        assert len(X_train) == len(data) - test_size

    for test_size in range(1, 6):
        """
        Edge-case: There are fewer test rows than classes
         This only works because of special try/except logic in `generate_train_test_split`.
        """
        X_train, X_test, y_train, y_test = generate_train_test_split(X=data, y=data["label"], problem_type="multiclass", test_size=test_size)
        assert len(X_train) == len(y_train)
        assert list(X_train.index) == list(y_train.index)
        assert len(X_test) == len(y_test)
        assert list(X_test.index) == list(y_test.index)
        assert len(X_test) <= test_size
        assert len(X_train) == len(data) - len(X_test)

    for test_size in range(6, 7):
        """
        Normal Case
        """
        X_train, X_test, y_train, y_test = generate_train_test_split(X=data, y=data["label"], problem_type="multiclass", test_size=test_size)
        assert len(X_train) == len(y_train)
        assert list(X_train.index) == list(y_train.index)
        assert len(X_test) == len(y_test)
        assert list(X_test.index) == list(y_test.index)
        assert len(X_test) <= test_size
        assert len(X_train) == len(data) - len(X_test)

    for test_size in range(7, 12):
        """
        Edge-case: There are fewer train rows than classes
        Error due to not enough training data to have at least one instance of every class in train.
        Note: Ideally this shouldn't raise an exception, but writing the logic to avoid the error is tricky and the scenario should never occur in practice.
        """
        with pytest.raises(ValueError):
            X_train, X_test, y_train, y_test = generate_train_test_split(X=data, y=data["label"], problem_type="multiclass", test_size=test_size)
