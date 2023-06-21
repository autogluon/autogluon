import unittest

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, MULTICLASS_UPPER_LIMIT, REGRESSION
from autogluon.core.utils import infer_problem_type


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
