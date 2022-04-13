from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas._testing import assert_series_equal, assert_frame_equal
from sklearn.preprocessing import RobustScaler

from autogluon.common.features.types import S_TEXT_SPECIAL, R_FLOAT, R_DATETIME
from autogluon.core.constants import REGRESSION, QUANTILE, BINARY
from autogluon.tabular.models.widedeep_nn.preprocessing_utils import ContinuousNormalizer, MissingFiller, CategoricalFeaturesFilter, TargetScaler


def test_continuous_normalizer():
    cont_columns = ['a', 'b']
    normalizer = ContinuousNormalizer(cont_columns)

    x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, np.NaN, 3, 4, 5],
        'c': [1, 2, 3, 4, 5]
    })

    normalizer.fit(x)
    x_transformed = normalizer.transform(x.copy())
    mean, std = normalizer.stats

    assert x['c'].equals(x_transformed['c'])

    for c in ['a', 'b']:
        series = x[c]
        assert series.mean() == mean[c]
        assert series.std() == std[c]
        assert not x[c].equals(x_transformed[c])
        assert_series_equal((series - series.mean()) / series.std(), x_transformed[c], check_names=False)


def test_missing_filler():
    feature_metadata = Mock()
    feature_metadata.get_features = Mock(return_value=['a', 'b'])

    filler = MissingFiller(feature_metadata)

    x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, np.NaN, 3, 4, 5],
        'c': [1, 2, 3, 4, 5]
    })

    x_transformed = filler.fit_transform(x)
    assert {'a': 3.0, 'b': 3.25} == filler.columns_fills

    x_expected = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1.0, 3.25, 3.0, 4.0, 5.0],
        'c': [1, 2, 3, 4, 5]
    })

    assert_frame_equal(x_expected, x_transformed)

    feature_metadata.get_features.assert_called_with(valid_raw_types=[R_FLOAT, R_DATETIME], invalid_special_types=[S_TEXT_SPECIAL])


def test_missing_filler_noop():
    feature_metadata = Mock()
    feature_metadata.get_features = Mock(return_value=[])

    filler = MissingFiller(feature_metadata)

    x = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

    x_transformed = filler.fit_transform(x)
    assert {} == filler.columns_fills

    assert_frame_equal(x, x_transformed)

    feature_metadata.get_features.assert_called_with(valid_raw_types=[R_FLOAT, R_DATETIME], invalid_special_types=[S_TEXT_SPECIAL])


def test_categorical_features_filter():
    x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 1, 2, 2, 3],
        'c': [1, 1, 1, 2, 2],
        'd': [1, 2, 3, 4, 5]
    }).astype(str)

    cat_columns = ['a', 'b', 'c']
    cat_columns_filtered = CategoricalFeaturesFilter.filter(x, cat_columns, max_unique_categorical_values=3)
    assert cat_columns_filtered == ['b', 'c']

    cat_columns_filtered = CategoricalFeaturesFilter.filter(x, cat_columns, max_unique_categorical_values=2)
    assert cat_columns_filtered == ['c']


def test_categorical_features_filter_no_cats():
    x = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    cat_columns = []
    cat_columns_filtered = CategoricalFeaturesFilter.filter(x, cat_columns, max_unique_categorical_values=3)
    assert cat_columns_filtered == []


def test_categorical_features_filter_numerics():
    x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 1, 2, 2, 3],
        'c': [1, 1, 1, 2, 2],
        'd': [1, 2, 3, 4, 5],
        'e': ['1', '1', '2', '2', '3'],
    })
    cat_columns = ['a', 'b', 'c']
    cat_columns_filtered = CategoricalFeaturesFilter.filter(x, cat_columns, max_unique_categorical_values=2)
    assert cat_columns_filtered == ['a', 'b', 'c']


def test_target_scaler_regression():
    scaler = TargetScaler(REGRESSION)
    y = pd.Series([100, 200, 3000, 200])
    y_val = pd.Series([100, 200, 300, 200])
    y_tx, y_val_tx = scaler.fit_transform(y, y_val)

    # Forward fit-transform
    y_tx_exp = pd.Series([-0.631340, -0.549876, 1.731093, -0.549876])
    y_val_tx_exp = pd.Series([-0.631340, -0.549876, -0.468413, -0.549876])
    __verify_forward_scaler_transform(y_tx, y_tx_exp, y_val_tx, y_val_tx_exp)

    # Inverse transform
    y_tx_inv = scaler.inverse_transform(y_tx_exp.values)
    assert np.allclose(y, y_tx_inv, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_target_scaler_quantile():
    scaler, y, y_val, y_tx, y_val_tx = __scaler_setup(problem_type=QUANTILE)

    # Forward fit-transform
    y_tx_exp = pd.Series([0.0, 0.03448276, 1.0, 0.03448276])
    y_val_tx_exp = pd.Series([0.0, 0.03448276, 0.06896552, 0.03448276])
    __verify_forward_scaler_transform(y_tx, y_tx_exp, y_val_tx, y_val_tx_exp)

    # Inverse transform
    y_tx_inv = scaler.inverse_transform(y_tx_exp.values)
    assert np.allclose(y, y_tx_inv, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_target_scaler_other():
    scaler, y, y_val, y_tx, y_val_tx = __scaler_setup(problem_type=BINARY)

    # Forward fit-transform
    assert np.allclose(y, y_tx, rtol=1e-05, atol=1e-08, equal_nan=False)
    assert np.allclose(y_val, y_val_tx, rtol=1e-05, atol=1e-08, equal_nan=False)

    # Inverse transform
    y_tx_inv = scaler.inverse_transform(y.values)
    assert np.allclose(y, y_tx_inv, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_target_scaler_custom_scaler():
    scaler, y, y_val, y_tx, y_val_tx = __scaler_setup(problem_type=REGRESSION, y_scaler=RobustScaler())

    # Forward fit-transform
    y_tx_exp = pd.Series([-0.137931, 0.0, 3.862069, 0.0])
    y_val_tx_exp = pd.Series([-0.137931, 0.0, 0.137931, 0.0])
    __verify_forward_scaler_transform(y_tx, y_tx_exp, y_val_tx, y_val_tx_exp)

    # Inverse transform
    y_tx_inv = scaler.inverse_transform(y_tx_exp.values)
    assert np.allclose(y, y_tx_inv, rtol=1e-05, atol=1e-08, equal_nan=False)


def __verify_forward_scaler_transform(y_tx, y_tx_exp, y_val_tx, y_val_tx_exp):
    assert np.allclose(y_tx_exp, y_tx, rtol=1e-05, atol=1e-08, equal_nan=False)
    assert np.allclose(y_val_tx_exp, y_val_tx, rtol=1e-05, atol=1e-08, equal_nan=False)


def __scaler_setup(**scaler_args):
    scaler = TargetScaler(**scaler_args)
    y = pd.Series([100, 200, 3000, 200])
    y_val = pd.Series([100, 200, 300, 200])
    y_tx, y_val_tx = scaler.fit_transform(y, y_val)
    return scaler, y, y_val, y_tx, y_val_tx
