from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas._testing import assert_series_equal, assert_frame_equal

from autogluon.common.features.types import S_TEXT_SPECIAL, R_FLOAT, R_DATETIME
from autogluon.tabular.models.widedeep_nn.preprocessing_utils import ContinuousNormalizer, MissingFiller, CategoricalFeaturesFilter


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

def test_target_scaler():
    pass  # TODO