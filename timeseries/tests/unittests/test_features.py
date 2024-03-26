import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from .common import get_data_frame_with_variable_lengths

ITEM_ID_TO_LENGTH = {1: 10, 5: 20, 2: 30}


def get_data_frame_with_covariates(
    item_id_to_length: Dict[str, int] = ITEM_ID_TO_LENGTH,
    target: str = "target",
    covariates_cat: Optional[List[str]] = None,
    covariates_real: Optional[List[str]] = None,
    static_features_cat: Optional[List[str]] = None,
    static_features_real: Optional[List[str]] = None,
):
    data = get_data_frame_with_variable_lengths(item_id_to_length)
    data.rename(columns={"target": target}, inplace=True)
    if covariates_cat:
        for col in covariates_cat:
            data[col] = np.random.choice(["foo", "bar", "baz"], size=len(data))
    if covariates_real:
        for col in covariates_real:
            data[col] = np.random.rand(len(data))
    if static_features_cat or static_features_real:
        static_dict = {}
        if static_features_cat:
            for col in static_features_cat:
                static_dict[col] = np.random.choice(["cat", "dog", "cow"], size=data.num_items)
        if static_features_real:
            for col in static_features_real:
                static_dict[col] = np.random.rand(data.num_items)
        data.static_features = pd.DataFrame(static_dict, index=data.item_ids)
    return data


@pytest.mark.parametrize("known_covariates_cat", [["known_cat_1"], []])
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"], []])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("static_features_real", [["static_real_1", "static_real_2"], []])
def test_when_covariates_present_in_data_then_they_are_included_in_metadata(
    known_covariates_cat,
    known_covariates_real,
    past_covariates_cat,
    past_covariates_real,
    static_features_cat,
    static_features_real,
):
    known_covariates_names = known_covariates_cat + known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        covariates_cat=known_covariates_cat + past_covariates_cat,
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_cat=static_features_cat,
        static_features_real=static_features_real,
    )
    feat_generator.fit(data)
    metadata = feat_generator.covariate_metadata

    assert metadata.known_covariates_cat == known_covariates_cat
    assert metadata.known_covariates_real == known_covariates_real
    assert metadata.past_covariates_cat == past_covariates_cat
    assert metadata.past_covariates_real == past_covariates_real
    assert metadata.static_features_cat == static_features_cat
    assert metadata.static_features_real == static_features_real


@pytest.mark.skipif(sys.version_info <= (3, 8), reason="np.dtypes not available in Python 3.8")
def test_when_transform_applied_then_numeric_features_are_converted_to_float32():
    data = get_data_frame_with_covariates(covariates_cat=["cov_cat"], static_features_cat=["static_cat"])

    data["int1"] = np.random.randint(0, 100, size=len(data), dtype=np.int32)
    data["int2"] = np.random.randint(0, 100, size=len(data), dtype=np.int64)
    data["float1"] = np.random.rand(len(data)).astype(np.float32)
    data["float2"] = np.random.rand(len(data)).astype(np.float64)

    data.static_features["int1_s"] = np.random.randint(0, 100, size=data.num_items, dtype=np.int32)
    data.static_features["int2_s"] = np.random.randint(0, 100, size=data.num_items, dtype=np.int64)
    data.static_features["float1_s"] = np.random.rand(data.num_items).astype(np.float32)
    data.static_features["float2_s"] = np.random.rand(data.num_items).astype(np.float64)

    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["int1", "float2"])
    data_transformed = feat_generator.fit_transform(data)
    for col in ["int1", "int2", "float1", "float2"]:
        assert isinstance(data_transformed[col].dtype, np.dtypes.Float32DType)
    for col in ["int1_s", "int2_s", "float1_s", "float2_s"]:
        assert isinstance(data_transformed.static_features[col].dtype, np.dtypes.Float32DType)


def test_when_duplicate_columns_provided_during_fit_then_they_are_removed():
    data = get_data_frame_with_covariates(covariates_cat=["known", "past"], static_features_real=["static"])
    data["known_duplicate"] = data["known"]
    data["past_duplicate"] = data["past"]
    data.static_features["static_duplicate"] = data.static_features["static"]
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known", "known_duplicate"])
    data_transformed = feat_generator.fit_transform(data)
    assert "known_duplicate" not in data_transformed.columns
    assert "past_duplicate" not in data_transformed.columns
    assert "static_duplicate" not in data_transformed.static_features.columns


def test_when_duplicate_columns_provided_during_fit_then_they_can_be_omitted_during_transform():
    data = get_data_frame_with_covariates(covariates_cat=["known", "past"], static_features_real=["static"])
    data_with_duplicates = data.copy()
    data_with_duplicates["known_duplicate"] = data_with_duplicates["known"]
    data_with_duplicates["past_duplicate"] = data_with_duplicates["past"]
    data_with_duplicates.static_features["static_duplicate"] = data.static_features["static"]
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known", "known_duplicate"])
    feat_generator.fit(data_with_duplicates)
    feat_generator.transform(data)


def test_when_known_covariates_have_non_numeric_non_cat_dtypes_then_they_can_be_omitted_at_predict_time():
    data = get_data_frame_with_covariates(covariates_real=["past"])
    data["known"] = pd.date_range(start="2020-01-01", freq="D", periods=len(data))
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known"])
    feat_generator.fit(data)
    feat_generator.transform_future_known_covariates(None)


def test_when_bool_columns_provided_then_they_are_converted_to_cat():
    data = get_data_frame_with_covariates(covariates_cat=["known", "past"], static_features_real=["static"])
    data["known_bool"] = np.random.randint(0, 1, size=len(data)).astype(bool)
    data["past_bool"] = np.random.randint(0, 1, size=len(data)).astype(bool)
    data.static_features["static_bool"] = np.random.randint(0, 1, size=data.num_items).astype(bool)
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known", "known_bool"])

    data_transformed = feat_generator.fit_transform(data)
    assert isinstance(data_transformed["known_bool"].dtype, pd.CategoricalDtype)
    assert isinstance(data_transformed["past_bool"].dtype, pd.CategoricalDtype)
    assert isinstance(data_transformed.static_features["static_bool"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("known_covariates_names", [[], ["real_1", "cat_1"], ["real_1", "real_2", "cat_1", "cat_2"]])
def test_when_covariates_contain_missing_values_then_they_are_filled_during_transform(known_covariates_names):
    prediction_length = 5
    data_full = get_data_frame_with_covariates(covariates_cat=["cat_1", "cat_2"], covariates_real=["real_1", "real_2"])
    data_full.iloc[[0, 1, 8, 9, 10, 12, 15, -2, -1]] = float("nan")
    data_full.loc[data_full.item_ids[1]] = float("nan")

    data, known_covariates = data_full.get_model_inputs_for_scoring(prediction_length, known_covariates_names)
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)

    data_transformed = feat_generator.fit_transform(data)
    assert not data_transformed[feat_generator.covariate_metadata.covariates].isna().any(axis=None)

    known_covariates_transformed = feat_generator.transform_future_known_covariates(known_covariates)
    if known_covariates_names == []:
        assert known_covariates_transformed is None
    else:
        assert not known_covariates_transformed[known_covariates_names].isna().any(axis=None)


def test_when_static_features_contain_missing_values_then_they_are_filled_during_transform():
    data = get_data_frame_with_covariates(
        static_features_cat=["cat_1", "cat_2"], static_features_real=["real_1", "real_2"]
    )
    data.static_features.iloc[[0], [1, 2]] = float("nan")
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=[])

    data_transformed = feat_generator.fit_transform(data)
    assert not data_transformed.static_features.isna().any(axis=None)
