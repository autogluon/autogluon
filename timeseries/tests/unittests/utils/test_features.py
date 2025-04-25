import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.features import (
    ConstantReplacementFeatureImportanceTransform,
    PermutationFeatureImportanceTransform,
    TimeSeriesFeatureGenerator,
)

from ..common import get_data_frame_with_covariates


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
    covariate_metadata = feat_generator.covariate_metadata

    assert covariate_metadata.known_covariates_cat == known_covariates_cat
    assert covariate_metadata.known_covariates_real == known_covariates_real
    assert covariate_metadata.past_covariates_cat == past_covariates_cat
    assert covariate_metadata.past_covariates_real == past_covariates_real
    assert covariate_metadata.static_features_cat == static_features_cat
    assert covariate_metadata.static_features_real == static_features_real


def test_when_transform_applied_then_numeric_features_are_converted_to_float32():
    data = get_data_frame_with_covariates(covariates_cat=["cov_cat"], static_features_cat=["static_cat"])

    data["int1"] = np.random.randint(0, 100, size=len(data), dtype=np.int32)
    data["int2"] = np.random.randint(0, 100, size=len(data), dtype=np.int64)
    data["float1"] = np.random.rand(len(data)).astype(np.float32)
    data["float2"] = np.random.rand(len(data)).astype(np.float64)
    data["bool"] = np.random.randint(0, 1, size=len(data), dtype="bool")

    data.static_features["int1_s"] = np.random.randint(0, 100, size=data.num_items, dtype=np.int32)
    data.static_features["int2_s"] = np.random.randint(0, 100, size=data.num_items, dtype=np.int64)
    data.static_features["float1_s"] = np.random.rand(data.num_items).astype(np.float32)
    data.static_features["float2_s"] = np.random.rand(data.num_items).astype(np.float64)
    data.static_features["bool_s"] = np.random.randint(0, 1, size=data.num_items, dtype="bool")

    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["int1", "float2"])
    data_transformed = feat_generator.fit_transform(data)
    for col in ["int1", "int2", "float1", "float2", "bool"]:
        assert isinstance(data_transformed[col].dtype, np.dtypes.Float32DType)
    for col in ["int1_s", "int2_s", "float1_s", "float2_s", "bool_s"]:
        assert isinstance(data_transformed.static_features[col].dtype, np.dtypes.Float32DType)


def test_when_data_transformed_then_original_data_is_not_modified():
    data = get_data_frame_with_covariates(
        covariates_cat=["known_cat", "past_cat"],
        covariates_real=["known_real", "past_real"],
        static_features_real=["static_real"],
        static_features_cat=["static_cat"],
    )
    data = data.astype({"known_real": "int64"})
    data.iloc[2, [0, 1, 2]] = float("nan")
    data.static_features.iloc[0, [0, 1]] = float("nan")
    data_orig = data.copy()
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known_cat", "known_real"])
    feat_generator.fit(data)
    data_transformed = feat_generator.transform(data)
    assert data.equals(data_orig)
    assert data.dtypes.equals(data_orig.dtypes)
    assert data.static_features.equals(data_orig.static_features)
    assert data.static_features.dtypes.equals(data_orig.static_features.dtypes)
    assert data_transformed.index is data.index


def test_when_known_covariates_transformed_then_original_known_covariates_not_modified():
    data_full = get_data_frame_with_covariates(covariates_cat=["known_cat"], covariates_real=["known_real"])
    known_covariates_names = ["known_cat", "known_real"]
    data, known_covariates = data_full.get_model_inputs_for_scoring(
        prediction_length=3, known_covariates_names=known_covariates_names
    )
    known_covariates.iloc[2, [0, 1]] = float("nan")
    known_covariates_orig = known_covariates.copy()
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    feat_generator.fit_transform(data)
    feat_generator.transform_future_known_covariates(known_covariates)
    assert known_covariates.equals(known_covariates_orig)
    assert known_covariates.dtypes.equals(known_covariates_orig.dtypes)


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


@pytest.mark.parametrize("use_fit_transform", [True, False])
@pytest.mark.parametrize("known_covariates_names", [[], ["real_1", "cat_1"], ["real_1", "real_2", "cat_1", "cat_2"]])
def test_when_covariates_contain_missing_values_then_they_are_filled_during_transform(
    known_covariates_names, use_fit_transform
):
    prediction_length = 5
    data_full = get_data_frame_with_covariates(covariates_cat=["cat_1", "cat_2"], covariates_real=["real_1", "real_2"])
    data_full.iloc[[0, 1, 8, 9, 10, 12, 15, -2, -1]] = float("nan")
    data_full.loc[data_full.item_ids[1]] = float("nan")

    data, known_covariates = data_full.get_model_inputs_for_scoring(prediction_length, known_covariates_names)
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)

    if use_fit_transform:
        data_transformed = feat_generator.fit_transform(data)
    else:
        feat_generator.fit(data)
        data_transformed = feat_generator.transform(data)
    assert not data_transformed[feat_generator.covariate_metadata.covariates].isna().any(axis=None)
    assert data_transformed["target"].isna().any()

    known_covariates_transformed = feat_generator.transform_future_known_covariates(known_covariates)
    if known_covariates_names == []:
        assert known_covariates_transformed is None
    else:
        assert not known_covariates_transformed[known_covariates_names].isna().any(axis=None)


@pytest.mark.parametrize("use_fit_transform", [True, False])
def test_when_static_features_contain_missing_values_then_they_are_filled_during_transform(use_fit_transform):
    data = get_data_frame_with_covariates(
        static_features_cat=["cat_1", "cat_2"], static_features_real=["real_1", "real_2"]
    )
    data.static_features.iloc[[0], [1, 2]] = float("nan")
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=[])

    if use_fit_transform:
        data_transformed = feat_generator.fit_transform(data)
    else:
        feat_generator.fit(data)
        data_transformed = feat_generator.transform(data)
    assert not data_transformed.static_features.isna().any(axis=None)


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 10, "B": 15},
        {"A": 10, "B": 10, "C": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_cat", [["known_cat_1"]])
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"], []])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("static_features_real", [["static_real_1", "static_real_2"], []])
@pytest.mark.parametrize(
    "importance_transform_class",
    [ConstantReplacementFeatureImportanceTransform, PermutationFeatureImportanceTransform],
)
def test_when_feature_importance_transforms_called_then_they_can_transform_all_features(
    item_id_to_length,
    known_covariates_cat,
    known_covariates_real,
    past_covariates_cat,
    past_covariates_real,
    static_features_cat,
    static_features_real,
    importance_transform_class,
):
    known_covariates_names = known_covariates_cat + known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_cat=known_covariates_cat + past_covariates_cat,
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_cat=static_features_cat,
        static_features_real=static_features_real,
    )
    feat_generator.fit(data)
    covariate_metadata = feat_generator.covariate_metadata

    transform = importance_transform_class(
        covariate_metadata=covariate_metadata,
        prediction_length=2,
    )

    for feature_name in covariate_metadata.all_features:
        transformed_data = transform.transform(data, feature_name)
        assert isinstance(transformed_data, TimeSeriesDataFrame)
        assert len(transformed_data) == len(data)


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 10, "B": 15},
        {"A": 10, "B": 10, "C": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"]])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("prediction_length", [2, 3])
@pytest.mark.parametrize(
    "importance_transform_class",
    [ConstantReplacementFeatureImportanceTransform, PermutationFeatureImportanceTransform],
)
def test_given_past_features_when_feature_importance_transforms_called_then_they_dont_change_forecast_horizon(
    item_id_to_length,
    known_covariates_real,
    past_covariates_cat,
    past_covariates_real,
    static_features_cat,
    prediction_length,
    importance_transform_class,
):
    known_covariates_names = known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_cat=past_covariates_cat,
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_cat=static_features_cat,
    )
    feat_generator.fit(data.copy())
    covariate_metadata = feat_generator.covariate_metadata

    transform = importance_transform_class(
        covariate_metadata=covariate_metadata,
        prediction_length=prediction_length,
        random_seed=None,
    )

    for feature_name in covariate_metadata.past_covariates:
        transformed_data = transform.transform(data, feature_name)
        assert all(
            x == y or np.isnan(x)
            for x, y in zip(
                data.slice_by_timestep(-prediction_length, None)[feature_name].values,
                transformed_data.slice_by_timestep(-prediction_length, None)[feature_name].values,
            )
        )


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 10, "B": 15},
        {"B": 10, "C": 10, "A": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"]])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("prediction_length", [2, 3])
def test_given_past_features_when_permutation_transform_called_then_shuffled_values_are_same(
    item_id_to_length,
    known_covariates_real,
    past_covariates_cat,
    past_covariates_real,
    static_features_cat,
    prediction_length,
):
    known_covariates_names = known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_cat=past_covariates_cat,
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_cat=static_features_cat,
    )
    feat_generator.fit(data.copy())
    covariate_metadata = feat_generator.covariate_metadata

    transform = PermutationFeatureImportanceTransform(
        covariate_metadata=covariate_metadata,
        prediction_length=prediction_length,
        random_seed=None,
    )

    for feature_name in covariate_metadata.covariates:
        slice_to_permute = (
            (None, -prediction_length) if feature_name in covariate_metadata.past_covariates else (None, None)
        )

        transformed_data = transform.transform(data, feature_name)
        assert all(
            (
                set(data.slice_by_timestep(*slice_to_permute).loc[item_id][feature_name])
                == set(transformed_data.slice_by_timestep(*slice_to_permute).loc[item_id][feature_name])
            )
            for item_id in data.item_ids
        )

    for feature_name in covariate_metadata.static_features:
        transformed_data = transform.transform(data, feature_name)
        assert set(data.static_features[feature_name]) == set(transformed_data.static_features[feature_name])


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 40, "B": 40},
        {"B": 10, "C": 10, "A": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"]])
@pytest.mark.parametrize("prediction_length", [2, 3])
@pytest.mark.parametrize("shuffle_type", ["naive", "itemwise"])
def test_given_past_features_when_permutation_transform_called_then_values_change_order(
    item_id_to_length,
    known_covariates_real,
    past_covariates_real,
    prediction_length,
    shuffle_type,
):
    known_covariates_names = known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_real=known_covariates_real + past_covariates_real,
    )
    feat_generator.fit(data.copy())
    covariate_metadata = feat_generator.covariate_metadata

    transform = PermutationFeatureImportanceTransform(
        covariate_metadata=covariate_metadata,
        prediction_length=prediction_length,
        shuffle_type=shuffle_type,
        random_seed=None,
    )

    for feature_name in covariate_metadata.covariates:
        slice_to_permute = (
            (None, -prediction_length) if feature_name in covariate_metadata.past_covariates else (None, None)
        )

        transformed_data = transform.transform(data, feature_name)
        assert not np.array_equal(
            data.slice_by_timestep(*slice_to_permute)[feature_name].values,
            transformed_data.slice_by_timestep(*slice_to_permute)[feature_name].values,
        )


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 40, "B": 40},
        {"B": 10, "C": 10, "A": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"]])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1", "static_cat_2"]])
@pytest.mark.parametrize("shuffle_type", ["naive", "itemwise"])
def test_given_fixed_seed_when_permutation_transform_called_then_shuffle_indices_are_same(
    item_id_to_length,
    known_covariates_real,
    static_features_cat,
    shuffle_type,
):
    known_covariates_names = known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_real=known_covariates_real,
        static_features_cat=static_features_cat,
    )
    feat_generator.fit(data.copy())
    covariate_metadata = feat_generator.covariate_metadata

    transform = PermutationFeatureImportanceTransform(
        covariate_metadata=covariate_metadata,
        prediction_length=2,
        shuffle_type=shuffle_type,
        random_seed=1234,
    )

    for feature_name in covariate_metadata.covariates:
        transformed_data_1 = transform.transform(data, feature_name)
        transformed_data_2 = transform.transform(data, feature_name)
        assert np.array_equal(
            transformed_data_1[feature_name].values,
            transformed_data_2[feature_name].values,
        )


@pytest.mark.parametrize(
    "item_id_to_length",
    [
        {"A": 40, "B": 40},
        {"B": 10, "C": 10, "A": 10},
    ],
)
@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"]])
@pytest.mark.parametrize("past_covariates_cat", [["past_cat_1", "past_cat_2"], []])
@pytest.mark.parametrize("static_features_cat", [["static_cat_1"], []])
@pytest.mark.parametrize("prediction_length", [2, 3])
@pytest.mark.parametrize("real_value_aggregation", ["mean", "median"])
def test_given_past_features_when_constant_transform_called_then_values_all_equal(
    item_id_to_length,
    known_covariates_real,
    past_covariates_real,
    past_covariates_cat,
    static_features_cat,
    prediction_length,
    real_value_aggregation,
):
    known_covariates_names = known_covariates_real
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        item_id_to_length=item_id_to_length,
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_cat=static_features_cat,
        covariates_cat=past_covariates_cat,
    )
    feat_generator.fit(data.copy())
    covariate_metadata = feat_generator.covariate_metadata

    transform = ConstantReplacementFeatureImportanceTransform(
        covariate_metadata=covariate_metadata,
        prediction_length=prediction_length,
        real_value_aggregation=real_value_aggregation,
    )

    for feature_name in covariate_metadata.covariates:
        slice_to_permute = (
            (None, -prediction_length) if feature_name in covariate_metadata.past_covariates else (None, None)
        )

        transformed_data = transform.transform(data, feature_name)
        assert all(
            (len(set(transformed_data.slice_by_timestep(*slice_to_permute).loc[item_id][feature_name])) == 1)
            for item_id in data.item_ids
        )

    for feature_name in covariate_metadata.static_features:
        transformed_data = transform.transform(data, feature_name)
        assert len(set(transformed_data.static_features[feature_name])) == 1


def test_if_categorical_feature_has_all_nan_values_then_feature_generator_works():
    data = get_data_frame_with_covariates(static_features_cat=["static"], covariates_cat=["past", "known"])
    data[["past", "known"]] = float("nan")
    data = data.astype({"past": "category", "known": "category"})
    data.static_features["static"] = float("nan")
    data.static_features = data.static_features.astype({"static": "category"})
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known"])
    transformed_data = feat_generator.fit_transform(data)
    assert "past" in transformed_data.columns
    assert "known" in transformed_data.columns
    assert "static" in transformed_data.static_features.columns
