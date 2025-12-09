import numpy as np
import pandas as pd
import pytest

from autogluon.features.generators import OOFTargetEncodingFeatureGenerator

can_transform_on_train = False

def _assert_all_between(series: pd.Series, low: float, high: float):
    assert ((series >= low) & (series <= high)).all()


def test_oof_target_encoding_regression(generator_helper, data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()
    # Simple regression target
    y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="regression",
        keep_original=False,
        n_splits=3,
        alpha=10.0,
        random_state=0,
    )

    # When
    X_out = generator_helper.fit_transform_assert(
        input_data=X,
        generator=generator,
        y=y,
        can_transform_on_train=can_transform_on_train,
    )

    # Then
    assert generator.is_fit()
    # Categorical columns detected
    assert generator.cols_ == ["obj", "cat"]
    assert set(generator.passthrough_cols_) == {"int", "float", "datetime"}

    expected_encoded_cols = ["obj__te", "cat__te"]
    expected_columns = generator.passthrough_cols_ + expected_encoded_cols
    assert list(X_out.columns) == expected_columns

    # Original categorical features removed
    assert "obj" not in X_out.columns
    assert "cat" not in X_out.columns

    # Encoded columns are float and non-null
    for col in expected_encoded_cols:
        assert np.issubdtype(X_out[col].dtype, np.floating)
        assert not X_out[col].isna().any()

    # Encoded values between min/max of target
    y_min, y_max = float(y.min()), float(y.max())
    for col in expected_encoded_cols:
        _assert_all_between(X_out[col], y_min, y_max)

    # FeatureMetadata: in = original features, out = transformed features
    fm_in = generator.feature_metadata_in
    fm_out = generator.feature_metadata
    assert set(fm_in.get_features()) == set(X.columns)
    assert set(fm_out.get_features()) == set(X_out.columns)
    for col in expected_encoded_cols:
        assert fm_out.get_feature_type_raw(col) == "float"


def test_oof_target_encoding_binary(generator_helper, data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0])

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="binary",
        keep_original=False,
        n_splits=3,
        alpha=5.0,
        random_state=1,
    )

    # When
    X_out = generator_helper.fit_transform_assert(
        input_data=X,
        generator=generator,
        y=y,
        can_transform_on_train=can_transform_on_train,
    )

    # Then
    assert generator.is_fit()
    assert generator.cols_ == ["obj", "cat"]
    assert set(generator.passthrough_cols_) == {"int", "float", "datetime"}

    expected_encoded_cols = ["obj__te", "cat__te"]
    expected_columns = generator.passthrough_cols_ + expected_encoded_cols
    assert list(X_out.columns) == expected_columns

    # classes_ must have exactly 2 classes
    assert hasattr(generator, "classes_")
    assert len(generator.classes_) == 2
    assert set(generator.classes_) == {0, 1}

    # Encoded values are probabilities in [0, 1]
    for col in expected_encoded_cols:
        _assert_all_between(X_out[col], 0.0, 1.0)


def test_oof_target_encoding_binary_raises_if_more_than_two_classes(data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()
    # 3 unique classes but target_type="binary"
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="binary",
        keep_original=False,
        n_splits=3,
        alpha=5.0,
        random_state=1,
    )

    # Then
    with pytest.raises(AssertionError):
        generator.fit_transform(X, y=y)


def test_oof_target_encoding_multiclass(generator_helper, data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()
    # 3-class labels
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="multiclass",
        keep_original=False,
        n_splits=3,
        alpha=5.0,
        random_state=2,
    )

    # When
    X_out = generator_helper.fit_transform_assert(
        input_data=X,
        generator=generator,
        y=y,
        can_transform_on_train=can_transform_on_train,
    )

    # Then
    assert generator.is_fit()
    assert generator.cols_ == ["obj", "cat"]
    assert set(generator.passthrough_cols_) == {"int", "float", "datetime"}

    n_classes = len(np.unique(y))
    expected_encoded_cols = [
        "obj__te_class0",
        "obj__te_class1",
        "obj__te_class2",
        "cat__te_class0",
        "cat__te_class1",
        "cat__te_class2",
    ]
    assert len(expected_encoded_cols) == 2 * n_classes

    expected_columns = generator.passthrough_cols_ + expected_encoded_cols
    assert list(X_out.columns) == expected_columns

    # classes_ must match unique classes
    assert hasattr(generator, "classes_")
    assert set(generator.classes_) == set(np.unique(y))

    # Each encoded column must be in [0, 1]
    for col in expected_encoded_cols:
        _assert_all_between(X_out[col], 0.0, 1.0)


def test_oof_target_encoding_keep_original_true(generator_helper, data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()
    y = pd.Series(range(len(X)))  # regression

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="regression",
        keep_original=True,
        n_splits=3,
        alpha=10.0,
        random_state=0,
    )

    # When
    X_out = generator_helper.fit_transform_assert(
        input_data=X,
        generator=generator,
        y=y,
        can_transform_on_train=can_transform_on_train,
    )

    # Then
    assert generator.is_fit()
    # All original columns must still be present
    for col in X.columns:
        assert col in X_out.columns

    expected_encoded_cols = ["obj__te", "cat__te"]
    for col in expected_encoded_cols:
        assert col in X_out.columns

    # Total columns = original + encoded
    assert len(X_out.columns) == len(X.columns) + len(expected_encoded_cols)

    # FeatureMetadata: original raw types preserved for original features
    fm_in = generator.feature_metadata_in
    fm_out = generator.feature_metadata
    for col in X.columns:
        assert fm_out.get_feature_type_raw(col) == fm_in.get_feature_type_raw(col)
    for col in expected_encoded_cols:
        assert fm_out.get_feature_type_raw(col) == "float"


def test_oof_target_encoding_no_categorical_columns(generator_helper):
    # Given: X has only numeric
    X = pd.DataFrame({"int": [1, 2, 3, 4, 5]})
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="regression",
        keep_original=False,
        n_splits=2,
        alpha=1.0,
        random_state=0,
    )

    # When
    X_out = generator_helper.fit_transform_assert(
        input_data=X,
        generator=generator,
        y=y,
        can_transform_on_train=can_transform_on_train,
    )

    # Then
    assert generator.is_fit()
    assert generator.cols_ == []
    assert generator.encodings_ == {}
    # Output should be identical to input
    assert X_out.equals(X)


def test_oof_target_encoding_unseen_and_nan_categories():
    # Given: simple single-categorical, binary
    X_train = pd.DataFrame({"feat": ["a", "b", "a", "c"]})
    y = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame(
        {
            "feat": ["a", "d", np.nan, "b"],  # "d" unseen, np.nan missing
        }
    )

    generator = OOFTargetEncodingFeatureGenerator(
        target_type="binary",
        keep_original=False,
        n_splits=2,
        alpha=5.0,
        random_state=0,
    )

    # When
    _ = generator.fit_transform(X_train, y=y)
    X_test_enc = generator.transform(X_test)

    # Then
    encoded_col = "feat__te"
    assert encoded_col in X_test_enc.columns

    # All encodings are probabilities
    _assert_all_between(X_test_enc[encoded_col], 0.0, 1.0)

    # Unseen category "d" and NaN should map to same global-mean encoding
    val_unseen = X_test_enc.loc[1, encoded_col]
    val_nan = X_test_enc.loc[2, encoded_col]
    assert val_unseen == pytest.approx(val_nan)


def test_oof_target_encoding_estimate_no_of_new_features(data_helper):
    # Given
    X = data_helper.generate_multi_feature_standard()  # has "obj" (object) and "cat" (category)
    num_cat_cols = 2
    num_classes = 3

    gen_reg = OOFTargetEncodingFeatureGenerator(target_type="regression")
    gen_bin = OOFTargetEncodingFeatureGenerator(target_type="binary")
    gen_multi = OOFTargetEncodingFeatureGenerator(target_type="multiclass")

    # When
    n_reg, cols_reg = gen_reg.estimate_no_of_new_features(X, num_classes=num_classes)
    n_bin, cols_bin = gen_bin.estimate_no_of_new_features(X, num_classes=num_classes)
    n_multi, cols_multi = gen_multi.estimate_no_of_new_features(X, num_classes=num_classes)

    # Then
    # Regression & binary: one encoded feature per categorical column
    assert n_reg == num_cat_cols
    assert n_bin == num_cat_cols
    assert cols_reg == ["obj", "cat"]
    assert cols_bin == ["obj", "cat"]

    # Multiclass: num_cat_cols * num_classes
    assert n_multi == num_cat_cols * num_classes
    assert cols_multi == ["obj", "cat"]
