from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import IdentityFeatureGenerator


def test_identity_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = IdentityFeatureGenerator()

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    expected_feature_metadata_full = expected_feature_metadata_in_full

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert input_data.equals(output_data)


def test_identity_feature_generator_int_float(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT]))

    expected_feature_metadata_in_full = {("float", ()): ["float"], ("int", ()): ["int_bool", "int"]}

    expected_feature_metadata_full = expected_feature_metadata_in_full

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert input_data[output_data.columns].equals(output_data)


def test_identity_feature_generator_int_float_with_banned_features(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = IdentityFeatureGenerator(
        infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT]),
        banned_feature_special_types=["my_banned_feature_type"],
    )

    expected_feature_metadata_in_full = {("int", ()): ["int_bool"], ("int", ("my_valid_feature_type",)): ["int"]}

    expected_feature_metadata_full = {("int", ()): ["int_bool"], ("int", ("my_valid_feature_type",)): ["int"]}

    feature_metadata_in = FeatureMetadata.from_df(input_data)

    feature_metadata_in = feature_metadata_in.add_special_types(
        {
            "float": ["my_banned_feature_type"],
            "int": ["my_valid_feature_type"],
        }
    )

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        feature_metadata_in=feature_metadata_in,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert input_data[output_data.columns].equals(output_data)
