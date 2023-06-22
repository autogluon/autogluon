import numpy as np

from autogluon.features.generators import OneHotEncoderFeatureGenerator


def test_one_hot_encoder_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()

    generator = OneHotEncoderFeatureGenerator()

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("int", ()): ["int"],
    }
    expected_feature_metadata_full = {
        ("int", ("bool", "sparse")): ["int_-8", "int_0", "int_2", "int_3", "int_5", "int_12", "cat_a", "cat_b", "cat_c", "cat_d", "cat_nan"]
    }

    expected_output_data_int_0_val = [0, 1, 0, 0, 0, 0, 0, 1, 0]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Therefore
    assert len(output_data.columns) == 11
    assert output_data["int_0"].dtype.subtype == np.uint8
    assert list(output_data["int_0"].values) == expected_output_data_int_0_val


def test_one_hot_encoder_feature_generator_advanced(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()

    generator = OneHotEncoderFeatureGenerator(max_levels=3, sparse=False, dtype=np.uint16)

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("int", ()): ["int"],
    }
    # TODO: improve readability of names when max_levels is specified
    expected_feature_metadata_full = {("int", ("bool",)): ["x0_0", "x0_13", "x0_2", "x0_5", "x1_13", "x1_a", "x1_c", "x1_d"]}

    expected_output_data_int_0_val = [0, 1, 0, 0, 0, 0, 0, 1, 0]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Therefore
    assert len(output_data.columns) == 8
    assert output_data["x0_0"].dtype == np.uint16
    assert list(output_data["x0_0"].values) == expected_output_data_int_0_val
