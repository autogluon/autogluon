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
        ("int", ("bool", "sparse")): [
            "_ohe_0",
            "_ohe_1",
            "_ohe_2",
            "_ohe_3",
            "_ohe_4",
            "_ohe_5",
            "_ohe_6",
            "_ohe_7",
            "_ohe_8",
            "_ohe_9",
            "_ohe_10",
        ]
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
    assert output_data["_ohe_1"].dtype.subtype == np.uint8
    assert list(output_data["_ohe_1"].values) == expected_output_data_int_0_val


def test_one_hot_encoder_feature_generator_advanced(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()

    generator = OneHotEncoderFeatureGenerator(max_levels=3, sparse=False, dtype=np.uint16)

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("int", ()): ["int"],
    }
    # TODO: improve readability of names when max_levels is specified
    expected_feature_metadata_full = {
        ("int", ("bool",)): ["_ohe_0", "_ohe_1", "_ohe_2", "_ohe_3", "_ohe_4", "_ohe_5", "_ohe_6", "_ohe_7"]
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
    assert len(output_data.columns) == 8
    assert output_data["_ohe_0"].dtype == np.uint16
    assert list(output_data["_ohe_0"].values) == expected_output_data_int_0_val
