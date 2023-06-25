import numpy as np
import pandas as pd

from autogluon.features.generators import IsNanFeatureGenerator


def test_isnan_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    expected_output_data = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ],
        columns=[
            "__nan__.int_bool",
            "__nan__.int",
            "__nan__.float",
            "__nan__.obj",
            "__nan__.cat",
            "__nan__.datetime",
            "__nan__.text",
            "__nan__.datetime_as_object",
        ],
        dtype=np.uint8,
    )

    generator = IsNanFeatureGenerator()

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    expected_feature_metadata_full = {
        ("int", ("bool",)): [
            "__nan__.int_bool",
            "__nan__.int",
            "__nan__.float",
            "__nan__.obj",
            "__nan__.cat",
            "__nan__.datetime",
            "__nan__.text",
            "__nan__.datetime_as_object",
        ]
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data.equals(output_data)
