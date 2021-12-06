
import numpy as np

from autogluon.features.generators import LabelEncoderFeatureGenerator


def test_label_encoder_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()

    generator = LabelEncoderFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('category', ()): ['cat'],
    }
    expected_feature_metadata_full = {
        ('int', ()): ['cat']
    }

    expected_output_data_cat_val = [0, 1, 0, 3, 3, 3, 2, -1, -1]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Therefore
    assert output_data['cat'].dtype == np.int8
    assert list(output_data['cat'].values) == expected_output_data_cat_val
