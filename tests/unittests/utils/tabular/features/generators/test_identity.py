
from autogluon.utils.tabular.features.generators import IdentityFeatureGenerator


def test_identity_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = IdentityFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('float', ()): ['float'],
        ('int', ()): ['int']
    }

    expected_feature_metadata_full = expected_feature_metadata_in_full

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert input_data[output_data.columns].equals(output_data)
