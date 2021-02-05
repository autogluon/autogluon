
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.core.features.types import R_INT, R_FLOAT


def test_identity_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = IdentityFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
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