
from autogluon.features.generators import DatetimeFeatureGenerator


def test_datetime_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = DatetimeFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('datetime', ()): ['datetime'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
    }

    expected_feature_metadata_full = {('int', ('datetime_as_int',)): [
        'datetime',
        'datetime_as_object',
    ]}

    expected_output_data_feat_datetime = [
        1533140820000000000,
        -9223372036854775808,
        -9223372036854775808,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1597475520000000000,
        1608257520000000000
    ]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert list(output_data['datetime'].values) == list(output_data['datetime_as_object'].values)
    assert expected_output_data_feat_datetime == list(output_data['datetime'].values)
