
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
        'datetime.year',
        'datetime.month',
        'datetime.day',
        'datetime.hour',
        'datetime.minute',
        'datetime.second',
        'datetime.dayofweek',
        'datetime.dayofyear',
        'datetime.quarter',
        'datetime.is_month_end',
        'datetime.is_leap_year',
        'datetime_as_object',
        'datetime_as_object.year',
        'datetime_as_object.month',
        'datetime_as_object.day',
        'datetime_as_object.hour',
        'datetime_as_object.minute',
        'datetime_as_object.second',
        'datetime_as_object.dayofweek',
        'datetime_as_object.dayofyear',
        'datetime_as_object.quarter',
        'datetime_as_object.is_month_end',
        'datetime_as_object.is_leap_year'
    ]}

    expected_output_data_feat_datetime = [
        1533140820000000000,
        1301322000000000000,
        1301322000000000000,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1301322000000000000,
        1301322000000000000
    ]

    expected_output_data_feat_datetime_year = [
        2018,
        2011,  # blank and nan values are set to the mean of good values = 2011
        2011,
        2018,
        2018,
        1800,
        2200,
        2011,  # 2700 and 1000 are out of range for a pandas datetime so they are set to the mean
        2011   # see limits at https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.max.html
    ]

    expected_output_data_feat_datetime_hour = [
        16,
        14, # blank, nan and bad are set to the average = 14
        14,
        15,
        15,
        0,
        23,
        14,
        14
    ]
    expected_output_data_feat_datetime_is_month_end = [
        0,
        0, # blank, nan and bad are set to the average = 0
        0,
        0,
        0,
        0,
        1,
        0,
        0
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
    assert expected_output_data_feat_datetime_year == list(output_data['datetime.year'].values)
    assert expected_output_data_feat_datetime_hour == list(output_data['datetime.hour'].values)
    assert expected_output_data_feat_datetime_is_month_end == list(output_data['datetime.is_month_end'].values)
    # Given we confirmed year, hour and is_month_end are working, 
    # adding tests for month/day/minute/second/etc is overkill.
