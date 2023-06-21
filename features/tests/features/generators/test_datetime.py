from autogluon.features.generators import DatetimeFeatureGenerator


def test_datetime_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator_1 = DatetimeFeatureGenerator()
    generator_2 = DatetimeFeatureGenerator(features=["hour"])

    expected_feature_metadata_in_full = {
        ("datetime", ()): ["datetime"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
    }

    expected_feature_metadata_full_1 = {
        ("int", ("datetime_as_int",)): [
            "datetime",
            "datetime.year",
            "datetime.month",
            "datetime.day",
            "datetime.dayofweek",
            "datetime_as_object",
            "datetime_as_object.year",
            "datetime_as_object.month",
            "datetime_as_object.day",
            "datetime_as_object.dayofweek",
        ]
    }

    expected_feature_metadata_full_2 = {
        ("int", ("datetime_as_int",)): [
            "datetime",
            "datetime.hour",
            "datetime_as_object",
            "datetime_as_object.hour",
        ]
    }

    expected_output_data_feat_datetime = [
        1533140820000000000,
        1301322000000000000,
        1301322000000000000,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1301322000000000000,
        1301322000000000000,
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
        2011,  # see limits at https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.max.html
    ]

    expected_output_data_feat_datetime_hour = [16, 14, 14, 15, 15, 0, 23, 14, 14]

    # When
    output_data_1 = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator_1,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full_1,
    )

    assert list(output_data_1["datetime"].values) == list(output_data_1["datetime_as_object"].values)
    assert expected_output_data_feat_datetime == list(output_data_1["datetime"].values)
    assert expected_output_data_feat_datetime_year == list(output_data_1["datetime.year"].values)

    output_data_2 = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator_2,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full_2,
    )

    assert list(output_data_2["datetime"].values) == list(output_data_2["datetime_as_object"].values)
    assert expected_output_data_feat_datetime == list(output_data_2["datetime"].values)
    assert expected_output_data_feat_datetime_hour == list(output_data_2["datetime.hour"].values)


# This covers the nightmare input scenario for a datetime column:
# multiple formats, multiple NaN's of different types, multiple time zones (including no time zone), all as strings.
# This is just about as bad as it could get. If we work here, we should work with practically anything.
def test_datetime_feature_generator_advanced(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_datetime_as_object_feature_advanced().to_frame(name="datetime_as_object")

    generator = DatetimeFeatureGenerator()

    expected_feature_metadata_in_full = {
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
    }

    expected_feature_metadata_full = {
        ("int", ("datetime_as_int",)): [
            "datetime_as_object",
            "datetime_as_object.year",
            "datetime_as_object.month",
            "datetime_as_object.day",
            "datetime_as_object.dayofweek",
        ]
    }

    expected_output_data_feat_datetime = [
        1533140820000000000,
        1600067037034500096,
        1600067037034500096,
        1628543228659000000,
        1628543295541000000,
        1610040803938000000,
    ]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_datetime == list(output_data["datetime_as_object"].values)
