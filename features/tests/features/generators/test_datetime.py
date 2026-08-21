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


def test_datetime_feature_generator_non_range_index(generator_helper, data_helper):
    # Given
    # Same datetime values as test_datetime_feature_generator, but a non-contiguous index.
    # broken_idx is label-based; series.iloc[broken_idx] raises IndexError.
    # Existing tests stay on RangeIndex, so they do not catch this.
    input_data = data_helper.generate_datetime_feature().to_frame(name="datetime")
    input_data.index = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    generator = DatetimeFeatureGenerator()

    expected_feature_metadata_in_full = {
        ("datetime", ()): ["datetime"],
    }

    expected_feature_metadata_full = {
        ("int", ("datetime_as_int",)): [
            "datetime",
            "datetime.year",
            "datetime.month",
            "datetime.day",
            "datetime.dayofweek",
        ]
    }

    # Mean-fill of invalid rows is unchanged vs the RangeIndex test.
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

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_datetime == list(output_data["datetime"].values)
    assert expected_output_data_feat_datetime_year == list(output_data["datetime.year"].values)
    assert list(output_data.index) == [10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_datetime_feature_generator_transform_non_range_index(data_helper):
    # Fit on RangeIndex (default pipeline), then transform a held-out frame that
    # kept leftover labels — the bag/split crash path. .iloc[broken_idx] IndexErrors.
    train = data_helper.generate_datetime_feature().to_frame(name="datetime")
    generator = DatetimeFeatureGenerator()
    generator.fit_transform(train)

    held_out = train.copy()
    held_out.index = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    output = generator.transform(held_out)

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
        2011,
        2011,
        2018,
        2018,
        1800,
        2200,
        2011,
        2011,
    ]
    assert expected_output_data_feat_datetime == list(output["datetime"].values)
    assert expected_output_data_feat_datetime_year == list(output["datetime.year"].values)
    assert list(output.index) == [10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_datetime_feature_generator_permuted_index_fill_value(data_helper):
    """A permuted index must not change the fill value.

    The invalid rows used to be selected positionally from an index of *labels*. With labels beyond
    the row count that raises, which is loud; with labels inside it -- any permuted index, as
    `sort_values`, `groupby` or `sample` produce -- it silently picks the wrong rows, and the
    invalid rows then survive into the mean as `NaT.astype(np.int64)`, a large negative sentinel.
    The fill value came out decades early with no error.
    """
    import numpy as np
    import pandas as pd

    values = ["2020-01-01", "2020-02-01", "not-a-date", "2020-04-01"]
    permuted = pd.DataFrame({"datetime_as_object": values}, index=[3, 2, 1, 0])

    generator = DatetimeFeatureGenerator()
    generator.fit_transform(permuted)

    good = pd.to_datetime(pd.Series([v for v in values if v != "not-a-date"]), utc=True).astype(np.int64)
    assert generator._fillna_map["datetime_as_object"] == pd.to_datetime(int(good.mean()), utc=True)


def test_datetime_feature_generator_duplicate_index(data_helper):
    """A duplicated index is now handled rather than raising.

    The label-based assignment that filled the invalid rows raised
    `InvalidIndexError: Reindexing only valid with uniquely valued Index objects`; a boolean mask
    does not need a unique index.
    """
    import numpy as np
    import pandas as pd

    values = ["not-a-date", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"]
    duplicated = pd.DataFrame({"datetime_as_object": values}, index=[0, 1, 0, 1, 0, 1])

    generator = DatetimeFeatureGenerator()
    output = generator.fit_transform(duplicated)

    good = pd.to_datetime(pd.Series(values[1:]), utc=True).astype(np.int64)
    expected_fill = pd.to_datetime(int(good.mean()), utc=True)
    assert generator._fillna_map["datetime_as_object"] == expected_fill
    # the invalid row was filled with the mean rather than left as NaT
    assert output["datetime_as_object"].iloc[0] == expected_fill.value
