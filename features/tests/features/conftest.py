import copy

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.features.generators import AbstractFeatureGenerator


class GeneratorHelper:
    @staticmethod
    def fit_transform_assert(
        input_data: DataFrame,
        generator: AbstractFeatureGenerator,
        y=None,
        feature_metadata_in: FeatureMetadata = None,
        expected_feature_metadata_in_full: dict = None,
        expected_feature_metadata_full: dict = None,
        can_transform_on_train: bool = True,
    ):
        # Given
        original_input_data = copy.deepcopy(input_data)

        # Raise exception
        with pytest.raises(AssertionError):
            # Can't call transform before fit_transform
            generator.transform(input_data)

        if len(input_data.columns) > 0:
            # Raise exception
            with pytest.raises(AssertionError):
                input_data_with_duplicate_columns = pd.concat([input_data, input_data], axis=1)
                # Can't call fit_transform with duplicate column names
                generator.fit_transform(input_data_with_duplicate_columns, y=y, feature_metadata_in=feature_metadata_in)

        assert not generator.is_fit()
        output_data = generator.fit_transform(input_data, y=y, feature_metadata_in=feature_metadata_in)
        assert generator.is_fit()
        with pytest.raises(AssertionError):
            # Can't call fit_transform after fit
            generator.fit_transform(input_data, y=y, feature_metadata_in=feature_metadata_in)

        # Ensure input_data is not altered inplace by fit_transform
        assert input_data.equals(original_input_data)

        # Ensure unchanged row count
        assert len(input_data) == len(output_data)

        output_data_og = output_data
        # Ensure transform and fit_transform output are the same for training data
        output_data_transform = generator.transform(input_data)
        if can_transform_on_train:
            assert output_data.equals(output_data_transform)
        else:
            output_data = output_data_transform  # Do this for future transform checks

        # Ensure input_data is not altered inplace by transform
        assert input_data.equals(original_input_data)

        # Ensure transform will be the same if unnecessary columns are removed from input
        input_data_features_in = input_data[generator.features_in]
        original_input_data_features_in = copy.deepcopy(input_data_features_in)
        output_data_transform = generator.transform(input_data_features_in)
        assert output_data.equals(output_data_transform)

        # Ensure input_data is not altered inplace by transform when features_in match exactly
        assert input_data_features_in.equals(original_input_data_features_in)

        # Ensure transform will be the same if input feature order is not the same as generator.features_in
        reversed_column_names = list(input_data.columns)
        reversed_column_names.reverse()
        input_data_reversed = input_data[reversed_column_names]
        original_input_data_reversed = copy.deepcopy(input_data_reversed)
        output_data_transform = generator.transform(input_data_reversed)
        assert output_data.equals(output_data_transform)

        # Ensure input_data is not altered inplace by transform when column order is reversed
        assert input_data_reversed.equals(original_input_data_reversed)

        # Ensure output feature order is correct
        assert generator.features_out == list(output_data.columns)

        if generator.features_in:
            with pytest.raises(KeyError):
                # Error if missing input feature
                generator.transform(input_data.drop(generator.features_in[0]))
            with pytest.raises(KeyError):
                # Error if missing all input features
                generator.transform(pd.DataFrame())

        # Ensure unknown input columns don't affect output
        input_data_with_extra = copy.deepcopy(input_data)
        input_data_with_extra["__UNKNOWN_COLUMN__"] = 0
        original_input_data_with_extra = copy.deepcopy(input_data_with_extra)
        output_data_transform = generator.transform(input_data_with_extra)
        assert output_data.equals(output_data_transform)

        # Ensure input_data is not altered inplace by transform when extra columns are present
        assert input_data_with_extra.equals(original_input_data_with_extra)

        # Ensure feature_metadata_in is as expected
        if expected_feature_metadata_in_full is not None:
            assert expected_feature_metadata_in_full == generator.feature_metadata_in.to_dict(inverse=True)
        # Ensure feature_metadata is as expected
        if expected_feature_metadata_full is not None:
            assert expected_feature_metadata_full == generator.feature_metadata.to_dict(inverse=True)

        return output_data_og


class DataHelper:
    @staticmethod
    def generate_empty() -> DataFrame:
        return DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    @staticmethod
    def generate_obj_feature() -> Series:
        return Series(["a", "b", "a", "d", "d", "d", "c", np.nan, np.nan])

    @staticmethod
    def generate_int_feature() -> Series:
        return Series([12, 0, -8, 3, 2, 2, 2, 0, 5])

    @staticmethod
    def generate_cat_feature() -> Series:
        return DataHelper.generate_obj_feature().astype("category")

    @staticmethod
    def generate_float_feature() -> Series:
        return Series([12, 0, np.nan, 3, -0.2, 2.3, 2.3, 0.1, 5])

    @staticmethod
    def generate_text_feature() -> Series:
        return Series(
            [
                "hello world",
                "sentence breaks.",
                "",
                "unique words",
                "the end of the sentence",
                "goodbye world",
                "the end is not the end is not the end is not the end is not the end",
                "the end of the world",
                "sentence. breaks. sentence. breaks. sentence. breaks. sentence. breaks.",
            ]
        )

    @staticmethod
    def generate_datetime_as_object_feature() -> Series:
        return Series(
            [
                "8/1/2018 16:27",
                "",  # nan
                np.nan,  # nan
                "4/20/2018 15:37",
                "4/20/2018 15:37",
                "1/01/1800 00:00",
                "12/31/2200 23:59",
                "8/15/2700 7:12",  # nan
                "12/18/1000 2:12",  # nan
            ]
        )

    @staticmethod
    def generate_datetime_as_object_feature_advanced() -> Series:
        """Nightmare input for datetime"""
        return Series(
            [
                "8/1/2018 16:27",
                "",  # nan
                np.nan,  # nan
                "2021-08-09T17:07:08.659-0400",  # With timezone 4
                "2021-08-09T17:08:15.541-0400",  # With timezone 4
                "2021-01-07T12:33:23.938-0500",  # With timezone 5
            ]
        )

    @staticmethod
    def generate_datetime_feature() -> Series:
        return pd.to_datetime(DataHelper.generate_datetime_as_object_feature(), errors="coerce", format="mixed")

    @staticmethod
    def generate_bool_feature_int(name="int_bool") -> DataFrame:
        return Series([0, 1, 1, 0, 0, 0, 1, 0, 1], name=name).to_frame()

    @staticmethod
    def generate_bool_feature_with_nan() -> DataFrame:
        return Series([None, 5, None, 5], name="edgecase_with_nan_bool").to_frame()

    @staticmethod
    def generate_bool_feature_edgecase() -> DataFrame:
        return Series([None, None, None, np.nan, np.nan, np.nan, None, None, None], name="edgecase_bool").to_frame()

    @staticmethod
    def generate_bool_feature_extreme_edgecase() -> DataFrame:
        return Series([5, "5", 5, 5, "5"], name="edgecase_extreme_bool").to_frame()

    @staticmethod
    def generate_multi_feature_standard() -> DataFrame:
        df = pd.concat(
            [
                DataHelper.generate_int_feature(),
                DataHelper.generate_float_feature(),
                DataHelper.generate_obj_feature(),
                DataHelper.generate_cat_feature(),
                DataHelper.generate_datetime_feature(),
            ],
            axis=1,
        )
        df.columns = ["int", "float", "obj", "cat", "datetime"]
        return df

    @staticmethod
    def generate_useless_category() -> DataFrame:
        return Series(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
            name="cat_useless",
        ).to_frame()

    @staticmethod
    def generate_duplicate() -> DataFrame:
        df_bool_dupes = []
        for i in range(20):
            df_bool_dupes.append(
                DataHelper.generate_bool_feature_int(name=f"int_bool_dup_{i + 1}"),
            )

        df_to_concat = (
            [
                DataHelper.generate_bool_feature_int(),
                DataHelper.generate_multi_feature_special(),
                DataHelper.generate_useless_category(),
                DataHelper.generate_multi_feature_standard(),
            ]
            + df_bool_dupes
            + [
                DataHelper.generate_bool_feature_int(name="int_bool_dup_final"),
            ]
        )

        df = pd.concat(
            df_to_concat,
            axis=1,
        )
        return df

    @staticmethod
    def generate_multi_feature_special() -> DataFrame:
        df = pd.concat(
            [
                DataHelper.generate_text_feature(),
                DataHelper.generate_datetime_as_object_feature(),
            ],
            axis=1,
        )
        df.columns = ["text", "datetime_as_object"]
        return df

    @staticmethod
    def generate_multi_feature_full() -> DataFrame:
        df = pd.concat(
            [
                DataHelper.generate_bool_feature_int(),
                DataHelper.generate_multi_feature_standard(),
                DataHelper.generate_multi_feature_special(),
            ],
            axis=1,
        )
        return df


@pytest.fixture
def generator_helper():
    return GeneratorHelper


@pytest.fixture
def data_helper():
    return DataHelper
