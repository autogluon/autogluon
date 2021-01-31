import copy
import pytest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.features.generators import AbstractFeatureGenerator


class GeneratorHelper:
    @staticmethod
    def fit_transform_assert(input_data: DataFrame, generator: AbstractFeatureGenerator, expected_feature_metadata_in_full: dict = None, expected_feature_metadata_full: dict = None):
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
                generator.fit_transform(input_data_with_duplicate_columns)

        assert not generator.is_fit()
        output_data = generator.fit_transform(input_data)
        assert generator.is_fit()
        with pytest.raises(AssertionError):
            # Can't call fit_transform after fit
            generator.fit_transform(input_data)
    
        # Ensure input_data is not altered inplace
        assert input_data.equals(original_input_data)

        # Ensure unchanged row count
        assert len(input_data) == len(output_data)

        # Ensure transform and fit_transform output are the same for training data
        output_data_transform = generator.transform(input_data)
        assert output_data.equals(output_data_transform)

        # Ensure transform will be the same if unnecessary columns are removed from input
        output_data_transform = generator.transform(input_data[generator.features_in])
        assert output_data.equals(output_data_transform)

        # Ensure transform will be the same if input feature order is not the same as generator.features_in
        reversed_column_names = list(input_data.columns)
        reversed_column_names.reverse()
        output_data_transform = generator.transform(input_data[reversed_column_names])
        assert output_data.equals(output_data_transform)

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
        input_data_with_extra['__UNKNOWN_COLUMN__'] = 0
        output_data_transform = generator.transform(input_data_with_extra)
        assert output_data.equals(output_data_transform)

        # Ensure feature_metadata_in is as expected
        if expected_feature_metadata_in_full is not None:
            assert expected_feature_metadata_in_full == generator.feature_metadata_in.to_dict(inverse=True)
        # Ensure feature_metadata is as expected
        if expected_feature_metadata_full is not None:
            assert expected_feature_metadata_full == generator.feature_metadata.to_dict(inverse=True)
    
        return output_data


class DataHelper:
    @staticmethod
    def generate_empty() -> DataFrame:
        return DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    @staticmethod
    def generate_obj_feature() -> Series:
        return Series(['a', 'b', 'a', 'd', 'd', 'd', 'c', np.nan, np.nan])

    @staticmethod
    def generate_int_feature() -> Series:
        return Series([12, 0, -8, 3, 2, 2, 2, 0, 5])

    @staticmethod
    def generate_cat_feature() -> Series:
        return DataHelper.generate_obj_feature().astype('category')

    @staticmethod
    def generate_float_feature() -> Series:
        return Series([12, 0, np.nan, 3, -0.2, 2.3, 2.3, 0.1, 5])

    @staticmethod
    def generate_text_feature() -> Series:
        return Series(
            [
                'hello world',
                'sentence breaks.',
                '',
                'unique words',
                'the end of the sentence',
                'goodbye world',
                'the end is not the end is not the end is not the end is not the end',
                'the end of the world',
                'sentence. breaks. sentence. breaks. sentence. breaks. sentence. breaks.',
            ]
        )

    @staticmethod
    def generate_datetime_as_object_feature() -> Series:
        return Series(
            [
                '8/1/2018 16:27',
                '',
                np.nan,
                '4/20/2018 15:37',
                '4/20/2018 15:37',
                '1/01/1800 00:00',
                '12/31/2200 23:59',
                '8/15/2020 7:12',
                '12/18/2020 2:12',
            ]
        )

    @staticmethod
    def generate_datetime_feature() -> Series:
        return pd.to_datetime(DataHelper.generate_datetime_as_object_feature())

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
        df.columns = ['int', 'float', 'obj', 'cat', 'datetime']
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
        df.columns = ['text', 'datetime_as_object']
        return df

    @staticmethod
    def generate_multi_feature_full() -> DataFrame:
        df = pd.concat(
            [
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
