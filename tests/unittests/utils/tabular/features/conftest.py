import copy
import pytest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.utils.tabular.features.generators import AbstractFeatureGenerator


class GeneratorHelper:
    @staticmethod
    def fit_transform_assert(input_data: DataFrame, generator: AbstractFeatureGenerator, expected_feature_metadata_in_full: dict = None, expected_feature_metadata_full: dict = None):
        # Given
        original_input_data = copy.deepcopy(input_data)
    
        # Raise exception
        with pytest.raises(AssertionError):
            # Can't call transform before fit_transform
            generator.transform(input_data)
    
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

        # Ensure feature_metadata_in is as expected
        if expected_feature_metadata_in_full is not None:
            assert expected_feature_metadata_in_full == generator.feature_metadata_in.get_feature_metadata_full()
        # Ensure feature_metadata is as expected
        if expected_feature_metadata_full is not None:
            assert expected_feature_metadata_full == generator.feature_metadata.get_feature_metadata_full()
    
        return output_data


class DataHelper:
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
    def generate_multi_feature() -> DataFrame:
        df = pd.concat(
            [
                DataHelper.generate_int_feature(),
                DataHelper.generate_float_feature(),
                DataHelper.generate_obj_feature(),
                DataHelper.generate_cat_feature(),
            ],
            axis=1,
        )
        df.columns = ['int', 'float', 'obj', 'cat']
        return df


@pytest.fixture
def generator_helper():
    return GeneratorHelper


@pytest.fixture
def data_helper():
    return DataHelper
