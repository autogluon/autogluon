import copy

import numpy as np
import pytest
from pandas import DataFrame

from autogluon.utils.tabular.features.generators import CategoryFeatureGenerator


def test_category_feature_generator():
    # Given
    input_data = DataFrame(['a', 'b', 'a', 'd', 'd', 'd', 'c', np.nan, np.nan], columns=['col'])
    original_input_data = copy.deepcopy(input_data)
    category_input_data = input_data.astype('category')

    generator_1 = CategoryFeatureGenerator()
    generator_2 = CategoryFeatureGenerator(maximum_num_cat=2)
    generator_3 = CategoryFeatureGenerator(minimum_cat_count=3)
    generator_4 = CategoryFeatureGenerator(cat_order='count')
    generator_5 = CategoryFeatureGenerator(minimize_memory=False)

    expected_feature_metadata_in_full = {('object', ()): ['col']}
    expected_feature_metadata_full = {('category', ()): ['col']}

    expected_cat_categories_lst = [
        [0, 1, 2, 3],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]

    expected_cat_values_lst = [
        [0, 1, 0, 3, 3, 3, 2, np.nan, np.nan],
        [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, np.nan],
        [2, 1, 2, 3, 3, 3, 0, np.nan, np.nan],
    ]

    expected_cat_codes_lst = [
        [0, 1, 0, 3, 3, 3, 2, -1, -1],
        [0, -1, 0, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 0, 0, 0, -1, -1, -1],
        [2, 1, 2, 3, 3, 3, 0, -1, -1],
    ]

    # When
    output_datas = []
    for generator in [generator_1, generator_2, generator_3, generator_4, generator_5]:
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
        output_data_transform = generator.transform(input_data)

        # Ensure transform and fit_transform output are the same for training data
        assert output_data.equals(output_data_transform)
        output_datas.append(output_data)
        assert len(input_data) == len(output_data)

        # Ensure feature metadata is as expected
        assert expected_feature_metadata_in_full == generator.feature_metadata_in.get_feature_metadata_full()
        assert expected_feature_metadata_full == generator.feature_metadata.get_feature_metadata_full()

        assert list(input_data.columns) == list(output_data.columns)

    # Therefore
    assert input_data['col'].dtype.name == 'object'
    assert category_input_data.equals(output_datas[4])
    output_datas = output_datas[:4]

    for i in range(len(output_datas)):
        output_data = output_datas[i]
        assert output_data['col'].dtype.name == 'category'
        assert list(output_data['col'].cat.categories) == expected_cat_categories_lst[i]
        assert list(output_data['col']) == expected_cat_values_lst[i]
        assert list(output_data['col'].cat.codes) == expected_cat_codes_lst[i]
