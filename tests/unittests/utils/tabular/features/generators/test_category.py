import copy

import numpy as np
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

    # When
    output_data_1 = generator_1.fit_transform(input_data)
    output_data_2 = generator_2.fit_transform(input_data)
    output_data_3 = generator_3.fit_transform(input_data)
    output_data_4 = generator_4.fit_transform(input_data)
    output_data_5 = generator_5.fit_transform(input_data)

    # Therefore
    assert input_data.equals(original_input_data)
    assert input_data['col'].dtype.name == 'object'
    assert category_input_data.equals(output_data_5)

    for output_data in [output_data_1, output_data_2, output_data_3, output_data_4]:
        assert output_data['col'].dtype.name == 'category'
        assert len(input_data) == len(output_data)
        assert list(input_data.columns) == list(output_data.columns)

    assert list(output_data_1['col'].cat.categories) == [0, 1, 2, 3]
    assert list(output_data_2['col'].cat.categories) == [0, 1]
    assert list(output_data_3['col'].cat.categories) == [0]
    assert list(output_data_4['col'].cat.categories) == [0, 1, 2, 3]

    assert list(output_data_1['col']) == [0, 1, 0, 3, 3, 3, 2, np.nan, np.nan]
    assert list(output_data_2['col']) == [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan]
    assert list(output_data_3['col']) == [np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, np.nan]
    assert list(output_data_4['col']) == [2, 1, 2, 3, 3, 3, 0, np.nan, np.nan]

    assert list(output_data_1['col'].cat.codes) == [0, 1, 0, 3, 3, 3, 2, -1, -1]
    assert list(output_data_2['col'].cat.codes) == [0, -1, 0, 1, 1, 1, -1, -1, -1]
    assert list(output_data_3['col'].cat.codes) == [-1, -1, -1, 0, 0, 0, -1, -1, -1]
    assert list(output_data_4['col'].cat.codes) == [2, 1, 2, 3, 3, 3, 0, -1, -1]
