
import numpy as np

from autogluon.features.generators import CategoryFeatureGenerator


def test_category_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()
    category_input_data = input_data[['obj', 'cat']].astype('category')

    generator_1 = CategoryFeatureGenerator()
    generator_2 = CategoryFeatureGenerator(maximum_num_cat=2)
    generator_3 = CategoryFeatureGenerator(minimum_cat_count=3)
    generator_4 = CategoryFeatureGenerator(cat_order='count')
    generator_5 = CategoryFeatureGenerator(fillna='mode')
    generator_6 = CategoryFeatureGenerator(minimize_memory=False)

    expected_feature_metadata_in_full = {
        ('object', ()): ['obj'],
        ('category', ()): ['cat'],
    }
    expected_feature_metadata_full = {
        ('category', ()): ['obj', 'cat']
    }

    expected_cat_categories_lst = [
        [0, 1, 2, 3],
        [0, 1],
        [0],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    expected_cat_values_lst = [
        [0, 1, 0, 3, 3, 3, 2, np.nan, np.nan],
        [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, np.nan],
        [2, 0, 2, 3, 3, 3, 1, np.nan, np.nan],
        [0, 1, 0, 3, 3, 3, 2, 3, 3],
    ]

    expected_cat_codes_lst = [
        [0, 1, 0, 3, 3, 3, 2, -1, -1],
        [0, -1, 0, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 0, 0, 0, -1, -1, -1],
        [2, 0, 2, 3, 3, 3, 1, -1, -1],
        [0, 1, 0, 3, 3, 3, 2, 3, 3],
    ]

    # When
    output_datas = []
    for generator in [generator_1, generator_2, generator_3, generator_4, generator_5, generator_6]:
        output_data = generator_helper.fit_transform_assert(
            input_data=input_data,
            generator=generator,
            expected_feature_metadata_in_full=expected_feature_metadata_in_full,
            expected_feature_metadata_full=expected_feature_metadata_full,
        )
        output_datas.append(output_data)

    # Therefore
    assert category_input_data.equals(output_datas[5])
    output_datas = output_datas[:5]

    for i in range(len(output_datas)):
        output_data = output_datas[i]
        for col in ['obj', 'cat']:
            assert output_data[col].dtype.name == 'category'
            assert list(output_data[col].cat.categories) == expected_cat_categories_lst[i]
            assert list(output_data[col]) == expected_cat_values_lst[i]
            assert list(output_data[col].cat.codes) == expected_cat_codes_lst[i]
