import numpy as np

from autogluon.features.generators import CategoryFeatureGenerator


def test_category_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()

    generator_1 = CategoryFeatureGenerator(minimum_cat_count=None)
    generator_2 = CategoryFeatureGenerator(minimum_cat_count=None, maximum_num_cat=2)
    generator_3 = CategoryFeatureGenerator(minimum_cat_count=3)
    generator_4 = CategoryFeatureGenerator(minimum_cat_count=None, cat_order="count")
    generator_5 = CategoryFeatureGenerator(minimum_cat_count=None, fillna="mode")
    generator_6 = CategoryFeatureGenerator(minimum_cat_count=None, fillna="rare")
    generator_7 = CategoryFeatureGenerator(minimum_cat_count=3, fillna="rare")
    generator_8 = CategoryFeatureGenerator(minimum_cat_count=4)
    generator_9 = CategoryFeatureGenerator(minimum_cat_count=4, fillna="rare")

    expected_feature_metadata_in_full = {
        ("object", ()): ["obj"],
        ("category", ()): ["cat"],
    }
    expected_feature_metadata_full = {("category", ()): ["obj", "cat"]}

    expected_cat_categories_lst = [
        [0, 1, 2, 3],
        [0, 1],
        [0],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1],
        [],
        [],
    ]

    expected_cat_values_lst = [
        [0, 1, 0, 3, 3, 3, 2, np.nan, np.nan],
        [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, np.nan],
        [2, 0, 2, 3, 3, 3, 1, np.nan, np.nan],
        [0, 1, 0, 3, 3, 3, 2, 3, 3],
        [0, 1, 0, 3, 3, 3, 2, 4, 4],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]

    expected_cat_codes_lst = [
        [0, 1, 0, 3, 3, 3, 2, -1, -1],
        [0, -1, 0, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 0, 0, 0, -1, -1, -1],
        [2, 0, 2, 3, 3, 3, 1, -1, -1],
        [0, 1, 0, 3, 3, 3, 2, 3, 3],
        [0, 1, 0, 3, 3, 3, 2, 4, 4],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]

    # When
    output_datas = []
    for generator in [
        generator_1,
        generator_2,
        generator_3,
        generator_4,
        generator_5,
        generator_6,
        generator_7,
        generator_8,
        generator_9,
    ]:
        output_data = generator_helper.fit_transform_assert(
            input_data=input_data,
            generator=generator,
            expected_feature_metadata_in_full=expected_feature_metadata_in_full,
            expected_feature_metadata_full=expected_feature_metadata_full,
        )
        output_datas.append(output_data)

    # Therefore
    for i in range(len(output_datas)):
        output_data = output_datas[i]
        for col in ["obj", "cat"]:
            assert output_data[col].dtype.name == "category"
            assert list(output_data[col].cat.categories) == expected_cat_categories_lst[i]
            assert list(output_data[col]) == expected_cat_values_lst[i]
            assert list(output_data[col].cat.codes) == expected_cat_codes_lst[i]


def test_category_feature_generator_no_op(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_standard()
    category_input_data = input_data[["obj", "cat"]].astype("category")

    generator = CategoryFeatureGenerator(minimum_cat_count=None, minimize_memory=False)

    expected_feature_metadata_in_full = {
        ("object", ()): ["obj"],
        ("category", ()): ["cat"],
    }
    expected_feature_metadata_full = {("category", ()): ["obj", "cat"]}

    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )
    assert category_input_data.equals(output_data)
