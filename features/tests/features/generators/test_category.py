import numpy as np
import pandas as pd
import pytest

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


def test_category_feature_generator_fillna_rare_integer_categories():
    """`fillna="rare"` must keep integer categories integral.

    A category index is normally backed by a numpy dtype, so its elements are numpy integers rather
    than Python ints. Detecting integer categories with `isinstance(c, int)` alone misses them and
    sends the column down the string branch, leaving it with mixed-type categories.
    """
    input_data = pd.DataFrame({"int": pd.Series([1, 1, 2, 2, 3, 3, np.nan, np.nan], dtype="Int64").astype("category")})

    generator = CategoryFeatureGenerator(minimum_cat_count=None, fillna="rare", minimize_memory=False)
    output_data = generator.fit_transform(input_data)

    categories = list(output_data["int"].cat.categories)
    assert categories == [1, 2, 3, 4], f"expected a new integer category, got {categories}"
    assert len({type(category) for category in categories}) == 1, (
        f"categories must share a single type, got {[type(c) for c in categories]}"
    )
    assert list(output_data["int"]) == [1, 1, 2, 2, 3, 3, 4, 4]


def test_category_feature_generator_fillna_rare_object_categories():
    """Non-integer categories fall back to a reserved `'_NaN_'` category."""
    input_data = pd.DataFrame({"obj": pd.Series(["a", "a", "b", "b", "c", "c", np.nan, np.nan]).astype("category")})

    generator = CategoryFeatureGenerator(minimum_cat_count=None, fillna="rare", minimize_memory=False)
    output_data = generator.fit_transform(input_data)

    assert list(output_data["obj"].cat.categories) == ["a", "b", "c", "_NaN_"]
    assert list(output_data["obj"]) == ["a", "a", "b", "b", "c", "c", "_NaN_", "_NaN_"]

    # Categories unseen at fit time are also grouped into the rare category.
    test_data = pd.DataFrame({"obj": pd.Series(["a", "unseen", np.nan]).astype("category")})
    assert list(generator.transform(test_data)["obj"]) == ["a", "_NaN_", "_NaN_"]


def test_category_feature_generator_invalid_fillna():
    with pytest.raises(ValueError, match="is not a valid value"):
        CategoryFeatureGenerator(fillna="not_a_method")
