from autogluon.features.generators import AsTypeFeatureGenerator


def test_astype_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = AsTypeFeatureGenerator(reset_index=True)

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int_bool", "int"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    expected_feature_metadata_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): ["int"],
        ("int", ("bool",)): ["int_bool"],
        ("object", ()): ["obj"],
        ("object", ("datetime_as_object",)): ["datetime_as_object"],
        ("object", ("text",)): ["text"],
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert not input_data.equals(output_data)


def test_astype_feature_generator_bool(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_bool_feature_int()

    generator = AsTypeFeatureGenerator(convert_bool_method="v2")  # v2 doesn't edit in-place, so no need to reset_index

    expected_feature_metadata_in_full = {
        ("int", ()): ["int_bool"],
    }

    expected_feature_metadata_full = {
        ("int", ("bool",)): ["int_bool"],
    }

    # When
    generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )


def test_astype_feature_generator_bool_edgecase_with_nan(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_bool_feature_with_nan()

    generator = AsTypeFeatureGenerator(reset_index=True)

    expected_feature_metadata_in_full = {
        ("float", ()): ["edgecase_with_nan_bool"],
    }

    # Since only NaN, don't convert to boolean
    expected_feature_metadata_full = {
        ("int", ("bool",)): ["edgecase_with_nan_bool"],
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Ensure `NaN` and `None` are mapped to 0, even if they are ordered first.
    assert list(output_data["edgecase_with_nan_bool"]) == [0, 1, 0, 1]


def test_astype_feature_generator_bool_edgecase(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_bool_feature_edgecase()

    generator = AsTypeFeatureGenerator(reset_index=True)

    expected_feature_metadata_in_full = {
        ("float", ()): ["edgecase_bool"],
    }

    # Since only NaN, don't convert to boolean
    expected_feature_metadata_full = {("float", ()): ["edgecase_bool"]}

    # When
    generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )


def test_astype_feature_generator_bool_extreme_edgecase(generator_helper, data_helper):
    """
    Ensure that int 5 and string 5 are considered different values
    Also ensure that AsTypeFeatureGenerator returns the same output regardless of hyperparameters
    """
    # Given
    input_data = data_helper.generate_bool_feature_extreme_edgecase()

    generator_1 = AsTypeFeatureGenerator(reset_index=True)
    generator_2 = AsTypeFeatureGenerator(convert_bool_method_v2_threshold=1)
    generator_3 = AsTypeFeatureGenerator(convert_bool_method="v2")
    generator_4 = AsTypeFeatureGenerator(convert_bool_method="v2", convert_bool_method_v2_row_threshold=-1)
    expected_feature_metadata_in_full = {
        ("object", ()): ["edgecase_extreme_bool"],
    }
    expected_feature_metadata_full = {
        ("int", ("bool",)): ["edgecase_extreme_bool"],
    }

    out_list = []
    for generator in [generator_1, generator_2, generator_3, generator_4]:
        # When
        output_data = generator_helper.fit_transform_assert(
            input_data=input_data,
            generator=generator,
            expected_feature_metadata_in_full=expected_feature_metadata_in_full,
            expected_feature_metadata_full=expected_feature_metadata_full,
        )
        out_list.append(output_data)

    for i in range(len(out_list) - 1):
        assert out_list[i].equals(out_list[i + 1])


def test_astype_bool_batch_matches_per_column_on_mixed_wide_table():
    """The block-wise bool conversion (many columns, many rows) equals the per-column methods,
    numeric, object and categorical columns alike, missing values included."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n_rows = 300
    input_data = pd.DataFrame(
        {
            **{f"int_{i}": rng.integers(0, 2, n_rows) for i in range(8)},
            **{f"float_{i}": rng.choice([0.5, 2.5], n_rows) for i in range(4)},
            "float_nan": rng.choice([1.0, np.nan], n_rows),
            "bool_col": rng.choice([True, False], n_rows),
            "obj": rng.choice(["a", "b"], n_rows),
            "obj_nan": rng.choice(["x", None], n_rows),
            "cat": pd.Categorical(rng.choice(["p", "q"], n_rows)),
            "cat_nan": pd.Categorical(rng.choice(["m", None], n_rows)),
            # three categories declared, two present: the code of the true value is not its position among the values
            "cat_unused_level": pd.Categorical(rng.choice(["r", "t"], n_rows), categories=["r", "s", "t"]),
            "not_bool_int": rng.integers(0, 5, n_rows),
            "not_bool_obj": rng.choice(["u", "v", "w"], n_rows),
        }
    )
    generator_per_column = AsTypeFeatureGenerator(convert_bool_method="v1", reset_index=True)
    generator_realtime = AsTypeFeatureGenerator(convert_bool_method="v2", convert_bool_method_v2_row_threshold=10**9)
    generator_batch = AsTypeFeatureGenerator(convert_bool_method="v2", convert_bool_method_v2_row_threshold=1)

    outputs = [g.fit_transform(input_data.copy()) for g in (generator_per_column, generator_realtime, generator_batch)]
    assert len(generator_batch._bool_features) == 19  # 8 int + 4 float + float_nan + bool_col + obj + obj_nan + 3 cat
    for output in outputs[1:]:
        pd.testing.assert_frame_equal(outputs[0], output)
    # transform on unseen rows, including values the fit never saw (compare False, as before)
    new_rows = input_data.head(20).copy()
    new_rows.loc[new_rows.index[:5], "int_0"] = 7
    new_rows.loc[new_rows.index[:5], "obj"] = "zzz"
    new_rows["cat"] = new_rows["cat"].cat.add_categories(["new"])
    new_rows.loc[new_rows.index[:5], "cat"] = "new"
    transformed = [g.transform(new_rows.copy()) for g in (generator_per_column, generator_realtime, generator_batch)]
    for output in transformed[1:]:
        pd.testing.assert_frame_equal(transformed[0], output)
    assert (transformed[2]["int_0"].iloc[:5] == 0).all()
    assert (transformed[2]["cat"].iloc[:5] == 0).all()
