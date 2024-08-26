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
