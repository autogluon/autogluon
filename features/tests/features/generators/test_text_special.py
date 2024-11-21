import numpy as np
from packaging.version import Version

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.features.generators import TextSpecialFeatureGenerator

expected_feature_metadata_full_np_lt_2_0 = {
    ("int", ("binned", "text_special")): [
        "text.char_count",
        "text.word_count",
        "text.capital_ratio",
        "text.lower_ratio",
        "text.special_ratio",
        "text.symbol_ratio. ",
    ]
}

expected_feature_metadata_full_np_ge_2_0 = {
    ("int", ("binned", "text_special")): [
        "text.char_count",
        "text.word_count",
        "text.capital_ratio",
        "text.lower_ratio",
        "text.special_ratio",
        "text.symbol_count..",
        "text.symbol_ratio. ",
    ]
}


def test_text_special_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator = TextSpecialFeatureGenerator(min_occur_ratio=0, min_occur_offset=0)

    expected_feature_metadata_in_full = {
        ("object", ("text",)): ["text"],
    }

    expected_output_data_feat_lower_ratio_np_lt_2_0 = [3, 2, 0, 3, 3, 3, 3, 3, 1]
    expected_output_data_feat_lower_ratio_np_ge_2_0 = [2, 2, 0, 2, 2, 2, 2, 2, 1]

    if Version(np.__version__) < Version("2.0.0"):
        expected_feature_metadata_full = expected_feature_metadata_full_np_lt_2_0
        expected_output_data_feat_lower_ratio = expected_output_data_feat_lower_ratio_np_lt_2_0
    else:
        expected_feature_metadata_full = expected_feature_metadata_full_np_ge_2_0
        expected_output_data_feat_lower_ratio = expected_output_data_feat_lower_ratio_np_ge_2_0

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_lower_ratio == list(map(int, output_data["text.lower_ratio"].values))


def test_text_special_feature_generator_categorical_nan(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()
    input_data.loc[2, "text"] = None
    input_data["text"] = input_data["text"].astype("category")

    type_map_raw = {
        "int": "int",
        "float": "float",
        "obj": "object",
        "cat": "category",
        "datetime": "datetime",
        "text": "category",
        "datetime_as_object": "object",
    }
    type_map_special = {
        "text": ["text"],
    }
    feature_metadata = FeatureMetadata(
        type_map_raw,
        type_map_special=type_map_special,
    )

    generator = TextSpecialFeatureGenerator(min_occur_ratio=0, min_occur_offset=0)

    expected_feature_metadata_in_full = {
        ("category", ("text",)): ["text"],
    }

    expected_output_data_feat_lower_ratio_np_lt_2_0 = [2, 1, 2, 2, 2, 2, 2, 2, 0]
    expected_output_data_feat_lower_ratio_np_ge_2_0 = [1, 1, 1, 1, 1, 1, 1, 1, 0]

    if Version(np.__version__) < Version("2.0.0"):
        expected_feature_metadata_full = expected_feature_metadata_full_np_lt_2_0
        expected_output_data_feat_lower_ratio = expected_output_data_feat_lower_ratio_np_lt_2_0
    else:
        expected_feature_metadata_full = expected_feature_metadata_full_np_ge_2_0
        expected_output_data_feat_lower_ratio = expected_output_data_feat_lower_ratio_np_ge_2_0

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        feature_metadata_in=feature_metadata,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_lower_ratio == list(map(int, output_data["text.lower_ratio"].values))
