import numpy as np
from packaging.version import Version
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    BulkFeatureGenerator,
    CategoryFeatureGenerator,
    DatetimeFeatureGenerator,
    DropUniqueFeatureGenerator,
    FillNaFeatureGenerator,
    IdentityFeatureGenerator,
    TextNgramFeatureGenerator,
    TextSpecialFeatureGenerator,
)


def test_bulk_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    text_ngram_feature_generator = TextNgramFeatureGenerator(vectorizer=toy_vectorizer)
    text_ngram_feature_generator.max_memory_ratio = (
        None  # Necessary in test to avoid CI non-deterministically pruning ngram counts.
    )

    generator = BulkFeatureGenerator(
        generators=[
            [AsTypeFeatureGenerator(convert_bool_method="v2")],
            [FillNaFeatureGenerator()],
            [
                IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
                CategoryFeatureGenerator(),
                DatetimeFeatureGenerator(),
                TextSpecialFeatureGenerator(),
                text_ngram_feature_generator,
            ],
            [DropUniqueFeatureGenerator()],
        ],
        reset_index=True,
    )

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
        ("category", ()): ["obj", "cat"],
        ("float", ()): ["float"],
        ("int", ()): ["int"],
        ("int", ("binned", "text_special")): [
            "text.char_count",
            "text.word_count",
            "text.lower_ratio",
            "text.special_ratio",
            "text.symbol_ratio. ",
        ],
        ("int", ("bool",)): ["int_bool"],
        ("int", ("datetime_as_int",)): [
            "datetime",
            "datetime.year",
            "datetime.month",
            "datetime.day",
            "datetime.dayofweek",
            "datetime_as_object",
            "datetime_as_object.year",
            "datetime_as_object.month",
            "datetime_as_object.day",
            "datetime_as_object.dayofweek",
        ],
        ("int", ("text_ngram",)): [
            "__nlp__.breaks",
            "__nlp__.end",
            "__nlp__.end of",
            "__nlp__.end of the",
            "__nlp__.of",
            "__nlp__.of the",
            "__nlp__.sentence",
            "__nlp__.sentence breaks",
            "__nlp__.the",
            "__nlp__.the end",
            "__nlp__.the end of",
            "__nlp__.world",
            "__nlp__._total_",
        ],
    }

    expected_output_data_feat_datetime = [
        1533140820000000000,
        1301322000000000000,
        1301322000000000000,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1301322000000000000,
        1301322000000000000,
    ]

    expected_output_data_feat_lower_ratio_np_lt_2_0 = [3, 2, 0, 3, 3, 3, 3, 3, 1]
    expected_output_data_feat_lower_ratio_np_ge_2_0 = [2, 2, 0, 2, 2, 2, 2, 2, 1]

    expected_output_data_feat_total = [1, 3, 0, 0, 9, 1, 3, 9, 3]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert list(output_data.columns) == [
        "int_bool",
        "int",
        "float",
        "obj",
        "cat",
        "datetime",
        "datetime.year",
        "datetime.month",
        "datetime.day",
        "datetime.dayofweek",
        "datetime_as_object",
        "datetime_as_object.year",
        "datetime_as_object.month",
        "datetime_as_object.day",
        "datetime_as_object.dayofweek",
        "text.char_count",
        "text.word_count",
        "text.lower_ratio",
        "text.special_ratio",
        "text.symbol_ratio. ",
        "__nlp__.breaks",
        "__nlp__.end",
        "__nlp__.end of",
        "__nlp__.end of the",
        "__nlp__.of",
        "__nlp__.of the",
        "__nlp__.sentence",
        "__nlp__.sentence breaks",
        "__nlp__.the",
        "__nlp__.the end",
        "__nlp__.the end of",
        "__nlp__.world",
        "__nlp__._total_",
    ]

    # int and float checks
    assert output_data["int"].equals(input_data["int"])
    assert output_data["float"].equals(input_data["float"])

    # object and category checks
    assert list(output_data["obj"].values) == [1, np.nan, 1, 2, 2, 2, np.nan, 0, 0]
    assert list(output_data["cat"].values) == [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan]

    # datetime checks.  There are further checks in test_datetime.py
    assert list(output_data["datetime"].values) == list(output_data["datetime_as_object"].values)
    assert expected_output_data_feat_datetime == list(output_data["datetime"].values)

    # text_special checks
    assert (
        list(map(int, output_data["text.lower_ratio"].values)) == expected_output_data_feat_lower_ratio_np_lt_2_0
        if Version(np.__version__) < Version("2.0.0")
        else expected_output_data_feat_lower_ratio_np_ge_2_0
    )

    # text_ngram checks
    assert expected_output_data_feat_total == list(output_data["__nlp__._total_"].values)


# Test that bulk does not create dummy output
def test_bulk_feature_generator_no_dummy(generator_helper, data_helper):
    input_data = data_helper.generate_multi_feature_full()

    input_data_transform = data_helper.generate_empty()

    generator = BulkFeatureGenerator(
        generators=[
            [
                IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=["unknown_raw_type"])),
            ]
        ]
    )

    expected_feature_metadata_in_full = {}
    expected_feature_metadata_full = {}

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # Then
    assert output_data.shape == (9, 0)

    assert generator.transform(input_data_transform).shape == (9, 0)
    assert generator.transform(input_data_transform.head(5)).shape == (5, 0)
