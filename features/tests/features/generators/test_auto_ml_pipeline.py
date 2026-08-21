import numpy as np
import pytest
from packaging.version import Version
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.features.generators import (
    AutoMLPipelineFeatureGenerator,
    IdentityFeatureGenerator,
    TextNgramFeatureGenerator,
)


def test_auto_ml_pipeline_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    with pytest.raises(KeyError):
        # generators is an invalid argument
        AutoMLPipelineFeatureGenerator(generators=[], vectorizer=toy_vectorizer)

    generator = AutoMLPipelineFeatureGenerator(vectorizer=toy_vectorizer)

    for generator_stage in generator.generators:
        for generator_inner in generator_stage:
            if isinstance(generator_inner, TextNgramFeatureGenerator):
                # Necessary in test to avoid CI non-deterministically pruning ngram counts.
                generator_inner.max_memory_ratio = None

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
        ],
        ("int", ("text_ngram",)): [
            "__nlp__.breaks",
            "__nlp__.end",
            "__nlp__.end of",
            "__nlp__.sentence",
            "__nlp__.the",
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

    # int and float checks
    assert output_data["int"].equals(input_data["int"])
    assert output_data["float"].equals(input_data["float"])

    # object and category checks
    assert list(output_data["obj"].values) == [1, np.nan, 1, 2, 2, 2, np.nan, 0, 0]
    assert list(output_data["cat"].values) == [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan]

    # datetime checks.  There are further checks in test_datetime.py
    assert expected_output_data_feat_datetime == list(output_data["datetime"].values)

    # text_special checks
    assert (
        list(map(int, output_data["text.lower_ratio"].values)) == expected_output_data_feat_lower_ratio_np_lt_2_0
        if Version(np.__version__) < Version("2.0.0")
        else expected_output_data_feat_lower_ratio_np_ge_2_0
    )

    # text_ngram checks
    assert expected_output_data_feat_total == list(output_data["__nlp__._total_"].values)


def test_auto_ml_pipeline_feature_generator_raw_text(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    generator = AutoMLPipelineFeatureGenerator(enable_raw_text_features=True, vectorizer=toy_vectorizer)

    for generator_stage in generator.generators:
        for generator_inner in generator_stage:
            if isinstance(generator_inner, TextNgramFeatureGenerator):
                # Necessary in test to avoid CI non-deterministically pruning ngram counts.
                generator_inner.max_memory_ratio = None

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
        ],
        ("int", ("text_ngram",)): [
            "__nlp__.breaks",
            "__nlp__.end",
            "__nlp__.end of",
            "__nlp__.sentence",
            "__nlp__.the",
            "__nlp__.world",
            "__nlp__._total_",
        ],
        ("object", ("text",)): ["text_raw_text"],
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert list(input_data["text"].values) == list(output_data["text_raw_text"].values)


def test_auto_ml_pipeline_feature_generator_only_raw_text(generator_helper, data_helper):
    """
    Specifically tests when only text columns are provided.
    This verifies the edge-case bug in v0.6.2 from https://github.com/autogluon/autogluon/issues/2688 is not present.
    """

    # Given
    input_data = data_helper.generate_text_feature().to_frame("text")

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    generator = AutoMLPipelineFeatureGenerator(enable_raw_text_features=True, vectorizer=toy_vectorizer)

    for generator_stage in generator.generators:
        for generator_inner in generator_stage:
            if isinstance(generator_inner, TextNgramFeatureGenerator):
                # Necessary in test to avoid CI non-deterministically pruning ngram counts.
                generator_inner.max_memory_ratio = None

    expected_feature_metadata_in_full = {("object", ("text",)): ["text"]}

    expected_feature_metadata_full = {
        ("int", ("binned", "text_special")): [
            "text.char_count",
            "text.word_count",
            "text.lower_ratio",
            "text.special_ratio",
            "text.symbol_ratio. ",
        ],
        ("int", ("text_ngram",)): [
            "__nlp__.breaks",
            "__nlp__.end",
            "__nlp__.end of",
            "__nlp__.sentence",
            "__nlp__.the",
            "__nlp__.world",
            "__nlp__._total_",
        ],
        ("object", ("text",)): ["text_raw_text"],
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert list(input_data["text"].values) == list(output_data["text_raw_text"].values)


def test_auto_ml_pipeline_feature_generator_duplicates(generator_helper, data_helper):
    """
    Test the most complicated situation: Many duplicate features, useless features, and all dtypes at once
    This test ensures the fix in https://github.com/autogluon/autogluon/pull/2986 works, test failed prior to fix
    """
    # Given
    input_data = data_helper.generate_duplicate()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    with pytest.raises(KeyError):
        # generators is an invalid argument
        AutoMLPipelineFeatureGenerator(generators=[], vectorizer=toy_vectorizer)

    generator = AutoMLPipelineFeatureGenerator(vectorizer=toy_vectorizer)

    for generator_stage in generator.generators:
        for generator_inner in generator_stage:
            if isinstance(generator_inner, TextNgramFeatureGenerator):
                # Necessary in test to avoid CI non-deterministically pruning ngram counts.
                generator_inner.max_memory_ratio = None

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
            "__nlp__.sentence",
            "__nlp__.the",
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

    # int and float checks
    assert output_data["int"].equals(input_data["int"])
    assert output_data["float"].equals(input_data["float"])

    # object and category checks
    assert list(output_data["obj"].values) == [1, np.nan, 1, 2, 2, 2, np.nan, 0, 0]
    assert list(output_data["cat"].values) == [0, np.nan, 0, 1, 1, 1, np.nan, np.nan, np.nan]

    # datetime checks.  There are further checks in test_datetime.py
    assert expected_output_data_feat_datetime == list(output_data["datetime_as_object"].values)

    # text_special checks
    assert (
        list(map(int, output_data["text.lower_ratio"].values)) == expected_output_data_feat_lower_ratio_np_lt_2_0
        if Version(np.__version__) < Version("2.0.0")
        else expected_output_data_feat_lower_ratio_np_ge_2_0
    )

    # text_ngram checks
    assert expected_output_data_feat_total == list(output_data["__nlp__._total_"].values)


def test_auto_ml_pipeline_feature_generator_duplicates_without_dedupe(generator_helper, data_helper):
    """
    This test additionally turns off the drop_duplicates logic

    Test the most complicated situation: Many duplicate features, useless features, and all dtypes at once
    This test ensures the fix in https://github.com/autogluon/autogluon/pull/2986 works, test failed prior to fix
    """
    # Given
    input_data = data_helper.generate_duplicate()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    with pytest.raises(KeyError):
        # generators is an invalid argument
        AutoMLPipelineFeatureGenerator(generators=[], vectorizer=toy_vectorizer, post_drop_duplicates=False)

    generator = AutoMLPipelineFeatureGenerator(vectorizer=toy_vectorizer, post_drop_duplicates=False)

    for generator_stage in generator.generators:
        for generator_inner in generator_stage:
            if isinstance(generator_inner, TextNgramFeatureGenerator):
                # Necessary in test to avoid CI non-deterministically pruning ngram counts.
                generator_inner.max_memory_ratio = None

    expected_feature_metadata_in_full = {
        ("category", ()): ["cat"],
        ("datetime", ()): ["datetime"],
        ("float", ()): ["float"],
        ("int", ()): [
            "int_bool",
            "int",
            "int_bool_dup_1",
            "int_bool_dup_2",
            "int_bool_dup_3",
            "int_bool_dup_4",
            "int_bool_dup_5",
            "int_bool_dup_6",
            "int_bool_dup_7",
            "int_bool_dup_8",
            "int_bool_dup_9",
            "int_bool_dup_10",
            "int_bool_dup_11",
            "int_bool_dup_12",
            "int_bool_dup_13",
            "int_bool_dup_14",
            "int_bool_dup_15",
            "int_bool_dup_16",
            "int_bool_dup_17",
            "int_bool_dup_18",
            "int_bool_dup_19",
            "int_bool_dup_20",
            "int_bool_dup_final",
        ],
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
        ("int", ("bool",)): [
            "int_bool",
            "int_bool_dup_1",
            "int_bool_dup_2",
            "int_bool_dup_3",
            "int_bool_dup_4",
            "int_bool_dup_5",
            "int_bool_dup_6",
            "int_bool_dup_7",
            "int_bool_dup_8",
            "int_bool_dup_9",
            "int_bool_dup_10",
            "int_bool_dup_11",
            "int_bool_dup_12",
            "int_bool_dup_13",
            "int_bool_dup_14",
            "int_bool_dup_15",
            "int_bool_dup_16",
            "int_bool_dup_17",
            "int_bool_dup_18",
            "int_bool_dup_19",
            "int_bool_dup_20",
            "int_bool_dup_final",
        ],
        ("int", ("datetime_as_int",)): [
            "datetime_as_object",
            "datetime_as_object.year",
            "datetime_as_object.month",
            "datetime_as_object.day",
            "datetime_as_object.dayofweek",
            "datetime",
            "datetime.year",
            "datetime.month",
            "datetime.day",
            "datetime.dayofweek",
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


def test_add_custom_feature_generators():
    """Test the _add_custom_feature_generators method of AutoMLPipelineFeatureGenerator.

    Ensures that the custom feature generator insertion logic works as expected
    for different use cases.
    """
    gen_1 = IdentityFeatureGenerator()
    gen_2 = TextNgramFeatureGenerator()
    fg = AutoMLPipelineFeatureGenerator(custom_feature_generators=[gen_1, gen_2])

    fg_main_stage = fg.generators[2]
    assert fg_main_stage[-2] == gen_1
    assert fg_main_stage[-1] == gen_2
