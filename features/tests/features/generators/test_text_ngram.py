import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.features.generators import TextNgramFeatureGenerator

expected_feature_metadata_in_full = {
    ("object", ("text",)): ["text"],
}
expected_feature_metadata_full = {
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
    ]
}

expected_output_data_feat_total = [1, 3, 0, 0, 9, 1, 3, 9, 3]


def test_text_ngram_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    # max_memory_ratio=None in test to avoid CI reducing ngrams non-deterministically.
    generator = TextNgramFeatureGenerator(max_memory_ratio=None, vectorizer=toy_vectorizer)

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_total == list(output_data["__nlp__._total_"].values)


def test_text_ngram_feature_generator_categorical_nan(generator_helper, data_helper):
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

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000, dtype=np.uint8)

    # max_memory_ratio=None in test to avoid CI reducing ngrams non-deterministically.
    generator = TextNgramFeatureGenerator(max_memory_ratio=None, vectorizer=toy_vectorizer)

    expected_feature_metadata_in_full = {
        ("category", ("text",)): ["text"],
    }

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        feature_metadata_in=feature_metadata,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_total == list(output_data["__nlp__._total_"].values)
