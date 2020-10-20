
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.tabular.features import R_INT, R_FLOAT
from autogluon.tabular.features import AsTypeFeatureGenerator, FillNaFeatureGenerator, BulkFeatureGenerator, IdentityFeatureGenerator, CategoryFeatureGenerator, DatetimeFeatureGenerator, TextSpecialFeatureGenerator, TextNgramFeatureGenerator, DropUniqueFeatureGenerator


def test_bulk_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=10, dtype=np.uint8)

    text_ngram_feature_generator = TextNgramFeatureGenerator(vectorizer=toy_vectorizer)
    text_ngram_feature_generator.max_memory_ratio = None  # Necessary in test to avoid CI non-deterministically pruning ngram counts.

    generator = BulkFeatureGenerator(
        generators=[
            [AsTypeFeatureGenerator()],
            [FillNaFeatureGenerator()],
            [
                IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
                CategoryFeatureGenerator(),
                DatetimeFeatureGenerator(),
                TextSpecialFeatureGenerator(),
                text_ngram_feature_generator,
            ],
            [DropUniqueFeatureGenerator()],
        ]
    )

    expected_feature_metadata_in_full = {
        ('category', ()): ['cat'],
        ('datetime', ()): ['datetime'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('object', ()): ['obj'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
        ('object', ('text',)): ['text']
    }

    expected_feature_metadata_full = {
        ('category', ()): ['obj', 'cat'],
        ('float', ()): ['float'],
        ('int', ()): ['int'],
        ('int', ('binned', 'text_special')): [
            'text.char_count',
            'text.word_count',
            'text.lower_ratio',
            'text.special_ratio',
            'text.symbol_ratio. '
        ],
        ('int', ('datetime_as_int',)): [
            'datetime',
            'datetime_as_object'
        ],
        ('int', ('text_ngram',)): [
            '__nlp__.breaks',
            '__nlp__.end',
            '__nlp__.end of',
            '__nlp__.end of the',
            '__nlp__.of',
            '__nlp__.sentence',
            '__nlp__.sentence breaks',
            '__nlp__.the',
            '__nlp__.the end',
            '__nlp__.world',
            '__nlp__._total_']
    }

    expected_output_data_feat_datetime = [
        1533140820000000000,
        -9223372036854775808,
        -9223372036854775808,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1597475520000000000,
        1608257520000000000
    ]

    expected_output_data_feat_lower_ratio = [3, 2, 0, 3, 3, 3, 3, 3, 1]
    expected_output_data_feat_total = [1, 3, 0, 0, 7, 1, 3, 7, 3]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    # int and float checks
    assert output_data['int'].equals(input_data['int'])
    assert output_data['float'].equals(input_data['float'])

    # object and category checks
    assert list(output_data['obj'].values) == [1, 2, 1, 4, 4, 4, 3, 0, 0]
    assert list(output_data['cat'].values) == [0, 1, 0, 3, 3, 3, 2, np.nan, np.nan]

    # datetime checks
    assert list(output_data['datetime'].values) == list(output_data['datetime_as_object'].values)
    assert expected_output_data_feat_datetime == list(output_data['datetime'].values)

    # text_special checks
    assert expected_output_data_feat_lower_ratio == list(output_data['text.lower_ratio'].values)

    # text_ngram checks
    assert expected_output_data_feat_total == list(output_data['__nlp__._total_'].values)


# Test that bulk does not create dummy output
def test_bulk_feature_generator_no_dummy(generator_helper, data_helper):
    input_data = data_helper.generate_multi_feature_full()

    input_data_transform = data_helper.generate_empty()

    generator = BulkFeatureGenerator(
            generators=[[
                IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=['unknown_raw_type'])),
            ]]
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
