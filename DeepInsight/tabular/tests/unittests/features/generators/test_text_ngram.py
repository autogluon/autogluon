
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.features.generators import TextNgramFeatureGenerator


def test_text_ngram_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    toy_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=10, dtype=np.uint8)

    # max_memory_ratio=None in test to avoid CI reducing ngrams non-deterministically.
    generator = TextNgramFeatureGenerator(max_memory_ratio=None, vectorizer=toy_vectorizer)

    expected_feature_metadata_in_full = {
        ('object', ('text',)): ['text'],
    }
    expected_feature_metadata_full = {('int', ('text_ngram',)): [
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
        '__nlp__._total_'
    ]}

    expected_output_data_feat_total = [1, 3, 0, 0, 7, 1, 3, 7, 3]

    # When
    output_data = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full,
    )

    assert expected_output_data_feat_total == list(output_data['__nlp__._total_'].values)
