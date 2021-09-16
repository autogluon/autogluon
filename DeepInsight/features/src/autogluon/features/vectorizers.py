from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer_auto_ml_default():
    return CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)


def get_ngram_freq(vectorizer, transform_matrix):
    names = vectorizer.get_feature_names()
    frequencies = transform_matrix.sum(axis=0).tolist()[0]
    ngram_freq = {ngram: freq for ngram, freq in zip(names, frequencies)}
    return ngram_freq


# Reduces vectorizer vocabulary size to vocab_size, keeping highest frequency ngrams
def downscale_vectorizer(vectorizer, ngram_freq, vocab_size):
    counter = Counter(ngram_freq)
    top_n = counter.most_common(vocab_size)
    top_n_names = sorted([name for name, _ in top_n])
    new_vocab = {name: i for i, name in enumerate(top_n_names)}
    vectorizer.vocabulary_ = new_vocab
