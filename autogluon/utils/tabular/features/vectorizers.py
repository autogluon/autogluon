from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def vectorizer_auto_ml_default():
    return CountVectorizer(min_df=30, ngram_range=(1, 5), max_features=30000)


def get_ngram_freq(vectorizer, transform_matrix):
    names = vectorizer.get_feature_names()
    frequencies = transform_matrix.sum(axis=0).tolist()[0]
    ngram_freq = {}
    for ngram, freq in zip(names, frequencies):
        ngram_freq[ngram] = freq
    return ngram_freq


# Reduces vectorizer vocabulary size to vocab_size, keeping highest frequency ngrams
def downscale_vectorizer(vectorizer, ngram_freq, vocab_size):
    counter = Counter(ngram_freq)
    top_n = counter.most_common(vocab_size)
    top_n_names = [name for name, _ in top_n]

    top_n_names.sort()
    new_vocab = {}
    for i, name in enumerate(top_n_names):
        new_vocab[name] = i

    vectorizer.vocabulary_ = new_vocab
