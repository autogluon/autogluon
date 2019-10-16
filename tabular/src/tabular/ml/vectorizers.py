
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer_auto_ml_default():
    return CountVectorizer(min_df=30, ngram_range=(1, 5), max_features=1100)
