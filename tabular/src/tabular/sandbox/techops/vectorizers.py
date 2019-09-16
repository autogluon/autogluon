
from sklearn.feature_extraction.text import CountVectorizer
# from f3_grail_monty.acs_prototype.vectorizer import Vectorizer


def vectorizer_auto_ml_default():
    return CountVectorizer(min_df=30, ngram_range=(1, 5), max_features=1100)

def vectorizer_3():
    return CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=1100)

def vectorizer_tst():
    return CountVectorizer(min_df=10, ngram_range=(30, 30), max_features=100)

# def vectorizer_tst_2():
#     return Vectorizer(min_df=10, ngram_range=(15, 100), max_features=100000)  # TODO: EXPERIMENT!
