import re

import spacy
from nltk.stem.porter import *

from .utils import *

__all__ = ['TextDataTransform']

stemmer = PorterStemmer()

spacy_nlp = spacy.load('en')

STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

NUM_REGEX = re.compile("[0-9]+")

PUNCTUATION_REGEX = re.compile("[.;:!\'?,\"()\[\]]")

HTML_REGEX = re.compile("(<br\s*/><br\s*/>)")  # TODO Refine this further


def replace_nums(text: AnyStr) -> AnyStr:
    """Removes numbers in a string"""
    return NUM_REGEX.sub('', text)


def replace_html(text: AnyStr) -> AnyStr:
    return HTML_REGEX.sub('', text)


def replace_punctuations(text: AnyStr) -> AnyStr:
    return PUNCTUATION_REGEX.sub('', text)


def stop_words(textlist: Collection[AnyStr]) -> Collection[AnyStr]:
    """Utility method for stop word removal"""
    tokens = [token for token in textlist if token not in STOP_WORDS]
    return tokens


def stemming(textlist: Collection[AnyStr]) -> Collection[AnyStr]:
    """Utility method for word stemming"""
    tokens = [stemmer.stem(token) for token in textlist]
    return tokens


def lemmatization():
    # TODO
    pass


class TextDataTransform(object):
    """
    Python class for performing data pre-processing on the text dataset.
    This class is constructed using
    """

    def __init__(self, vocab, tokenizer=nlp.data.SpacyTokenizer('en'), max_sequence_length=500, pre_tokenize_rules=None,
                 post_tokenize_rules=None):
        """
        Init method for TextDataTransform. This is a utility class for defining custom transforms on the text dataset.
        :param vocab:
        :param tokenizer:
        :param max_sequence_length:
        :param pre_tokenize_rules:
        :param post_tokenize_rules:
        """
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._pre_tokenize_rules = get_or_else(pre_tokenize_rules, [])
        self._post_tokenize_rules = get_or_else(post_tokenize_rules, [])
        self._length_clip = nlp.data.ClipSequence(max_sequence_length)

    def __call__(self, sample):
        text, label = sample

        text = self._tokenizer(self.pre_process(text))

        for rule in self._post_tokenize_rules:
            text = globals()[rule](text)

        text = self._vocab[self.post_process(text)]

        return text, label

    def pre_process(self, text):
        # text = self._pattern.sub('', text)
        for rule in self._pre_tokenize_rules:
            text = globals()[rule](text)
            if len(text) == 0:
                text = self._vocab.unknown_token

        return text

    def post_process(self, text):
        for rule in self._post_tokenize_rules:
            text = globals()[rule](text)

        return self._length_clip(text)
