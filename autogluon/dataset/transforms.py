from typing import List

from .utils import *

__all__ = ['TextDataTransform']


class TextDataTransform(object):
    """
    Python class for performing data pre-processing on the text dataset.
    This class is constructed using
    """

    def __init__(self, vocab, tokenizer=nlp.data.SpacyTokenizer('en'),
                 transforms: List = None):
        """
        Init method for TextDataTransform. This is a utility class for defining custom transforms on the text dataset.
        :param vocab:
        :param tokenizer:
        :param transforms: A List of transforms from gluonnlp.data.transforms. eg : ClipSequence, PadSequence
        """
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._transforms = transforms

    def __call__(self, sample):
        text, label = sample

        text = self._tokenizer(text)

        # TODO : Should this be called before tokenize ? Or after

        for rule in self._transforms:
            text = rule(text)

        text = self._vocab[text]

        return text, label
