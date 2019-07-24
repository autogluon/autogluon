from typing import List

import gluonnlp as nlp
import numpy as np
from gluonnlp.data import BERTSentenceTransform

__all__ = ['TextDataTransform', 'BERTDataTransform']


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


class BERTDataTransform(object):
    """This is adapted from https://github.com/dmlc/gluon-nlp/blob/0f5170baca2cfa6d4dfef2df71b28c568c6ee03a/scripts/tests/test_bert_dataset_transform.py.
    The file had to be copied over because it could not be imported using gluonnlp.
    """

    """Dataset transformation for BERT-style sentence classification or regression.

        Parameters
        ----------
        tokenizer : BERTTokenizer.
            Tokenizer for the sentences.
        max_seq_length : int.
            Maximum sequence length of the sentences.
        labels : list of int , float or None. defaults None
            List of all label ids for the classification task and regressing task.
            If labels is None, the default task is regression
        pad : bool, default True
            Whether to pad the sentences to maximum length.
        pair : bool, default True
            Whether to transform sentences or sentence pairs.
        label_dtype: int32 or float32, default float32
            label_dtype = int32 for classification task
            label_dtype = float32 for regression task
        """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True):
        self.class_labels = class_labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
            return input_ids, valid_length, segment_ids, label
        else:
            return self._bert_xform(line)
