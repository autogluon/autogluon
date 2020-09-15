# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT dataset transform."""


__all__ = ['BERTDatasetTransform']

import numpy as np
from ...utils.try_import import try_import_gluonnlp


class BERTDatasetTransform:
    """Dataset transformation for BERT-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    vocab : Vocab or BERTVocab
        The vocabulary.
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
                 vocab=None,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        self.vocab = vocab
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        nlp = try_import_gluonnlp()
        self._bert_xform = nlp.data.BERTSentenceTransform(
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
