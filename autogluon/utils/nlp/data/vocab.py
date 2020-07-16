# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Vocabulary."""
__all__ = ['Vocab']

import collections
import json
import warnings
import numpy as np
from typing import Dict, Hashable, List, Optional, Counter, Union, Tuple


def _check_special_token_identifier(key):
    """Raise error if the key is not valid as a key for the special token.

    Parameters
    ----------
    key
        The identifier
    """
    if not (key.endswith('_token') and key != '_token'):
        raise ValueError('Each key needs to have the form "name_token".'
                         ' Received {}'.format(key))


#TODO Revise examples
class Vocab:
    """Indexing the text tokens.

    Parameters
    ----------
    tokens
        You may specify the input tokens as a python counter object or a list.
        If it's a counter
            It represents the text tokens + the frequencies of these tokens in the text data.
            Its keys will be indexed according to frequency thresholds such as `max_size` and `min_freq`.
        If it's a list
            It represents the list of tokens we will add to the vocabulary.
            We will follow the order of the tokens in the list to assign the indices.
        The special tokens (those specified in kwargs) that are not specified in `tokens`
        will be added after the tokens.
    max_size
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count only count tokens in counter and does not
        count the special tokens like the padding token and bos token. Suppose
        that there are different keys of `counter` whose counts are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter`, this
        argument has no effect.
    min_freq
        The minimum frequency required for a token in the keys of `counter` to be indexed.
        If it is None, all keys in `counter` will be used.
    unk_token
        The representation for any unknown token. If `unk_token` is not
        `None`, looking up any token that is not part of the vocabulary and
        thus considered unknown will return the index of `unk_token`. If
        None, looking up an unknown token will result in `KeyError`.
    `**kwargs`
        Keyword arguments of the format `xxx_token` can be used to specify
        further special tokens that will be exposed as attribute of the
        vocabulary and associated with an index.
        For example, specifying `mask_token='<mask>` as additional keyword
        argument when constructing a vocabulary `v` leads to `v.mask_token`
        exposing the value of the special token: `<mask>`.
        If the specified token is not part of the Vocabulary, it will be added to the vocabulary.


    Examples
    --------

    >>> import gluonnlp as nlp
    >>> import collections
    >>> text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
    >>> counter = collections.Counter(text_data)
    >>> my_vocab = nlp.data.Vocab(counter)

    Extra keyword arguments of the format `xxx_token` are used to expose
    specified tokens as attributes.

    >>> my_vocab2 = nlp.data.Vocab(counter, special_token='hi')
    >>> my_vocab2.special_token
    'hi'

    """
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    CLS_TOKEN = '<cls>'
    SEP_TOKEN = '<sep>'
    MASK_TOKEN = '<mask>'

    def __init__(self, tokens: Optional[Union[Counter, List]] = None,
                 max_size: Optional[int] = None,
                 min_freq: Optional[int] = None, *,
                 unk_token: Optional[Hashable] = '<unk>',
                 **kwargs):
        self._all_tokens = []
        self._token_to_idx = dict()
        self._special_token_kv = collections.OrderedDict()
        # Sanity checks.
        if not (min_freq is None or min_freq > 0):
            raise ValueError('`min_freq` must be either a positive value or None.')
        # Add all tokens one by one, if the input is a python counter, we will sort the
        # (freq, token) pair in descending order to guarantee the insertion order.
        if isinstance(tokens, collections.Counter):
            if min_freq is None:
                valid_word_cnts = list(tokens.items())
            else:
                valid_word_cnts = [ele for ele in tokens.items() if ele[1] >= min_freq]
            valid_word_cnts.sort(key=lambda ele: (ele[1], ele[0]), reverse=True)
            if max_size is None or max_size >= len(valid_word_cnts):
                tokens = [ele[0] for ele in valid_word_cnts]
            else:
                tokens = [valid_word_cnts[i][0] for i in range(max_size)]
        else:
            if tokens is None:
                tokens = []
            if max_size is not None or min_freq is not None:
                warnings.warn('`max_size` and `min_freq` have no effect if the tokens is not'
                              ' a python Counter.')
        for token in tokens:
            if token in self._token_to_idx:
                raise ValueError('Find duplicated token. {} is already added to the vocabulary. '
                                 'Please check your input data.'.format(token))
            idx = len(self._all_tokens)
            self._all_tokens.append(token)
            self._token_to_idx[token] = idx
        for k, token in [('unk_token', unk_token)] + sorted(list(kwargs.items())):
            _check_special_token_identifier(k)
            if token is None:
                continue
            if hasattr(self, k) or k in self._special_token_kv:
                raise ValueError('Duplicated keys! "{}" is already in the class. '
                                 'Please consider to use another name as the identifier. '
                                 'Received kwargs["{}"] = "{}"'.format(k, k, token))
            if token in self.special_tokens:
                raise ValueError('Duplicate values! "{}" is already registered as a special token. '
                                 'All registered special tokens={}'.format(token,
                                                                           self.special_tokens))
            setattr(self, k, token)
            self._special_token_kv[k] = token
            if token in self._token_to_idx:
                idx = self._token_to_idx[token]
            else:
                idx = len(self._all_tokens)
                self._all_tokens.append(token)
                self._token_to_idx[token] = idx
            # Add the {name}_idx properties to the object
            setattr(self, k[:(-6)] + '_id', idx)
        self._special_token_kv = collections.OrderedDict(
            sorted(self._special_token_kv.items(),
                   key=lambda ele: self._token_to_idx[ele[1]]))
        special_tokens_set = frozenset(self._special_token_kv.values())
        self._non_special_tokens = [ele for ele in self._all_tokens
                                    if ele not in special_tokens_set]

    @property
    def has_unk(self) -> bool:
        return hasattr(self, 'unk_token')

    @property
    def all_tokens(self) -> List[Hashable]:
        """Return all tokens in the vocabulary"""
        return self._all_tokens

    @property
    def non_special_tokens(self) -> List[Hashable]:
        """Return all tokens that are not marked as special tokens."""
        return self._non_special_tokens

    @property
    def special_tokens(self) -> List[Hashable]:
        """Return all special tokens.  We will order the tokens in ascending order of their
        index in the vocabulary."""
        return list(self._special_token_kv.values())

    @property
    def special_token_keys(self) -> List[str]:
        """Return all the keys to fetch the special tokens. We will order them in ascending order
        of their index in the vocabulary."""
        return list(self._special_token_kv.keys())

    @property
    def special_tokens_kv(self) -> 'OrderedDict[str, Hashable]':
        """Return the dictionary that maps the special_token_key to the special token"""
        return self._special_token_kv

    @property
    def token_to_idx(self) -> Dict[Hashable, int]:
        return self._token_to_idx

    def to_tokens(self, idx: Union[int, Tuple[int], List[int], np.ndarray])\
            -> Union[Hashable, List[Hashable]]:
        """Get the tokens correspond to the chosen indices

        Parameters
        ----------
        idx
            The index used to select the tokens.

        Returns
        -------
        ret
            The tokens of these selected indices.
        """
        if isinstance(idx, (list, tuple)):
            return [self.all_tokens[i] for i in idx]
        elif isinstance(idx, np.ndarray):
            if idx.ndim == 0:
                return self.all_tokens[idx]
            elif idx.ndim == 1:
                return [self.all_tokens[i] for i in idx]
            else:
                raise ValueError('Unsupported numpy ndarray ndim={}'.format(idx.ndim))
        else:
            return self.all_tokens[idx]

    def __contains__(self, token: Hashable) -> bool:
        """Checks whether a text token exists in the vocabulary.


        Parameters
        ----------
        token
            A text token.


        Returns
        -------
        ret
            Whether the text token exists in the vocabulary (including `unknown_token`).
        """
        return token in self._token_to_idx

    def __getitem__(self, tokens: Union[Hashable, List[Hashable], Tuple[Hashable]])\
            -> Union[int, List[int]]:
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens
            A source token or tokens to be converted.


        Returns
        -------
        ret
            A token index or a list of token indices according to the vocabulary.
        """

        if isinstance(tokens, (list, tuple)):
            if self.has_unk:
                return [self._token_to_idx.get(token, self.unk_id) for token in tokens]
            else:
                return [self._token_to_idx[token] for token in tokens]
        else:
            if self.has_unk:
                return self._token_to_idx.get(tokens, self.unk_id)
            else:
                return self._token_to_idx[tokens]

    def __len__(self):
        return len(self.all_tokens)

    def __call__(self, tokens: Union[Hashable, List[Hashable], Tuple[Hashable]])\
            -> Union[int, np.ndarray]:
        """Looks up indices of text tokens according to the vocabulary.

        Parameters
        ----------
        tokens
            A source token or tokens to be converted.


        Returns
        -------
        ret
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __repr__(self):
        unk = '"{}"'.format(self.unk_token) if self.has_unk else 'None'
        extra_special_tokens = []
        for k, v in self._special_token_kv.items():
            if k != 'unk_token':
                extra_special_tokens.append('{}="{}"'.format(k, v))
        if len(extra_special_tokens) > 0:
            extra_special_token_str = ', {}'.format(', '.join(extra_special_tokens))
        else:
            extra_special_token_str = ''
        return 'Vocab(size={}, unk_token={}{})'.format(len(self), unk, extra_special_token_str)

    def to_json(self) -> str:
        """Serialize Vocab object into a json string.

        Returns
        -------
        ret
            The serialized json string
        """
        vocab_dict = dict()
        # Perform sanity check to make sure that we are able to reconstruct the original vocab
        for i, tok in enumerate(self._all_tokens):
            if self._token_to_idx[tok] != i:
                warnings.warn('The vocabulary is corrupted! One possible reason is that the '
                              'tokens are changed manually without updating the '
                              '_token_to_idx map. Please check your code or report an issue in '
                              'Github!')
        vocab_dict['all_tokens'] = self._all_tokens
        vocab_dict['special_token_key_value'] = self._special_token_kv
        ret = json.dumps(vocab_dict, ensure_ascii=False)
        return ret

    def save(self, path: str):
        """Save vocab to a json file

        Parameters
        ----------
        path
            The file to write the json string. Nothing happens if it is None.
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: Union[str, bytes, bytearray]) -> 'Vocab':
        """Deserialize Vocab object from json string.

        Parameters
        ----------
        json_str
            Serialized json string of a Vocab object.

        Returns
        -------
        vocab
            The constructed Vocab object
        """
        vocab_dict = json.loads(json_str)
        all_tokens = vocab_dict.get('all_tokens')
        special_token_kv = vocab_dict.get('special_token_key_value')
        if 'unk_token' not in special_token_kv:
            special_token_kv['unk_token'] = None
        vocab = cls(tokens=all_tokens, **special_token_kv)
        return vocab

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        """Save the vocabulary to location specified by the filename

        Parameters
        ----------
        path
            The path to load the vocabulary

        Returns
        -------
        vocab
            The constructed Vocab object
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
