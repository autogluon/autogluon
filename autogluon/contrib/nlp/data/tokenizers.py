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
"""Tokenizers."""
__all__ = ['WhitespaceTokenizer', 'SpacyTokenizer', 'JiebaTokenizer', 'MosesTokenizer',
           'SubwordNMTTokenizer', 'YTTMTokenizer', 'SentencepieceTokenizer',
           'HuggingFaceBPETokenizer', 'HuggingFaceByteBPETokenizer',
           'HuggingFaceWordPieceTokenizer',
           'create', 'create_with_json', 'list_all']

from typing import List, Tuple, Union, Optional
import os
import json
from collections import OrderedDict
import abc
import sys
import warnings
import itertools
from typing import NewType
import sacremoses
from uuid import uuid4
from .vocab import Vocab
from ..registry import TOKENIZER_REGISTRY
from ..utils.lazy_imports import try_import_subword_nmt, \
    try_import_sentencepiece, \
    try_import_huggingface_tokenizers, \
    try_import_yttm, \
    try_import_spacy, \
    try_import_jieba

SentencesType = NewType('SentencesType', Union[str, List[str]])
TokensType = NewType('TokensType', Union[List[str], List[List[str]]])
TokenIDsType = NewType('TokenIDsType', Union[List[int], List[List[int]]])
TokenOffsetsType = NewType('TokenOffsetsType', Union[List[Tuple[int, int]],
                                                     List[List[Tuple[int, int]]]])


def _encode_no_vocab_err_msg():
    return 'There is no vocab bound to the tokenizer. ' \
           'Must set vocab if the output_type is "int". You may use' \
           ' `tokenizer.set_vocab(vocab)` to attach the vocabulary.'


def _decode_no_vocab_err_msg():
    return 'Decode has "int" as the input token type. You must specify the ' \
           'vocabulary in order to decode from integers. ' \
           'You can use `tokenizer.set_vocab(vocab)`' \
           ' to attach the vocabulary.'


def _token_type_unsupported_err_msg(token_type):
    return 'The token type is not supported, we only support ' \
           '"str" and "int" as the inner token types. Received type(token)="{}"'.format(token_type)


def _is_tokens_from_multiple_sentences(tokens: Union[TokensType, TokenIDsType]) -> bool:
    """Return True if the input is List[List[Any]]"""
    return len(tokens) > 0 and isinstance(tokens[0], list)


def _get_token_type(tokens: Union[List[str], List[int], List[List[str]],
                                  List[List[int]]]) -> type:
    """

    Parameters
    ----------
    tokens
        The input tokens.

    Returns
    -------
    token_type
        If the tokens is empty, return `str`.
        Otherwise, return `str` if the input is str and `int` if the input is int.
    """
    if len(tokens) == 0:
        return str
    if isinstance(tokens[0], int):
        return int
    elif isinstance(tokens[0], str):
        return str
    elif isinstance(tokens[0], list):
        flatten_tokens_it = itertools.chain.from_iterable(tokens)
        try:
            first_token = next(flatten_tokens_it)
            return type(first_token)
        except StopIteration:
            return str
    else:
        raise ValueError(_token_type_unsupported_err_msg(type(tokens[0])))


def _get_vocab(vocab: Union[str, Vocab]) -> Vocab:
    if isinstance(vocab, Vocab):
        return vocab
    elif isinstance(vocab, str):
        return Vocab.load(vocab)
    else:
        raise NotImplementedError('Type of the input vocab is not supported. '
                                  'We only support "str" or "Vocab". type(vocab) = "{}".'
                                  .format(type(vocab)))


def _rebuild_offset_from_tokens(sentence: str, tokens: List[str]) \
        -> List[Tuple[int, int]]:
    """Recover the offset of the tokens in the original sentence.

    If you are using a subword tokenizer, make sure to remove the prefix/postfix of the tokens
    before using this function. Also, this does not work for n-gram-based (n>1) subword
    tokenization, i.e.
    it works for "gluonnlp --> gluon + nlp" but not for "gluonnlp --> gl + lu + uo + on + nl + lp"

    Parameters
    ----------
    sentence
        The input sentence
    tokens
        A list of strings that represent the tokenization result

    Returns
    -------
    offsets
        A list of pairs: [(start0, end0), (start1, end1), ...].
        Each pair represents the start and end positions of the token in the original
        sentence.
    """
    running_offset = 0
    ret = []
    for token in tokens:
        token_offset = sentence.index(token, running_offset)
        token_len = len(token)
        running_offset = token_offset + token_len
        ret.append((token_offset, running_offset))
    return ret


def _get_char_offset_from_byte_offset(sentence: str, byte_offsets: List[Tuple[int, int]]):
    # This is the most naive implementation
    byte_offset_to_char_offset = {}
    byte_offset = 0
    for i, ele in enumerate(sentence):
        byte_offset_to_char_offset[byte_offset] = i
        byte_offset += len(ele.encode('utf-8'))
    byte_offset_to_char_offset[byte_offset] = i + 1  # Handle the last sentence
    ret = []
    for ele in byte_offsets:
        ret.append((byte_offset_to_char_offset[ele[0]],
                    byte_offset_to_char_offset[ele[1]]))
    return ret


class BaseTokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, sentences: SentencesType,
               output_type: Union[type, str] = str) \
            -> Union[TokensType, TokenIDsType]:
        """Encode the input sentence(s) into multiple tokens.

        Parameters
        ----------
        sentences
            The sentences to tokenize
        output_type
            The type of the output tokens.
            - str or `token` means each token is represented by its original text.
            - int or `id` means each token is represented by the index in the vocabulary.

        Returns
        -------
        tokens
            The output tokens.
        """
        pass

    @abc.abstractmethod
    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        """Detokenize a sequence/multiple sequences of tokens to a single sentence/multiple
         sentences.

        Parameters
        ----------
        tokens
            The input tokens to decode

        Returns
        -------
        sentences
            The detokenized sentence(s)
        """
        pass

    def encode_with_offsets(self, sentences: SentencesType,
                            output_type: type = str) \
            -> Tuple[Union[TokensType, TokenIDsType], TokenOffsetsType]:
        """Encode the input sentence(s) into multiple tokens. Different from encode, it
        will also return the character start and end positions of each token in the original text.
        The original text is assumed to be

        Here, the default implementation is to use the tokenized result to recover the offsets.

        Parameters
        ----------
        sentences
            The sentence(s) to tokenize
        output_type
            The type of the output tokens.
            - `str` means each token is represented by its original text.
            - `int` means each token is represented by the index in the vocabulary.

        Returns
        -------
        tokens
            The output tokens.
        offsets
            The offsets of these tokens. Each encodes the start and end location in the original
            unicode string. We return the character-offset instead of the byte-offset.
        """
        raise NotImplementedError


class BaseTokenizerWithVocab(BaseTokenizer):
    @property
    @abc.abstractmethod
    def vocab(self) -> Optional[Vocab]:
        """Get the vocab of the tokenizer

        Returns
        -------
        vocab
            The vocab of the tokenizer
        """
        pass

    @abc.abstractmethod
    def set_vocab(self, vocab: Vocab):
        pass


def load_tokenizer(method, **kwargs):
    if method == 'whitespace':
        return WhitespaceTokenizer()
    elif method == 'moses':
        return MosesTokenizer(**kwargs)


@TOKENIZER_REGISTRY.register('whitespace')
class WhitespaceTokenizer(BaseTokenizerWithVocab):
    def __init__(self, vocab: Optional[Vocab] = None):
        self._vocab = vocab

    def encode(self, sentences, output_type=str):
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        if output_type is str:
            tokens = [sentence.split() for sentence in sentences]
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            tokens = [self._vocab[sentence.split()] for sentence in sentences]
        else:
            raise NotImplementedError
        if is_multiple_sentences:
            return tokens
        else:
            return tokens[0]

    def encode_with_offsets(self, sentences, output_type=str):
        if output_type is int and (not hasattr(self, 'vocab') or self.vocab is None):
            raise ValueError(_encode_no_vocab_err_msg())
        if output_type not in [int, str]:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        all_tokens = self.encode(sentences, output_type=str)
        offsets = []
        for ele_tokens, ele_sentence in zip(all_tokens, sentences):
            ele_offsets = _rebuild_offset_from_tokens(ele_sentence, ele_tokens)
            offsets.append(ele_offsets)
        if is_multiple_sentences:
            return all_tokens, offsets
        else:
            return all_tokens[0], offsets[0]

    def decode(self, tokens):
        is_multiple_sentences = _is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = _get_token_type(tokens)
        if token_type is str:
            ret = [' '.join(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise ValueError(_decode_no_vocab_err_msg())
            ret = [' '.join(self._vocab.to_tokens(ele_tokens)) for ele_tokens in tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        """Set the vocabulary of the tokenizer

        Parameters
        ----------
        vocab
        """
        self._vocab = vocab


@TOKENIZER_REGISTRY.register('spacy')
class SpacyTokenizer(BaseTokenizerWithVocab):
    r"""Apply the Spacy Tokenizer.

    Users of this class are required to install `spaCy <https://spacy.io/usage/>`_
    and download corresponding NLP models, such as :samp:`python -m spacy download en`.

    Only spacy>=2.0.0 is supported.

    Parameters
    ----------
    lang
        The language of the input. If we just specify the lang and do not specify the model,
        we will provide the tokenizer with pre-selected models.
    model
        The language to tokenize. Default is 'en_core_web_sm', i.e, English.
        You may refer to https://spacy.io/usage/models for supported languages.
    vocab
        The vocabulary of the tokenizer. Can be optional.

    Examples
    --------
    >>> import gluonnlp
    >>> tokenizer = gluonnlp.data.SpacyTokenizer()
    >>> tokenizer.encode('Gluon NLP toolkit provides a suite of text processing tools.')
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer('de')
    >>> tokenizer.encode('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools'
    ...                  ' zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verfügung', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer(model='de_core_news_sm')
    >>> tokenizer.encode('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools'
    ...                  ' zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verfügung', '.']
    """

    def __init__(self, lang: Optional[str] = 'en', model: Optional[str] = None,
                 vocab: Optional[Vocab] = None):
        self._vocab = vocab
        spacy = try_import_spacy()
        if model is None:
            assert lang is not None
            if lang == 'en':
                model = 'en_core_web_sm'
            elif lang == 'de':
                model = 'de_core_news_sm'
            elif lang == 'fr':
                model = 'fr_core_news_sm'
            else:
                model = 'xx_ent_wiki_sm'
        retries = 5
        try:
            self._nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])
        except Exception:
            from spacy.cli import download
            while retries >= 0:
                try:
                    download(model, False, '--user')
                    self._nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])
                    break
                except Exception as download_err:
                    retries -= 1
                    if retries < 0:
                        print('SpaCy Model for the specified model="{model}" has not been '
                              'successfully loaded. You need to check the installation guide in '
                              'https://spacy.io/usage/models. Usually, the installation command '
                              'should be `python -m spacy download {model}`.\n'
                              'Complete Error Message: {err_msg}'.format(model=model,
                                                                        err_msg=str(download_err)))
                        raise

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [[tok.text for tok in self._nlp(sentence)] for sentence in sentences]
            else:
                return [tok.text for tok in self._nlp(sentences)]
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            tokens = self.encode(sentences, str)
            if isinstance(sentences, list):
                return [self._vocab[ele_tokens] for ele_tokens in tokens]
            else:
                return [self._vocab[tokens]]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))

    def encode_with_offsets(self, sentences: SentencesType, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        all_tokens = [self._nlp(sentence) for sentence in sentences]
        offsets = [[(tok.idx, tok.idx + len(tok.text)) for tok in tokens]
                   for tokens in all_tokens]
        if output_type is str:
            out_tokens = [[tok.text for tok in tokens] for tokens in all_tokens]
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            out_tokens = [self._vocab[[tok.text for tok in tokens]] for tokens in all_tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multi_sentences:
            return out_tokens, offsets
        else:
            return out_tokens[0], offsets[0]

    def decode(self, tokens):
        raise NotImplementedError(
            'We decide not to implement the decode feature for SpacyTokenizer'
            ' because detokenization is not well-supported by'
            ' spacy. For more details, you may refer to the stack-overflow discussion:'
            ' https://stackoverflow.com/questions/50330455/how-to-detokenize-spacy-text-without-doc-context. '
            'Also, we welcome your contribution for an approximate detokenizer for SpaCy.')

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        """Set the vocabulary of the tokenizer

        Parameters
        ----------
        vocab
            Update the inner vocabulary of the tokenizer to the given vocabulary.
        """
        self._vocab = vocab


@TOKENIZER_REGISTRY.register('moses')
class MosesTokenizer(BaseTokenizerWithVocab):
    r"""Apply the Moses Tokenizer/Detokenizer implemented in
     [sacremoses](https://github.com/alvations/sacremoses).

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Parameters
    ----------
    lang
        The language of the input.
    """

    def __init__(self, lang: str = 'en', vocab: Optional[Vocab] = None):
        self._lang = lang
        self._vocab = vocab
        if lang == 'zh':
            warnings.warn('You may not use MosesTokenizer for Chinese sentences because it is '
                          'not accurate. Try to use JiebaTokenizer. You may also tokenize the '
                          'chinese sentence to characters and learn a BPE.')
        self._tokenizer = sacremoses.MosesTokenizer(lang=lang)
        self._detokenizer = sacremoses.MosesDetokenizer(lang=lang)

        # Here, we need to warm-up the tokenizer to compile the regex
        # This will boost the performance in MacOS
        # For benchmarking results, see
        # https://gist.github.com/sxjscience/f59d2b88262fefd4fb08565c9dec6099
        self._warmup()

    def _warmup(self):
        _ = self.encode('hello world')
        _ = self.decode(['hello', 'world'])

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [self._tokenizer.tokenize(sentence, return_str=False)
                        for sentence in sentences]
            else:
                return self._tokenizer.tokenize(sentences, return_str=False)
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            tokens = self.encode(sentences, str)
            if isinstance(sentences, list):
                return [self._vocab[ele_tokens] for ele_tokens in tokens]
            else:
                return self._vocab[tokens]
        else:
            raise NotImplementedError

    def encode_with_offsets(self, sentences, output_type=str):
        raise NotImplementedError('We cannot obtain the original offsets for MosesTokenizer.')

    def decode(self, tokens):
        is_multiple_sentences = _is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = _get_token_type(tokens)
        if token_type is str:
            ret = [self._detokenizer.detokenize(ele_tokens, return_str=True)
                   for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise ValueError(_decode_no_vocab_err_msg())
            ret = [self._detokenizer.detokenize(self._vocab.to_tokens(ele_tokens), return_str=True)
                   for ele_tokens in tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        self._vocab = vocab


@TOKENIZER_REGISTRY.register('jieba')
class JiebaTokenizer(BaseTokenizerWithVocab):
    r"""Apply the jieba tokenizer to tokenize Chinese sentences.

    For more details, you may refer to [jieba](https://github.com/fxsjy/jieba)

    """

    def __init__(self, ditionary=None, vocab: Optional[Vocab] = None):
        self._vocab = vocab
        jieba = try_import_jieba()
        self._tokenizer = jieba.Tokenizer(ditionary)
        self._tokenizer.initialize(self._tokenizer.dictionary)

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [list(self._tokenizer.cut(sentence)) for sentence in sentences]
            else:
                return list(self._tokenizer.cut(sentences))
        elif output_type is int or output_type == 'id':
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            if isinstance(sentences, list):
                return [[self._vocab[ele] for ele in self._tokenizer.cut(sentence)]
                        for sentence in sentences]
            else:
                return [self._vocab[ele] for ele in self._tokenizer.cut(sentences)]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))

    def encode_with_offsets(self, sentences, output_type=str):
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        all_tokens = [list(self._tokenizer.tokenize(sentence)) for sentence in sentences]
        offsests = [[(ele[1], ele[2]) for ele in tokens] for tokens in all_tokens]
        if output_type is str:
            ret_tokens = [[ele[0] for ele in tokens] for tokens in all_tokens]
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            ret_tokens = [self._vocab[[ele[0] for ele in tokens]] for tokens in all_tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multiple_sentences:
            return ret_tokens, offsests
        else:
            return ret_tokens[0], offsests[0]

    def decode(self, tokens):
        is_multiple_sentences = _is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = _get_token_type(tokens)
        if token_type is str:
            ret = [''.join(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise ValueError(_decode_no_vocab_err_msg())
            ret = [''.join(self._vocab.to_tokens(ele_tokens)) for ele_tokens in tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        self._vocab = vocab

    def __getstate__(self):
        """Make the JiebaTokenizer pickleble. It is safe to remove the lock."""
        d = {k: v for k, v in self._tokenizer.__dict__.items() if k != 'lock'}
        return d

    def __setstate__(self, state):
        self._tokenizer = jieba.Tokenizer()
        for k, v in state.items():
            setattr(self._tokenizer, k, v)


@TOKENIZER_REGISTRY.register('subword_nmt')
class SubwordNMTTokenizer(BaseTokenizerWithVocab):
    def __init__(self, codec_path, vocab_path: Optional[Union[str, Vocab]] = None,
                 separator: str = '@@', bpe_dropout: float = 0.0,
                 suffix: str = '</w>'):
        """

        Parameters
        ----------
        codec_path
        vocab_path
        separator
        bpe_dropout
        """
        try_import_subword_nmt()
        from subword_nmt.apply_bpe import BPE
        self._codec_path = codec_path
        self._vocab = _get_vocab(vocab_path)
        self._separator = separator
        self._bpe_dropout = bpe_dropout
        self._suffix = suffix
        with open(self._codec_path, 'r', encoding='utf-8') as merge_codes:
            self._bpe = BPE(codes=merge_codes, separator=self._separator)
        self._last_subword_id_set = frozenset([self._vocab[ele]
                                               for ele in self._vocab.all_tokens
                                               if not ele.endswith(self._separator)])

    def transform_sentence(self, sentence):
        # replace the separator in encoded result with suffix
        # a@@, b@@, c ->  a, b, c</w>
        return [word[:-2] if len(word) > 2 and word[-2:] == self._separator else word + self._suffix
                for word in sentence]

    def encode(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        if output_type is str:
            ret = [self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))
                   for sentence in sentences]
        elif output_type is int:
            if self._vocab is None:
                raise ValueError(_encode_no_vocab_err_msg())
            ret = [self._vocab[self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))]
                   for sentence in sentences]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        tokens = []
        token_ids = []
        offsets = []
        for sentence in sentences:
            encode_token = self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))
            encode_id = self._vocab[encode_token]
            encode_token_without_suffix = [x.replace(self._suffix, '') for x in encode_token]
            encode_offset = _rebuild_offset_from_tokens(sentence, encode_token_without_suffix)
            tokens.append(encode_token)
            token_ids.append(encode_id)
            offsets.append(encode_offset)
        if not is_multi_sentences:
            tokens = tokens[0]
            token_ids = token_ids[0]
            offsets = offsets[0]
        if output_type is str:
            return tokens, offsets
        elif output_type is int:
            return token_ids, offsets
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))

    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        is_multiple_sentences = _is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = _get_token_type(tokens)
        if token_type is str:
            ret = [''.join(ele_tokens).replace(self._suffix, ' ').strip()
                   for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise ValueError(_decode_no_vocab_err_msg())
            ret = [''.join(self._vocab.to_tokens(ele_tokens)).replace(self._suffix, ' ').strip()
                   for ele_tokens in tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    def is_last_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the last subword token

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the last subword token in the list of subwords
        """
        if isinstance(tokens, str):
            return not tokens.endswith(self._separator)
        elif isinstance(tokens, int):
            return tokens in self._last_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [not ele.endswith(self._separator) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._last_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def vocab(self) -> Optional[Vocab]:
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        self._vocab = vocab

    def set_bpe_dropout(self, bpe_dropout: float):
        self._bpe_dropout = bpe_dropout

    def __repr__(self):
        ret = '{}(\n' \
              '   codec_path = {}\n' \
              '   separator = {}\n' \
              '   bpe_dropout = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._codec_path),
                         self._separator,
                         self._bpe_dropout,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        from subword_nmt.apply_bpe import BPE
        with open(self._codec_path, 'r', encoding='utf-8') as merge_codes:
            self._bpe = BPE(codes=merge_codes, separator=self._separator)


class HuggingFaceTokenizer(BaseTokenizerWithVocab):
    def encode(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        encode_sentences = self._bpe.encode_batch(sentences)
        if output_type is str:
            ret = [encode_sentence.tokens for encode_sentence in encode_sentences]
        elif output_type is int:
            ret = [encode_sentence.ids for encode_sentence in encode_sentences]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        encode_sentences = self._bpe.encode_batch(sentences)
        if output_type is str:
            ret = [encode_sentence.tokens for encode_sentence in encode_sentences]
            offsets = [encode_sentence.offsets for encode_sentence in encode_sentences]
        elif output_type is int:
            ret = [encode_sentence.ids for encode_sentence in encode_sentences]
            offsets = [encode_sentence.offsets for encode_sentence in encode_sentences]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multi_sentences:
            return ret, offsets
        else:
            return ret[0], offsets[0]

    def decode(self, tokens):
        is_multiple_sentences = _is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = _get_token_type(tokens)
        if token_type is str:
            id_tokens = [[self._bpe.token_to_id(token) for token in sentence] for sentence in
                         tokens]
            ret = self._bpe.decode_batch(id_tokens)
        elif token_type is int:
            ret = self._bpe.decode_batch(tokens)
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab):
        raise NotImplementedError('Cannot set vocabulary for the HuggingFaceTokenizer.')


@TOKENIZER_REGISTRY.register('hf_bpe')
class HuggingFaceBPETokenizer(HuggingFaceTokenizer):
    def __init__(self, merges_file: Optional[str] = None, vocab_file: Optional[str] = None,
                 unk_token: Optional[str] = Vocab.UNK_TOKEN, suffix: Optional[str] = '</w>',
                 dropout: Optional[float] = None, lowercase: bool = False,
                 unicode_normalizer: Optional[str] = None):
        self._merges_file = merges_file
        self._vocab_file = vocab_file
        self._unk_token = unk_token
        self._suffix = suffix
        self._dropout = dropout
        self._lowercase = lowercase
        self._unicode_normalizer = unicode_normalizer
        self.__rebuild_tokenizer()
        self._last_subword_id_set = frozenset([self._vocab[ele]
                                               for ele in self._vocab.all_tokens
                                               if ele.endswith(self._suffix)])

    def is_last_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the last subword token

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the last subword token in the list of subwords
        """
        if isinstance(tokens, str):
            return tokens.endswith(self._suffix)
        elif isinstance(tokens, int):
            return tokens in self._last_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [ele.endswith(self._suffix) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._last_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def set_bpe_dropout(self, bpe_dropout: float):
        self._dropout = bpe_dropout
        self.__rebuild_tokenizer()

    def set_lowercase(self, lowercase: float):
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # build vocab and temp_hf_vocab_file
        try:
            # using Vocab obj file
            self._vocab = _get_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
            hf_vocab = OrderedDict()
            for i in range(len(all_tokens)):
                hf_vocab[all_tokens[i]] = i
            temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
            with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
                json.dump(hf_vocab, ftv, ensure_ascii=False)
        except TypeError:
            # using hf_bpe vocab file
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                hf_vocab = json.load(fv)
            hf_vocab = sorted(list(hf_vocab.items()), key=lambda x: x[1])
            all_tokens = [x[0] for x in hf_vocab]
            # defualt special tokens corresponding to the default
            # special_tokens setting in CharBPETokenizer.train
            # and the default special_tokens=[unk]
            self._vocab = Vocab(all_tokens, unk_token=self._unk_token)
            temp_hf_vocab_file = None
        assert self._unk_token == self._vocab.unk_token
        self._bpe = tokenizers.CharBPETokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            merges_file=self._merges_file,
            unk_token=self._unk_token, suffix=self._suffix, dropout=self._dropout,
            lowercase=self._lowercase, unicode_normalizer=self._unicode_normalizer)
        if temp_hf_vocab_file:
            os.remove(temp_hf_vocab_file)

    def __repr__(self):
        ret = '{}(\n' \
              '   merges_file = {}\n' \
              '   vocab_file = {}\n' \
              '   unk_token = {}, suffix = {}\n' \
              '   dropout = {}, lowercase = {}\n' \
              '   unicode_normalizer = {}' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._merges_file),
                         os.path.realpath(self._vocab_file),
                         self._unk_token, self._suffix,
                         self._dropout, self._lowercase,
                         self._unicode_normalizer,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()


@TOKENIZER_REGISTRY.register('hf_bytebpe')
class HuggingFaceByteBPETokenizer(HuggingFaceTokenizer):
    def __init__(self, merges_file: Optional[str] = None, vocab_file: Optional[str] = None,
                 add_prefix_space: bool = False, lowercase: bool = False,
                 dropout: Optional[float] = None,
                 unicode_normalizer: Optional[str] = None,
                 continuing_subword_prefix: Optional[str] = None,
                 end_of_word_suffix: Optional[str] = None, trim_offsets: bool = False):
        self._merges_file = merges_file
        self._vocab_file = vocab_file
        self._add_prefix_space = add_prefix_space
        self._lowercase = lowercase
        self._dropout = dropout
        self._unicode_normalizer = unicode_normalizer
        self._continuing_subword_prefix = continuing_subword_prefix
        self._end_of_word_suffix = end_of_word_suffix
        self._trim_offsets = trim_offsets
        self.__rebuild_tokenizer()

    def set_bpe_dropout(self, bpe_dropout: float):
        self._dropout = bpe_dropout
        self.__rebuild_tokenizer()

    def set_lowercase(self, lowercase: float):
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # build vocab and temp_hf_vocab_file
        try:
            # using Vocab obj file
            self._vocab = _get_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
            hf_vocab = OrderedDict()
            for i in range(len(all_tokens)):
                hf_vocab[all_tokens[i]] = i
            temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
            with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
                json.dump(hf_vocab, ftv, ensure_ascii=False)
        except TypeError:
            # using hf_bytebpe vocab file
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                hf_vocab = json.load(fv)
            hf_vocab = sorted(list(hf_vocab.items()), key=lambda x: x[1])
            all_tokens = [x[0] for x in hf_vocab]
            # defualt special tokens corresponding to the default
            # special_tokens setting in ByteBPETokenizer.train
            # and the default special_tokens=[]
            self._vocab = Vocab(all_tokens)
            temp_hf_vocab_file = None
        self._bpe = tokenizers.ByteLevelBPETokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            merges_file=self._merges_file,
            add_prefix_space=self._add_prefix_space, lowercase=self._lowercase,
            dropout=self._dropout, unicode_normalizer=self._unicode_normalizer,
            continuing_subword_prefix=self._continuing_subword_prefix,
            end_of_word_suffix=self._end_of_word_suffix,
            trim_offsets=self._trim_offsets)
        if temp_hf_vocab_file:
            os.remove(temp_hf_vocab_file)

    def __repr__(self):
        ret = '{}(\n' \
              '   merges_file = {}\n' \
              '   vocab_file = {}\n' \
              '   add_prefix_space = {}, lowercase = {}, dropout = {}\n' \
              '   unicode_normalizer = {}, continuing_subword_prefix = {}\n' \
              '   end_of_word_suffix = {}\n' \
              '   trim_offsets = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._merges_file),
                         os.path.realpath(self._vocab_file),
                         self._add_prefix_space, self._lowercase, self._dropout,
                         self._unicode_normalizer, self._continuing_subword_prefix,
                         self._end_of_word_suffix,
                         self._trim_offsets,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()


@TOKENIZER_REGISTRY.register('hf_wordpiece')
class HuggingFaceWordPieceTokenizer(HuggingFaceTokenizer):
    def __init__(self, vocab_file: Optional[str] = None,
                 unk_token: str = Vocab.UNK_TOKEN,
                 sep_token: str = Vocab.SEP_TOKEN,
                 cls_token: str = Vocab.CLS_TOKEN,
                 pad_token: str = Vocab.PAD_TOKEN,
                 mask_token: str = Vocab.MASK_TOKEN,
                 clean_text: bool = True, handle_chinese_chars: bool = True,
                 strip_accents: bool = False, lowercase: bool = False,
                 wordpieces_prefix: str = "##"):
        self._vocab_file = vocab_file
        self._unk_token = unk_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._clean_text = clean_text
        self._handle_chinese_chars = handle_chinese_chars
        self._strip_accents = strip_accents
        self._lowercase = lowercase
        self._wordpieces_prefix = wordpieces_prefix
        self.__rebuild_tokenizer()
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in self._vocab.all_tokens
                                                if not ele.startswith(self._wordpieces_prefix) and
                                                not ele in [self._sep_token, self._cls_token]])

    def encode(self, sentences, output_type=str):
        """
        remove the cls and sep tokens of original huggingface wordpiece encoding
        """
        is_multi_sentences = isinstance(sentences, list)
        ret = HuggingFaceTokenizer.encode(self, sentences, output_type)
        if not is_multi_sentences:
            ret = [ret]
        if output_type == str:
            cls_token, sep_token = self._vocab.cls_token, self._vocab.sep_token
            ret = [x[1:-1] if x[0] == cls_token and x[-1] == sep_token else x
                   for x in ret]
        else:
            cls_id, sep_id = self._vocab.cls_id, self._vocab.sep_id
            ret = [x[1:-1] if x[0] == cls_id and x[-1] == sep_id else x
                   for x in ret]
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        """
        remove the cls and sep tokens of original huggingface wordpiece encoding
        """
        is_multi_sentences = isinstance(sentences, list)
        ret, offsets = HuggingFaceTokenizer.encode_with_offsets(self, sentences, output_type)
        if not is_multi_sentences:
            ret, offsets = [ret], [offsets]
        if output_type == str:
            cls_token, sep_token = self._vocab.cls_token, self._vocab.sep_token
            for i in range(len(ret)):
                if ret[i][0] == cls_token and ret[i][-1] == sep_token:
                    ret[i], offsets[i] = ret[i][1:-1], offsets[i][1:-1]
        else:
            cls_id, sep_id = self._vocab.cls_id, self._vocab.sep_id
            for i in range(len(ret)):
                if ret[i][0] == cls_id and ret[i][-1] == sep_id:
                    ret[i], offsets[i] = ret[i][1:-1], offsets[i][1:-1]
        if is_multi_sentences:
            return ret, offsets
        else:
            return ret[0], offsets[0]

    def is_first_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        if isinstance(tokens, str):
            return not tokens.startswith(self._wordpieces_prefix) and not tokens in [
                self._cls_token, self._sep_token]
        elif isinstance(tokens, int):
            return tokens in self._first_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [not ele.startswith(self._wordpieces_prefix) and not ele in [self._cls_token,
                                                                                    self._sep_token]
                        for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._first_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def set_lowercase(self, lowercase: float):
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # build vocab and temp_hf_vocab_file
        try:
            # using Vocab obj file
            self._vocab = _get_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
        except json.JSONDecodeError:
            # using hf_wordpiece vocab file
            all_tokens = []
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                for line in fv:
                    all_tokens.append(line.strip())
            # defualt special tokens corresponding to the default
            # special_tokens setting in BertWordPieceTokenizer.train
            # and the default special_tokens=[pad, unk, cls, sep, mask]
            default_special_tokens = {'pad_token': self._pad_token,
                                      'cls_token': self._cls_token,
                                      'sep_token': self._sep_token,
                                      'mask_token': self._mask_token}
            self._vocab = Vocab(all_tokens, unk_token=self._unk_token, **default_special_tokens)
            all_tokens = self._vocab.all_tokens
        # for safety, also use temp file when using wordpiece vocab file
        # for situation that original all_tokens not cotain special tokens
        # (vocab file of BERT do not contain all special tokens)
        temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
        with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
            ftv.write('\n'.join(all_tokens))
        self._vocab.mask_token_id = self._vocab.mask_id
        assert [self._unk_token, self._sep_token, self._cls_token, self._pad_token,
                self._mask_token] == \
               [self._vocab.unk_token, self._vocab.sep_token, self._vocab.cls_token,
                self._vocab.pad_token, self._vocab.mask_token]
        self._bpe = tokenizers.BertWordPieceTokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            unk_token=self._unk_token,
            sep_token=self._sep_token,
            cls_token=self._cls_token,
            pad_token=self._pad_token,
            mask_token=self._mask_token,
            clean_text=self._clean_text,
            handle_chinese_chars=self._handle_chinese_chars,
            strip_accents=self._strip_accents, lowercase=self._lowercase,
            wordpieces_prefix=self._wordpieces_prefix)
        os.remove(temp_hf_vocab_file)

    def __repr__(self):
        ret = '{}(\n' \
              '   vocab_file = {}\n' \
              '   unk_token = {}, sep_token = {}, cls_token = {}\n' \
              '   pad_token = {}, mask_token = {}\n' \
              '   clean_text = {}, handle_chinese_chars = {}\n' \
              '   strip_accents = {}, lowercase = {}\n' \
              '   wordpieces_prefix = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._vocab_file),
                         self._unk_token, self._sep_token, self._cls_token,
                         self._pad_token, self._mask_token,
                         self._clean_text, self._handle_chinese_chars,
                         self._strip_accents, self._lowercase,
                         self._wordpieces_prefix,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()


@TOKENIZER_REGISTRY.register('spm')
class SentencepieceTokenizer(BaseTokenizerWithVocab):
    r"""Apply the Sentencepiece Tokenizer, which trains subword tokenization via the
    unigram language modeling.

    Users of this class are required to `install sentencepiece
    <https://github.com/google/sentencepiece>`_. For example, one can use
    :samp:`pip install sentencepiece`


    Parameters
    ----------
    model_path
        Path to the pre-trained sentencepiece model.
    vocab
        Path to the vocabulary of the sentencepiece model in GluonNLP
    num_best
        A scalar for sampling subwords. If num_best = {0,1}, no sampling is performed.
        If num_best > 1, then samples from the num_best results.
        If num_best < 0, then assume that num_best is infinite and
        samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling
        algorithm.
    alpha
        A scalar for a smoothing parameter for probability rescaling.
    do_lower
        Whether to convert the input string to lower-case strings
    **kwargs

    Examples
    --------
    >>> from mxnet import gluon
    >>> url = 'https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/tokenizer_test_models/sentencepiece/test_ende-a9bee4.model'
    >>> model_f = gluon.utils.download(url)
    -etc-
    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(model_f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer.encode(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']
    >>> tokenizer.decode(tokenizer.encode(sentence))
    'This is a very awesome, life-changing sentence.'
    >>> os.remove('test_ende-a9bee4.model')

    """

    def __init__(self, model_path: Optional[str] = None,
                 vocab: Optional[Union[str, Vocab]] = None,
                 nbest: int = 0, alpha: float = 0.0, do_lower=False,
                 **kwargs):
        self._model_path = model_path
        sentencepiece = try_import_sentencepiece()
        from ..third_party.sentencepiece_pb2 import SentencePieceText
        self._spt_cls = SentencePieceText
        self._sp_model = sentencepiece.SentencePieceProcessor()
        self._sp_model.load(model_path)
        self._nbest = nbest
        self._alpha = alpha
        self._do_lower = do_lower
        self._meta_symbol = u'▁'
        sp_model_all_tokens = [self._sp_model.id_to_piece(i) for i in range(len(self._sp_model))]
        special_tokens_kv = dict()
        existing_control_token_ids = set()
        token_id_to_token_name = dict()
        if self._sp_model.unk_id() != -1:
            special_tokens_kv['unk_token'] = self._sp_model.id_to_piece(self._sp_model.unk_id())
            existing_control_token_ids.add(self._sp_model.unk_id())
            token_id_to_token_name[self._sp_model.unk_id()] = 'unk_token'
        if self._sp_model.pad_id() != -1:
            special_tokens_kv['pad_token'] = self._sp_model.id_to_piece(self._sp_model.pad_id())
            existing_control_token_ids.add(self._sp_model.pad_id())
            token_id_to_token_name[self._sp_model.pad_id()] = 'pad_token'
        if self._sp_model.bos_id() != -1:
            special_tokens_kv['bos_token'] = self._sp_model.id_to_piece(self._sp_model.bos_id())
            existing_control_token_ids.add(self._sp_model.bos_id())
            token_id_to_token_name[self._sp_model.bos_id()] = 'bos_token'
        if self._sp_model.eos_id() != -1:
            special_tokens_kv['eos_token'] = self._sp_model.id_to_piece(self._sp_model.eos_id())
            existing_control_token_ids.add(self._sp_model.eos_id())
            token_id_to_token_name[self._sp_model.eos_id()] = 'eos_token'
        existing_control_tokens = set([self._sp_model.id_to_piece(ele)
                                       for ele in existing_control_token_ids])
        other_control_tokens_ids = \
            [i for i in range(len(self._sp_model))
             if self._sp_model.is_control(i) and i not in existing_control_token_ids]
        other_control_tokens = set([self._sp_model.id_to_piece(ele)
                                    for ele in other_control_tokens_ids])
        matched_other_control_tokens = dict()
        for k, v in kwargs.items():
            if k in special_tokens_kv:
                if v != special_tokens_kv[k]:
                    raise ValueError(
                        '"vocab.{}" is already set to "{}" in the sentencepiece model. '
                        'Cannot reset it to "{}"'.format(k, special_tokens_kv[k], v))
                continue
            if v in existing_control_tokens:
                if k != token_id_to_token_name[v]:
                    raise ValueError('"{}" is already registered as "vocab.{}". '
                                     'We cannot rename it to "vocab.{}".'
                                     .format(v, token_id_to_token_name[v], k))
                continue
            if v in other_control_tokens:
                if v in matched_other_control_tokens:
                    raise ValueError(
                        '"{}" has already been registered as "vocab.{}", '
                        'we cannot register it again as "vocab.{}".'
                            .format(v, matched_other_control_tokens[v], k))
                matched_other_control_tokens[v] = k
                special_tokens_kv[k] = v
            else:
                raise ValueError('Mismatch vocabulary! All special tokens specified '
                                 'must be control tokens in the sentencepiece vocabulary.')
        if vocab is None:
            if len(matched_other_control_tokens) < len(other_control_tokens):
                for i, token in enumerate(other_control_tokens.difference(
                        set(matched_other_control_tokens.keys()))):
                    token_key = 'other{}_token'.format(i)
                    assert token_key not in special_tokens_kv
                    special_tokens_kv[token_key] = token
            self._vocab = Vocab(sp_model_all_tokens, **special_tokens_kv)
        else:
            self._vocab = _get_vocab(vocab)
        # Sanity check
        assert self._vocab.all_tokens == sp_model_all_tokens
        for token in self._vocab.special_tokens:
            piece_id = self._sp_model.piece_to_id(token)
            if self._sp_model.is_unknown(piece_id):
                assert self._vocab[token] == self._sp_model.unk_id()
            else:
                assert self._sp_model.is_control(piece_id), \
                    'Vocab mismatch! "{}" is a special token in the given vocab but not in the ' \
                    'sentencepiece model!'.format(token)
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in sp_model_all_tokens
                                                if ele.startswith(self._meta_symbol)])

    def encode(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        if self._do_lower:
            sentences = [sentence.lower() for sentence in sentences]
        if output_type is str:
            ret = [self._sp_model.sample_encode_as_pieces(sentence, self._nbest, self._alpha)
                   for sentence in sentences]
        elif output_type is int:
            ret = [self._sp_model.sample_encode_as_ids(sentence, self._nbest, self._alpha)
                   for sentence in sentences]
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def decode(self, tokens):
        is_multi_sentences = _is_tokens_from_multiple_sentences(tokens)
        token_type = _get_token_type(tokens)
        if not is_multi_sentences:
            tokens = [tokens]
        if token_type is str:
            ret = [self._sp_model.decode_pieces(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            ret = [self._sp_model.decode_ids(ele_tokens) for ele_tokens in tokens]
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        tokens = []
        token_ids = []
        offsets = []
        for sentence in sentences:
            if self._do_lower:
                sentence = sentence.lower()
            spt = self._spt_cls()
            spt.ParseFromString(self._sp_model.SampleEncodeAsSerializedProto(
                sentence, self._nbest, self._alpha))
            tokens.append([ele.piece for ele in spt.pieces])
            token_ids.append([ele.id for ele in spt.pieces])
            # In theory, we can recover the character offset from byte offset
            sentence_byte_offsets = [(ele.begin, ele.end) for ele in spt.pieces]
            char_offsets = _get_char_offset_from_byte_offset(sentence, sentence_byte_offsets)
            offsets.append(char_offsets)
        if not is_multi_sentences:
            tokens = tokens[0]
            token_ids = token_ids[0]
            offsets = offsets[0]
        if output_type is str:
            return tokens, offsets
        elif output_type is int:
            return token_ids, offsets
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))

    def is_first_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the first subword token

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the first subword token in the list of subwords
        """
        if isinstance(tokens, str):
            return tokens.startswith(self._meta_symbol)
        elif isinstance(tokens, int):
            return tokens in self._first_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [ele.startswith(self._meta_symbol) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._first_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab):
        raise NotImplementedError('Currently, we cannot set the vocabulary of a '
                                  'SentencepieceTokenizer.')

    @property
    def do_lower(self):
        return self._do_lower

    def set_subword_regularization(self, nbest, alpha):
        self._nbest = nbest
        self._alpha = alpha

    def __repr__(self):
        ret = '{}(\n' \
              '   model_path = {}\n' \
              '   do_lower = {}, nbest = {}, alpha = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._model_path),
                         self._do_lower, self._nbest, self._alpha,
                         self._vocab)
        return ret

    def __getstate__(self):
        """Make the SentencepieceTokenizer pickleble.
         We will remove the _spt_cls and _sp_model, which are not picklable, and try to
         reconstruct the class via the saved model_path. This behavior is only acceptable for
         multiprocessing and should not be used to save sentencepiece models."""
        state = self.__dict__.copy()
        state['_spt_cls'] = None
        state['_sp_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        sentencepiece = try_import_sentencepiece()
        from ..third_party.sentencepiece_pb2 import SentencePieceText
        self._spt_cls = SentencePieceText
        self._sp_model = sentencepiece.SentencePieceProcessor()
        ret = self._sp_model.load(self._model_path)
        assert ret is True, 'Cannot load data from the saved seralized protobuffer!'


@TOKENIZER_REGISTRY.register('yttm')
class YTTMTokenizer(BaseTokenizerWithVocab):
    def __init__(self, model_path: str, bpe_dropout: float = 0.0, n_threads: int = -1):
        """

        Parameters
        ----------
        model_path
        bpe_dropout
            The dropout probability in BPE-Dropout:
                "BPE-Dropout: Simple and Effective Subword Regularization"
        n_threads
            The number of threads for encoding
        """
        yttm = try_import_yttm()
        self._model_path = model_path
        self._bpe = yttm.BPE(model=model_path, n_threads=n_threads)
        self._bpe_dropout = bpe_dropout
        self._out_type = yttm.OutputType
        all_tokens = self._bpe.vocab()
        self._vocab = Vocab(all_tokens,
                            unk_token='<UNK>', pad_token='<PAD>',
                            bos_token='<BOS>', eos_token='<EOS>')
        self._meta_symbol = u'▁'  # U+2581 as the symbol for the first subword token
        if len(self._vocab) != len(all_tokens):
            raise ValueError('Cannot load the trained YTTM model file!')
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in self._vocab.all_tokens
                                                if ele.startswith(self._meta_symbol)])

    def encode(self, sentences, output_type=str):
        is_single_sentence = not isinstance(sentences, list)
        if is_single_sentence:
            sentences = [sentences]
        if output_type is str:
            tokens = self._bpe.encode(sentences, output_type=self._out_type.SUBWORD,
                                      dropout_prob=self._bpe_dropout)
        elif output_type is int:
            tokens = self._bpe.encode(sentences, output_type=self._out_type.ID,
                                      dropout_prob=self._bpe_dropout)
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))
        if is_single_sentence:
            return tokens[0]
        else:
            return tokens

    def decode(self, tokens):
        is_multi_sentences = _is_tokens_from_multiple_sentences(tokens)
        token_type = _get_token_type(tokens)
        if not is_multi_sentences:
            tokens = [tokens]
        if token_type is int:
            ret = self._bpe.decode(tokens)
        elif token_type is str:
            ret = []
            for ele_tokens in tokens:
                sentence = ''.join(ele_tokens)
                if sentence[0] == self._meta_symbol:
                    sentence = sentence[1:]
                sentence = sentence.replace(self._meta_symbol, ' ')
                ret.append(sentence)
        else:
            raise ValueError(_token_type_unsupported_err_msg(token_type))
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        tokens = []
        token_ids = []
        offsets = []
        for sentence in sentences:
            encode_token = self._bpe.encode([sentence],
                                            output_type=self._out_type.SUBWORD,
                                            dropout_prob=self._bpe_dropout)[0]
            encode_id = self._bpe.encode([sentence],
                                         output_type=self._out_type.ID,
                                         dropout_prob=self._bpe_dropout)[0]
            encode_token_without_meta_symbol = [x.replace(self._meta_symbol, ' ')
                                                for x in encode_token]
            if len(encode_token_without_meta_symbol) > 0:
                encode_token_without_meta_symbol[0] = \
                    encode_token_without_meta_symbol[0].replace(' ', '')
            encode_offset = _rebuild_offset_from_tokens(sentence, encode_token_without_meta_symbol)
            tokens.append(encode_token)
            token_ids.append(encode_id)
            offsets.append(encode_offset)
        if not is_multi_sentences:
            tokens = tokens[0]
            token_ids = token_ids[0]
            offsets = offsets[0]
        if output_type is str:
            return tokens, offsets
        elif output_type is int:
            return token_ids, offsets
        else:
            raise ValueError(_token_type_unsupported_err_msg(output_type))

    def is_first_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the first subword token

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the first subword token in the list of subwords
        """
        if isinstance(tokens, str):
            return tokens.startswith(self._meta_symbol)
        elif isinstance(tokens, int):
            return tokens in self._first_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [ele.startswith(self._meta_symbol) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._first_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        raise NotImplementedError('Cannot set vocabulary for the YTTMTokenizer.')

    def set_bpe_dropout(self, bpe_dropout: float):
        """Set the bpe dropout probability

        Parameters
        ----------
        bpe_dropout
            The dropout ratio for BPE Dropout
        """
        self._bpe_dropout = bpe_dropout

    def __repr__(self):
        ret = '{}(\n' \
              '   model_path = {}\n' \
              '   bpe_dropout = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._model_path),
                         self._bpe_dropout,
                         self._vocab)
        return ret

    def __getstate__(self):
        """Support multiprocessing by making it pickleble"""
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        yttm = try_import_yttm()
        self._bpe = yttm.BPE(self._model_path)


def create(name: str, *args, **kwargs) -> BaseTokenizer:
    """

    Parameters
    ----------
    name
    args
    kwargs

    Returns
    -------
    tokenizer
    """
    return TOKENIZER_REGISTRY.create(name, *args, **kwargs)


def create_with_json(name: str, json_str: str) -> BaseTokenizer:
    """

    Parameters
    ----------
    name
    json_str

    Returns
    -------
    tokenizer
    """
    return TOKENIZER_REGISTRY.create_with_json(name, json_str)


def list_all():
    return TOKENIZER_REGISTRY.list_keys()
