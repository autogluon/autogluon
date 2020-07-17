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
"""Lazy import some third-party libraries."""
__all__ = ['try_import_sentencepiece',
           'try_import_yttm',
           'try_import_subword_nmt',
           'try_import_huggingface_tokenizers',
           'try_import_spacy',
           'try_import_scipy',
           'try_import_mwparserfromhell',
           'try_import_fasttext',
           'try_import_langid',
           'try_import_boto3',
           'try_import_jieba']


def try_import_sentencepiece():
    try:
        import sentencepiece  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            'sentencepiece is not installed. You must install sentencepiece '
            'in order to use the Sentencepiece tokenizer. '
            'You can refer to the official installation guide '
            'in https://github.com/google/sentencepiece#installation')
    return sentencepiece


def try_import_yttm():
    try:
        import youtokentome as yttm
    except ImportError:
        raise ImportError('YouTokenToMe is not installed. You may try to install it via '
                          '`pip install youtokentome`.')
    return yttm


def try_import_subword_nmt():
    try:
        import subword_nmt
    except ImportError:
        raise ImportError('subword-nmt is not installed. You can run `pip install subword_nmt` '
                          'to install the subword-nmt BPE implementation. You may also '
                          'refer to the official installation guide in '
                          'https://github.com/rsennrich/subword-nmt.')
    return subword_nmt


def try_import_huggingface_tokenizers():
    try:
        import tokenizers
    except ImportError:
        raise ImportError(
            'HuggingFace tokenizers is not installed. You can run `pip install tokenizers` '
            'to use the HuggingFace BPE tokenizer. You may refer to the official installation '
            'guide in https://github.com/huggingface/tokenizers.')
    return tokenizers


def try_import_spacy():
    try:
        import spacy  # pylint: disable=import-outside-toplevel
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        assert parse_version(spacy.__version__) >= parse_version('2.0.0'), \
            'We only support spacy>=2.0.0'
    except ImportError:
        raise ImportError(
            'spaCy is not installed. You must install spaCy in order to use the '
            'SpacyTokenizer. You can refer to the official installation guide '
            'in https://spacy.io/usage/.')
    return spacy


def try_import_scipy():
    try:
        import scipy
    except ImportError:
        raise ImportError('SciPy is not installed. '
                          'You must install SciPy >= 1.0.0 in order to use the '
                          'TruncNorm. You can refer to the official '
                          'installation guide in https://www.scipy.org/install.html .')
    return scipy


def try_import_mwparserfromhell():
    try:
        import mwparserfromhell
    except ImportError:
        raise ImportError('mwparserfromhell is not installed. You must install '
                          'mwparserfromhell in order to run the script. You can use '
                          '`pip install mwparserfromhell` or refer to guide in '
                          'https://github.com/earwig/mwparserfromhell.')
    return mwparserfromhell


def try_import_autogluon():
    try:
        import autogluon
    except ImportError:
        raise ImportError('AutoGluon is not installed. You must install autogluon in order to use '
                          'the functionality. You can follow the guide in '
                          'https://github.com/awslabs/autogluon for installation.')
    return autogluon


def try_import_fasttext():
    try:
        import fasttext
    except ImportError:
        raise ImportError('FastText is not installed. You must install fasttext in order to use the'
                          ' functionality. See https://github.com/facebookresearch/fastText for '
                          'more information.')
    return fasttext


def try_import_langid():
    try:
        import langid
    except ImportError:
        raise ImportError('"langid" is not installed. You must install langid in order to use the'
                          ' functionality. You may try to use `pip install langid`.')
    return langid


def try_import_boto3():
    try:
        import boto3
    except ImportError:
        raise ImportError('"boto3" is not installed. To enable fast downloading in EC2. You should '
                          'install boto3 and correctly configure the S3. '
                          'See https://boto3.readthedocs.io/ for more information. '
                          'If you are using EC2, downloading from s3:// will '
                          'be multiple times faster than using the traditional http/https URL.')
    return boto3


def try_import_jieba():
    try:
        import jieba
    except ImportError:
        raise ImportError('"jieba" is not installed. You must install jieba tokenizer. '
                          'You may try to use `pip install jieba`')
    return jieba
