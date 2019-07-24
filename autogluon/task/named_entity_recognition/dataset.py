"""Some methods are modified from below link
https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/data/ner.py"""

import os
import re
from multiprocessing import cpu_count
from typing import AnyStr
import logging
import numpy as np
import ConfigSpace as CS
import gluonnlp as nlp
from mxnet import gluon
from mxnet.gluon.utils import download

from ... import dataset
from ...space import Exponential
from .utils import load_segment

LOG = logging.getLogger(__name__)

__all__ = ['Dataset']
NULL_TAG = "X"


class Dataset(dataset.Dataset):
    """
    Named Entity Recognition Dataset class
    """
    def __init__(self, name: AnyStr = None, train_path: AnyStr = None, val_path: AnyStr = None,
                 lazy: bool = True, vocab: nlp.Vocab = None, max_sequence_length: int = None,
                 tokenizer: nlp.data.transforms = None, indexes_format: dict = None,
                 batch_size: int = 8):
        super(Dataset, self).__init__(name, train_path, val_path)
        # TODO : add search space, handle batch_size, num_workers
        self._num_classes: int = 0
        self._vocab: nlp.Vocab = vocab
        self.pretrained_dataset_name = 'book_corpus_wiki_en_cased'
        self._train_ds_transformed = None
        self._val_ds_transformed = None
        self.tag_vocab = None
        self.null_tag_index = None
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length  # TODO This should come from config
        self.tokenizer = tokenizer
        self.indexes_format = indexes_format
        self.train_dataloader = None
        self.val_dataloader = None
        self._download_dataset()
        self.add_search_space()

        if self._vocab is None:
            _, self._vocab = nlp.model.get_model(name='bert_12_768_12',
                                                 dataset_name=self.pretrained_dataset_name)
            if self.pretrained_dataset_name == 'book_corpus_wiki_en_cased':
                LOG.info("Taking default vocabulary of pre-trained model `bert_12_768_12` "
                         "on `book_corpus_wiki_en_cased`. If you would like to use different"
                         "vocabulary then please pass parameter `vocab` as an instance of"
                         "`nlp.Vocab`.")

        # if not lazy:
        if not lazy:
            self._init_()

    def _init_(self):
        self._read_dataset()
        self.add_search_space()

    def add_search_space(self):
        exp = int(np.log2(self.batch_size))
        cs = CS.ConfigurationSpace()
        data_hyperparams = Exponential(name='batch_size', base=2, lower_exponent=exp,
                                       upper_exponent=exp).get_hyper_param()
        cs.add_hyperparameter(data_hyperparams)
        self._set_search_space(cs)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def vocab(self) -> nlp.vocab:
        return self._vocab

    @vocab.setter
    def vocab(self, value: nlp.vocab) -> None:
        self._vocab = value
        self._init_()

    def _root_path(self):
        return os.path.abspath(os.sep)

    def _download_dataset(self) -> None:
        self.name = re.sub('[^0-9a-zA-Z]+', '', self.name).lower()
        if self.name in {'conll2003'}:
            if self.train_path is None or self.val_path is None:
                raise ValueError("%s can't be downloaded directly from Gluon due to"
                                 "license Issue. Please provide the downloaded filepath"
                                 "as `train_path` and `val_path`" %self.name)
            self.max_sequence_length = self.max_sequence_length or 180
            self.indexes_format = self.indexes_format or {0: 'text', 3: 'ner'}
        elif self.name == 'wnut2017':
            url_format = 'https://noisy-text.github.io/2017/files/{}'
            train_filename = 'wnut17train.conll'
            val_filename = 'emerging.dev.conll'
            data_dir = os.path.join(self._root_path(), 'tmp', self.name)
            train_path = os.path.join(data_dir, train_filename)
            val_path = os.path.join(data_dir, val_filename)
            download(url_format.format(train_filename), path=train_path)
            download(url_format.format(val_filename), path=val_path)
            if self.train_path is not None or self.val_path is not None:
                LOG.info("AutoGluon downloads the dataset automatically and saved it"
                         " in `%s` directory", data_dir)
            self.train_path = train_path
            self.val_path = val_path
            self.max_sequence_length = self.max_sequence_length or 200
            self.indexes_format = self.indexes_format or {0: 'text', 1: 'ner'}
        elif self.name == 'ontonotesv5':
            if self.train_path is None or self.val_path is None:
                raise ValueError("%s can't be downloaded directly from Gluon due to"
                                 "license Issue. Please provide the downloaded filepath"
                                 "as `train_path` and `val_path`" %self.name)
            self.max_sequence_length = self.max_sequence_length or 300
            self.indexes_format = self.indexes_format or {0: 'text', 3: 'ner'}
        elif self.name == 'jnlpba':
            if self.train_path is None or self.val_path is None:
                raise ValueError("%s can't be downloaded directly from Gluon due to"
                                 "license Issue. Please provide the downloaded filepath"
                                 "as `train_path` and `val_path`" %self.name)
            self.indexes_format = self.indexes_format or {0: 'text', 1: 'ner'}
            # Use SciBERT vocab for this dataset
            self.pretrained_dataset_name = 'scibert_scivocab_cased'
        elif self.name == 'bc5cdr':
            if self.train_path is None or self.val_path is None:
                raise ValueError("%s can't be downloaded directly from Gluon due to"
                                 "license Issue. Please provide the downloaded filepath"
                                 "as `train_path` and `val_path`" %self.name)
            self.indexes_format = self.indexes_format or {0: 'text', 3: 'ner'}
            # Use SciBERT vocab for this dataset
            self.pretrained_dataset_name = 'scibert_scivocab_cased'
        else:
            raise NotImplementedError  # TODO: Add support for more dataset

    def _read_dataset(self) -> None:
        """
        This method reads the datasets. Preprocesses the data.
        Prepares data loader from it.
        """
        self._load_dataset()
        self._preprocess()
        self._prepare_data_loader()

    def _load_dataset(self):
        """Load the dataset from file, convert into BIOES scheme and build tag
        vocabulary"""
        if self.tokenizer is None:
            self.tokenizer = getattr(nlp.data.transforms, 'BERTTokenizer')
        self.tokenizer = self.tokenizer(vocab=self._vocab, lower=False)
        self.train_dataset, max_seq_len_train = load_segment(file_path=self.train_path, tokenizer=self.tokenizer,
                                          indexes_format=self.indexes_format)
        self.val_dataset, max_seq_len_val = load_segment(file_path=self.val_path, tokenizer=self.tokenizer,
                                        indexes_format=self.indexes_format)

        # Set max sequence length based on data if not provided by user
        self.max_sequence_length = self.max_sequence_length or max(max_seq_len_train, max_seq_len_val) + 10

        LOG.info("Train data length: %d", len(self.train_dataset))
        LOG.info("Validation data length: %d", len(self.val_dataset))

        # Build tag vocab
        all_sentences = self.train_dataset + self.val_dataset
        tag_counter = nlp.data.count_tokens(token.tag for sentence in all_sentences
                                            for token in sentence)
        self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                   bos_token=None, eos_token=None, unknown_token=None)
        self.null_tag_index = self.tag_vocab[NULL_TAG]
        self._num_classes = len(self.tag_vocab)
        LOG.info("Number of tag types: %d", self._num_classes)

    def encode_as_input(self, sentence):
        """Encode a single sentence into numpy arrays as input to the BERTTagger model.

        Parameters
        ----------
        sentence: List[TaggedToken]
            A sentence as a list of tagged tokens.

        Returns
        -------
        np.array: token text ids (batch_size, seq_len)
        np.array: token types (batch_size, seq_len),
                which is all zero because we have only one sentence for tagging.
        np.array: valid_length (batch_size,) the number of tokens until [SEP] token
        np.array: tag_ids (batch_size, seq_len)
        np.array: flag_nonnull_tag (batch_size, seq_len),
                which is simply tag_ids != self.null_tag_index
        """
        # check whether the given sequence can be fit into `seq_len`.
        assert len(sentence) <= self.max_sequence_length - 2, \
            'the number of tokens {} should not be larger than {} - 2. offending sentence: {}'\
                .format(len(sentence), self.max_sequence_length, sentence)

        text_tokens = ([self._vocab.cls_token] + [token.text for token in sentence] +
                       [self._vocab.sep_token])
        padded_text_ids = (self._vocab.to_indices(text_tokens)
                           + [self._vocab[self._vocab.padding_token]] *
                           (self.max_sequence_length - len(text_tokens)))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags)
                          + [self.tag_vocab[NULL_TAG]] * (self.max_sequence_length - len(tags)))

        assert len(text_tokens) == len(tags)
        assert len(padded_text_ids) == len(padded_tag_ids)
        assert len(padded_text_ids) == self.max_sequence_length

        valid_length = len(text_tokens)

        # in sequence tagging problems, only one sentence is given
        token_types = [0] * self.max_sequence_length

        np_tag_ids = np.array(padded_tag_ids, dtype='int32')
        # gluon batchify cannot batchify numpy.bool? :(
        flag_nonnull_tag = (np_tag_ids != self.null_tag_index).astype('int32')

        return (np.array(padded_text_ids, dtype='int32'),
                np.array(token_types, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np_tag_ids,
                flag_nonnull_tag)

    def _preprocess(self):
        self.train_dataset = [self.encode_as_input(sentence) for sentence in self.train_dataset]
        self.val_dataset = [self.encode_as_input(sentence) for sentence in self.val_dataset]

    def _prepare_data_loader(self):
        self.train_dataloader = gluon.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            last_batch='rollover', num_workers=cpu_count())
        self.val_dataloader = gluon.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            last_batch='rollover', num_workers=cpu_count())
