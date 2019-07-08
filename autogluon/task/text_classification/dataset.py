import os
from multiprocessing import cpu_count
from typing import AnyStr, List

import gluonnlp as nlp
from mxnet import gluon

from autogluon.dataset import TextDataTransform, BERTDataTransform, utils
from ... import dataset

__all__ = ['Dataset', 'BERTDataset']


def get_gluon_nlp_dataset_fn(name):
    if name == 'sst_2':
        return nlp.data.SST_2
    elif name == 'imdb':
        return nlp.data.IMDB
    elif name == 'glue_sst':
        return nlp.data.GlueSST2
    elif name == 'glue_mnli':
        return nlp.data.GlueMNLI
    else:
        raise NotImplementedError


def _get_len(x):
    return int(len(x[0]))


def _get_len_bert(x):
    return int(x[1])


class Dataset(dataset.Dataset):
    """
    Python class to represent TextClassification Datasets
    """

    def __init__(self, name: AnyStr = None, train_path: AnyStr = None, val_path: AnyStr = None, lazy: bool = True,
                 transform: TextDataTransform = None, batch_size: int = 32, data_format='json', field_indices=None):
        super(Dataset, self).__init__(name, train_path, val_path)
        # TODO : This currently works only for datasets from GluonNLP. This needs to be made more generic.
        # TODO : add search space, handle batch_size, num_workers
        self._num_classes: int = 0
        self._transform: TextDataTransform = transform
        self._train_ds_transformed = None
        self._val_ds_transformed = None
        self._train_data_lengths = None  # TODO There is an alternative way possible to avoid creating this list
        self.data_format = data_format  # TODO This should come from config
        self._label_set = set()
        self.batch_size = batch_size
        self._download_dataset()
        self.add_search_space()

        self._vocab = None

        if data_format == 'tsv' and field_indices is None:
            raise ValueError('Specified tsv, but found the field indices empty.')

        self.field_indices = field_indices
        if not lazy:
            self._init_()

    def _init_(self):

        self._read_dataset()
        self.add_search_space()
        self._num_classes = len(self._label_set)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def vocab(self) -> nlp.vocab:
        return self._vocab

    @vocab.setter
    def vocab(self, value: nlp.vocab) -> None:
        self.vocab = value
        self._init_()

    def _download_dataset(self) -> None:
        """
        This method downloads the datasets and returns the file path where the data was downloaded.
        :return:
        """
        if self.train_path is None:  # We need to download the dataset.
            root = os.path.join(os.getcwd(), 'data')
            gluon_nlp_data_fn = get_gluon_nlp_dataset_fn(self.name)

            val_set = 'dev_matched'

            train_dataset, val_dataset = [gluon_nlp_data_fn(root=root, segment=segment)
                                          for segment in ('train', val_set)]
            self.train_path = os.path.join(os.getcwd(), 'data', 'train.{}'.format(self.data_format))
            self.val_path = os.path.join(os.getcwd(), 'data', '{}.{}'.format(val_set, self.data_format))

    def _read_dataset(self) -> None:
        """
        This method reads the datasets. Performs transformations on it. Preprocesses the data.
        Prepares data loader from it.
        :return:
        """
        self._load_dataset()
        self._preprocess()
        self._prepare_data_loader()

    def _load_dataset(self):
        """
        Loads data from a given data path. If a url is passed, it downloads the data in the init method
        :return:
        """
        # TODO Different datasets have a different parsing schema.
        # TODO Should load_dataset be a part of a config as well ?
        # What to do with nested folders ?

        if self.data_format == 'json':

            if self.val_path is None:
                # Read the training data and perform split on it.
                dataset, self._label_set = utils.get_dataset_from_json_files(path=self.train_path)
                self.train_dataset, self.val_dataset = self._train_valid_split(dataset, valid_ratio=0.2)

            else:
                self.train_dataset, self._label_set = utils.get_dataset_from_json_files(path=self.train_path)
                self.val_dataset, _ = utils.get_dataset_from_json_files(path=self.val_path)

        elif self.data_format == 'tsv':

            if self.val_path is None:
                # Read the training data and perform split on it.
                dataset, self._label_set = utils.get_dataset_from_tsv_files(path=self.train_path,
                                                                            field_indices=self.field_indices)
                self.train_dataset, self.val_dataset = self._train_valid_split(dataset, valid_ratio=0.2)

            else:
                self.train_dataset, self._label_set = utils.get_dataset_from_tsv_files(path=self.train_path,
                                                                                       field_indices=self.field_indices)
                self.val_dataset, _ = utils.get_dataset_from_tsv_files(path=self.val_path,
                                                                       field_indices=self.field_indices)

        else:
            raise NotImplementedError("Error. Different formats are not supported yet")

    def _preprocess(self):

        self.train_dataset = self.train_dataset.transform(self._transform, lazy=True)
        self.val_dataset = self.val_dataset.transform(self._transform, lazy=True)
        # self._train_data_lengths = self._get_train_data_lengths()

    def _get_train_data_lengths(self) -> List[int]:

        import multiprocessing as mp
        with mp.Pool(cpu_count() - 1) as pool:
            lengths = gluon.data.SimpleDataset(pool.map(_get_len, self.train_dataset))
        return lengths

    def _prepare_data_loader(self):
        # TODO : Currently hardcoding the batch_samplers and batchify_fn.
        # These need to come from configs.
        """
        Method which prepares and populates the data loader.
        :return:
        """

        # batch_sampler = nlp.data.FixedBucketSampler(self._train_data_lengths, batch_size=self.batch_size, shuffle=True)
        batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, ret_length=True),
                                              nlp.data.batchify.Stack(dtype='float32'))

        self.train_data_loader = gluon.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                       shuffle=True, last_batch='rollover',
                                                       batchify_fn=batchify_fn, num_workers=cpu_count())
        # TODO Think about cpu_count here.

        self.val_data_loader = gluon.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                                     last_batch='rollover',
                                                     batchify_fn=batchify_fn, num_workers=cpu_count(), shuffle=False)

    @staticmethod
    def _train_valid_split(dataset: gluon.data.Dataset, valid_ratio=0.20) -> [gluon.data.Dataset,
                                                                              gluon.data.Dataset]:
        """
        Splits the dataset into training and validation sets.

        :param valid_ratio: float, default 0.20
                    Proportion of training samples to be split into validation set.
                    range: [0, 1]
        :return:

        """
        return nlp.data.utils.train_valid_split(dataset, valid_ratio)

    def __repr__(self):
        return "AutoGluon Dataset %s" % self.name


class BERTDataset(Dataset):

    def _preprocess(self):
        self._transform = BERTDataTransform(tokenizer=self._transform.tokenizer,
                                            max_seq_length=self._transform.max_seq_length,
                                            class_labels=self._label_set, pair=False)
        self.train_dataset = self.train_dataset.transform(self._transform, lazy=True)
        self.val_dataset = self.val_dataset.transform(self._transform, lazy=True)
        # self._train_data_lengths = self._get_train_data_lengths()

    def _get_train_data_lengths(self) -> List[int]:
        import multiprocessing as mp
        with mp.Pool() as pool:
            lengths = gluon.data.SimpleDataset(pool.map(_get_len_bert, self.train_dataset))
        return lengths

    def _prepare_data_loader(self):
        """
        Method which prepares and populates the data loader.
        :return:
        """

        # batch_sampler = nlp.data.FixedBucketSampler(self._train_data_lengths, batch_size=self.batch_size, shuffle=True,
        #                                             num_buckets=5)

        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(dtype='int32'))

        self.train_data_loader = gluon.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                       shuffle=True, last_batch='rollover',
                                                       batchify_fn=batchify_fn, num_workers=cpu_count())
        # TODO Think about cpu_count here.

        self.val_data_loader = gluon.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                                     last_batch='rollover',
                                                     batchify_fn=batchify_fn, num_workers=cpu_count(), shuffle=False)
