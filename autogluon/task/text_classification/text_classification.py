import json
import logging
from typing import AnyStr

import ConfigSpace as CS
import gluonnlp as nlp

from autogluon.optim import Optimizers
from .dataset import *
from .losses import *
from .metrics import *
from .model_zoo import *
from .optims import get_optim
from .pipeline import *
from .transforms import TextDataTransform, BERTDataTransform, ELMODataTransform
from .utils import *
from ..base import BaseTask, Results
from ...loss import Losses
from ...metric import Metrics
from ...network import Nets
from ...space import List

__all__ = ['TextClassification']
logger = logging.getLogger(__name__)

default_nets = Nets([
    get_model('standard_lstm_lm_200'),
    get_model('standard_lstm_lm_650'),
    get_model('standard_lstm_lm_1500'),
    get_model('awd_lstm_lm_600'),
    get_model('awd_lstm_lm_1150'),
    get_model('bert_12_768_12'),
    get_model('bert_24_1024_16'),
    get_model('elmo_2x2048_256_2048cnn_1xhighway')
])

default_optimizers = Optimizers([
    get_optim('adam'),
    get_optim('sgd'),
    get_optim('ftml'),
    get_optim('bertadam')
])

default_stop_criterion = {
    'time_limits': 1 * 60 * 60,
    'max_metric': 1.0,
    'max_trial_count': 1
}

default_resources_per_trial = {
    'max_num_gpus': 4,
    'max_num_cpus': 4,
    'max_training_epochs': 5
}


class TextClassification(BaseTask):
    class Dataset(BaseTask.Dataset):
        """
        Python class to represent TextClassification Datasets.
        This is a lightweight version of the dataset. This class downloads the dataset if not found, and is used to
        pass on the attributes to pipeline.py where actual dataset objects are constructed.
        """

        def __init__(self, name: AnyStr, train_path: AnyStr = None, val_path: AnyStr = None,
                     batch_size: int = 32, num_workers=4,
                     transform_train_fn=None, transform_val_fn=None, transform_train_list=None, transform_val_list=None,
                     batchify_train_fn=None, batchify_val_fn=None,
                     data_format='txt', **kwargs):

            super(TextClassification.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                                             transform_train_fn, transform_val_fn,
                                                             transform_train_list, transform_val_list,
                                                             batchify_train_fn, batchify_val_fn, **kwargs)

            self.data_format = data_format
            self._label_set = set()
            self.class_labels = None
            self.train_field_indices = None
            self.val_field_indices = None
            self.pair = False

            if kwargs:
                if 'train_field_indices' in kwargs:
                    self.train_field_indices = kwargs['train_field_indices']
                    self.pair = True if len(self.train_field_indices) == 3 else False
                if 'val_field_indices' in kwargs:
                    self.val_field_indices = kwargs['val_field_indices']
                else:
                    self.val_field_indices = self.train_field_indices

                if 'class_labels' in kwargs:
                    self.class_labels = kwargs['class_labels']

            if data_format == 'tsv' and self.train_field_indices is None:
                raise ValueError('Specified tsv, but found the field indices empty.')

            self._read_dataset(**kwargs)
            self._add_search_space()

        def _add_search_space(self):
            cs = CS.ConfigurationSpace()
            data_hyperparams = List(name='batch_size', choices=[8, 16, 32, 64]).get_hyper_param()

            seq_length_hyperparams = List(name='max_sequence_length', choices=[50, 100, 150, 200]).get_hyper_param()
            cs.add_hyperparameters([data_hyperparams, seq_length_hyperparams])
            self.search_space = cs

        def _read_dataset(self, **kwargs):
            self._load_dataset(**kwargs)
            self._label_set = self._infer_labels()
            self.num_classes = len(self._label_set)

        def _infer_labels(self):
            label_set = set()

            for elem in self.train:
                label_set.add(elem[-1])  # Assuming label is at the end.

            lbl_dict = dict([(y, x) for x, y in enumerate(label_set)])
            for elem in self.train:
                elem[-1] = lbl_dict[elem[-1]]

            if self.val:
                for elem in self.val:
                    elem[-1] = lbl_dict[elem[-1]]

            # Also map the labels back to the dataset
            return label_set

        def __repr__(self):
            # TODO
            return "AutoGluon Dataset %s" % self.name

        def _load_dataset(self, **kwargs):
            """
            Loads data from a given data path. If a url is passed, it downloads the data in the init method
            :return:
            """
            if self.train_path is None:
                # Read dataset from gluonnlp.
                import os
                root = os.path.join(os.getcwd(), 'data', self.name)

                self.train_path = '{}/{}.{}'.format(root, 'train', self.data_format)
                self.val_path = '{}/{}.{}'.format(root, 'dev', self.data_format)

                get_dataset(self.name, root=root, segment='train')
                get_dataset(self.name, root=root, segment='dev')

            if self.data_format == 'json':

                if self.val_path is None:
                    # Read the training data and perform split on it.
                    dataset = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
                    self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
                    self.val = TextClassification._Dataset(path=self.val_path, data_format=self.data_format)

            elif self.data_format == 'tsv':

                if self.val_path is None:
                    # Read the training data and perform split on it.
                    dataset = nlp.data.TSVDataset(filename=self.train_path, num_discard_samples=1,
                                                  field_indices=self.train_field_indices)

                    self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = nlp.data.TSVDataset(filename=self.train_path, num_discard_samples=1,
                                                     field_indices=self.train_field_indices)
                    self.val = nlp.data.TSVDataset(filename=self.val_path, num_discard_samples=1,
                                                   field_indices=self.val_field_indices)

            elif self.data_format == 'txt':

                if self.val_path is None:
                    dataset = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
                    self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
                    self.val = TextClassification._Dataset(path=self.val_path, data_format=self.data_format)

            else:
                raise NotImplementedError("Error. Different formats are not supported yet")
            pass


    class _Dataset(nlp.data.TextLineDataset):
        """
        Internal class needed to read the files into a Dataset object.
        This wraps over gluon nlp functions.
        """

        def __init__(self, path, data_format):
            self.path = path
            self.data_format = data_format
            self._data = None

            if self.data_format == 'json':
                with open(self.path) as f:
                    self._data = json.load(f)

            elif self.data_format == 'txt':
                super().__init__(filename=path)

            if self.data_format == 'txt':
                # Need to convert the data into desired format of [TEXT, LABEL] pair.
                # It is input as [LABEL, TEXT] pair.
                self._data = self.transform(lambda line: line.lower().strip().split(' ', 1), lazy=False)
                self._data = self.transform(lambda line: [line[1], line[0]], lazy=False)

    @staticmethod
    def fit(data,
            nets=default_nets,
            optimizers=default_optimizers,
            metrics=Metrics([
                get_metric('Accuracy')]),
            losses=Losses([
                get_loss('SoftmaxCrossEntropyLoss')]),
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion=default_stop_criterion,
            resources_per_trial=default_resources_per_trial,
            backend='default',
            **kwargs) -> Results:
        r"""
               Fit networks on dataset

               Parameters
               ----------
               data: Input data. It could be:
                   autogluon.Datasets
                   task.Datasets
               nets: autogluon.Nets
               optimizers: autogluon.Optimizers
               metrics: autogluon.Metrics
               losses: autogluon.Losses
               stop_criterion (dict): The stopping criteria. The keys may be any field in
                   the return result of 'train()', whichever is reached first.
                   Defaults to empty dict.
               resources_per_trial (dict): Machine resources to allocate per trial,
                   e.g. ``{"max_num_cpus": 64, "max_num_gpus": 8}``. Note that GPUs will not be
                   assigned unless you specify them here.
               savedir (str): Local dir to save training results to.
               searcher: Search Algorithm.
               trial_scheduler: Scheduler for executing
                   the experiment. Choose among FIFO (default) and HyperBand.
               resume (bool): If checkpoint exists, the experiment will
                   resume from there.
               backend: support autogluon default backend, ray. (Will support SageMaker)
               **kwargs: Used for backwards compatibility.

               Returns
               ----------
               results:
                   model: the parameters associated with the best model.
                   val_accuracy: validation set accuracy
                   config: best configuration
                   time: total time cost
               """
        if kwargs is None:
            kwargs = dict()
        kwargs['data_format'] = data.data_format
        kwargs['data_name'] = data.name
        kwargs['train_path'] = data.train_path
        kwargs['val_path'] = data.val_path
        kwargs['log_dir'] = savedir

        return BaseTask.fit(data=data, nets=nets, optimizers=optimizers, metrics=metrics, losses=losses,
                            searcher=searcher,
                            trial_scheduler=trial_scheduler,
                            resume=resume, savedir=savedir, visualizer=visualizer,
                            stop_criterion=stop_criterion, resources_per_trial=resources_per_trial, backend=backend,
                            train_func=train_text_classification, reward_attr='accuracy', **kwargs)
