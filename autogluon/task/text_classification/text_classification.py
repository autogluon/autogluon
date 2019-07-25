import logging
import os
from typing import AnyStr

import ConfigSpace as CS
from mxnet import gluon

from .losses import *
from .metrics import *
from .model_zoo import *
from .pipeline import *
from .transforms import TextDataTransform
from ..base import BaseTask, Results
from ...loss import Losses
from ...metric import Metrics
from ...network import Nets
from ...optim import Optimizers, get_optim
from ...space import List, Exponential

__all__ = ['TextClassification']
logger = logging.getLogger(__name__)

default_nets = Nets([
    get_model('standard_lstm_lm_200'),
    get_model('standard_lstm_lm_650'),
    get_model('standard_lstm_lm_1500'),
    get_model('awd_lstm_lm_600'),
    get_model('awd_lstm_lm_1150'),
    get_model('bert_12_768_12'),
    get_model('bert_24_1024_16')
])

default_optimizers = Optimizers([
    get_optim('adam'),
    get_optim('sgd'),
    get_optim('ftml'),
    get_optim('bertadam')
])

default_stop_criterion = {
    'time_limits': 1 * 60 * 60,
    'max_metric': 0.80,  # TODO Should be place a bound on metric?
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

        def __init__(self, name: AnyStr, url: AnyStr = None, train_path: AnyStr = None, val_path: AnyStr = None,
                     transform: TextDataTransform = None, batch_size: int = 32, data_format='json',
                     num_workers=4, **kwargs):
            super(TextClassification.Dataset, self).__init__(name, train_path, val_path)
            
            self._transform: TextDataTransform = transform
            self._train_ds_transformed = None
            self._val_ds_transformed = None
            self._train_data_lengths = None
            self.data_format = data_format
            self._label_set = set()
            self.class_labels = None
            self.batch_size = batch_size
            self._download_dataset(url)
            self._add_search_space()
            self.train_field_indices = None
            self.val_field_indices = None
            self.class_labels = None
            self.num_workers = num_workers

            if kwargs:
                if 'train_field_indices' in kwargs:
                    self.train_field_indices = kwargs['train_field_indices']
                if 'val_field_indices' in kwargs:
                    self.val_field_indices = kwargs['val_field_indices']
                else:
                    self.val_field_indices = self.train_field_indices

            if data_format == 'tsv' and self.train_field_indices is None:
                raise ValueError('Specified tsv, but found the field indices empty.')

        def _add_search_space(self):
            cs = CS.ConfigurationSpace()
            data_hyperparams = Exponential(name='batch_size', base=2, lower_exponent=3,
                                           upper_exponent=3).get_hyper_param()

            seq_length_hyperparams = List(name='max_sequence_length', choices=[50, 100, 150, 200]).get_hyper_param()
            cs.add_hyperparameters([data_hyperparams, seq_length_hyperparams])
            self.search_space = cs

        def _download_dataset(self, url) -> None:
            """
            This method downloads the datasets and returns the file path where the data was downloaded.
            :return:
            """
            if self.train_path is None:  # We need to download the dataset.

                if url is None:
                    raise ValueError('Cannot download the dataset as the url is None.')

                root = os.path.join(os.getcwd(), 'data', self.name)
                train_segment = '{}/{}.{}'.format(root, 'train', self.data_format)
                val_segment = '{}/{}.{}'.format(root, 'val', self.data_format)
                segments = [train_segment, val_segment]

                for path in segments:
                    gluon.utils.download(url='{}/{}'.format(url, path.split('/')[-1]), path=path, overwrite=True)

                self.train_path = train_segment
                self.val_path = val_segment

            return self.train_path, self.val_path

        def __repr__(self):
            return "AutoGluon Dataset %s" % self.name

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

        return BaseTask.fit(data=data, nets=nets, optimizers=optimizers, losses=losses, searcher=searcher,
                            trial_scheduler=trial_scheduler,
                            resume=resume, savedir=savedir, visualizer=visualizer,
                            stop_criterion=stop_criterion, resources_per_trial=resources_per_trial, backend=backend,
                            train_func=train_text_classification, reward_attr='accuracy', **kwargs)
