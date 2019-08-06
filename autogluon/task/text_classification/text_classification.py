import logging
import multiprocessing
import glob

import ConfigSpace as CS

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
from ...space import List, Exponential

__all__ = ['TextClassification']
logger = logging.getLogger(__name__)

default_nets = Nets([
    #get_model('standard_lstm_lm_200'),
    #get_model('standard_lstm_lm_650'),
    #get_model('standard_lstm_lm_1500'),
    #get_model('awd_lstm_lm_600'),
    #get_model('awd_lstm_lm_1150'),
    #get_model('bert_12_768_12'),
    #get_model('bert_24_1024_16'),
    get_model('elmo_2x1024_128_2048cnn_1xhighway')
])

default_optimizers = Optimizers([
    get_optim('adam'),
    #get_optim('sgd'),
    #get_optim('ftml'),
    #get_optim('bertadam')
])

default_stop_criterion = {
    'time_limits': 1 * 60 * 60,
    'max_metric': 1.0,  # TODO Should be place a bound on metric?
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
            data_hyperparams = Exponential(name='batch_size', base=2, lower_exponent=5,
                                           upper_exponent=5).get_hyper_param()

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
            for elem in self.val:
                elem[-1] = lbl_dict[elem[-1]]

            # Also map the labels back to the dataset
            return label_set

        def __repr__(self):
            # TODO
            return "AutoGluon Dataset %s" % self.name

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

        def _load_dataset(self, **kwargs):
            """
            Loads data from a given data path. If a url is passed, it downloads the data in the init method
            :return:
            """
            if self.train_path is None:
                # Read dataset from gluonnlp.
                import os
                root = os.path.join(os.getcwd(), 'data', self.name)
                self.train_path = glob.glob('{}/{}.{}'.format(root, 'train*', self.data_format))[0]
                self.val_path = glob.glob('{}/{}.{}'.format(root, 'dev*', self.data_format))[0]

                get_dataset(self.name, root=root, segment='train')
                get_dataset(self.name, root=root, segment='dev')

            if self.data_format == 'json':

                if self.val_path is None:
                    # Read the training data and perform split on it.
                    dataset = get_dataset_from_json_files(path=self.train_path)
                    self.train, self.val = TextClassification.Dataset._train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = get_dataset_from_json_files(path=self.train_path)
                    self.val = get_dataset_from_json_files(path=self.val_path)

            elif self.data_format == 'tsv':

                if self.val_path is None:
                    # Read the training data and perform split on it.
                    dataset = get_dataset_from_tsv_files(path=self.train_path,
                                                         field_indices=self.train_field_indices)
                    self.train, self.val = TextClassification.Dataset._train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = get_dataset_from_tsv_files(path=self.train_path,
                                                            field_indices=self.train_field_indices)
                    self.val = get_dataset_from_tsv_files(path=self.val_path,
                                                          field_indices=self.val_field_indices)

            elif self.data_format == 'txt':

                if self.val_path is None:
                    dataset = get_dataset_from_txt_files(path=self.train_path)
                    self.train, self.val = TextClassification.Dataset._train_valid_split(dataset, valid_ratio=0.2)

                else:
                    self.train = get_dataset_from_txt_files(path=self.train_path)
                    self.val = get_dataset_from_txt_files(path=self.val_path)

            else:
                raise NotImplementedError("Error. Different formats are not supported yet")
            pass

        def transform(self, dataset, transform_fn):
            # The model type is necessary to pre-process it based on the inputs required to the model.
            with multiprocessing.Pool(self.num_workers) as pool:
                return gluon.data.SimpleDataset(pool.map(transform_fn, dataset))

        def get_train_data_lengths(self, model_name: AnyStr, dataset):
            with multiprocessing.Pool(self.num_workers) as pool:
                if 'bert' in model_name:
                    return dataset.transform(lambda token_id, length, segment_id, label_id: length,
                                             lazy=False)
                else:
                    return dataset.transform(lambda data, label: int(len(data)), lazy=False)

        def get_batchify_fn(self, model_name: AnyStr):
            if 'bert' in model_name:
                return nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(dtype='int32'))
            else:
                return nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, ret_length=True),
                                        nlp.data.batchify.Stack(dtype='float32'))

        def get_transform_train_fn(self, model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length):
            if 'bert' in model_name:
                class_labels = self.class_labels if self.class_labels else list(self._label_set)
                dataset_transform = BERTDataTransform(tokenizer=nlp.data.BERTTokenizer(vocab=vocab, lower=True),
                                                      max_seq_length=max_sequence_length,
                                                      pair=self.pair, class_labels=class_labels)
            elif 'elmo' in model_name:
                dataset_transform = ELMODataTransform(vocab, max_sequence_length=max_sequence_length)
            else:
                dataset_transform = TextDataTransform(vocab, transforms=[
                    nlp.data.ClipSequence(length=max_sequence_length)],
                                                      pair=self.pair, max_sequence_length=max_sequence_length)
            return dataset_transform

        def get_transform_val_fn(self, model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length):
            return self.get_transform_train_fn(model_name, vocab, max_sequence_length)

        def get_batch_sampler(self, model_name: AnyStr, train_dataset):
            train_data_lengths = self.get_train_data_lengths(model_name, train_dataset)
            return nlp.data.FixedBucketSampler(train_data_lengths, batch_size=self.batch_size,
                                               shuffle=True,
                                               num_buckets=10, ratio=0)

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
