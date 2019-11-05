import logging
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp

from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .model_zoo import get_network
from .dataset import TextClassificationDataset
from .pipeline import *
from .metrics import get_metric_instance
from .optimizers import *
from .predictor import TextClassificationPredictor

__all__ = ['TextClassification']

logger = logging.getLogger(__name__)

class TextClassification(BaseTask):
    """AutoGluon TextClassification Task
    """
    Dataset = TextClassificationDataset
    @staticmethod
    def fit(dataset='SST',
            net=Categorical('bert_12_768_12'),
            optimizer=Categorical(BERTAdam(learning_rate=Real(2e-06, 2e-04, log=True))),
            lr_scheduler='cosine',
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=32,
            epochs=3,
            metric='accuracy',
            nthreads_per_trial=4,
            ngpus_per_trial=1,
            hybridize=True,
            search_strategy='random',
            search_options={},
            time_limits=None,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=[],
            grace_period=None,
            auto_search=True,
            **kwargs):

        """
        Fit networks on dataset

        Args:
            dataset (str or autogluon.task.ImageClassification.Dataset): Training dataset.
            net (str, autogluon.AutoGluonObject, or ag.Choice of AutoGluonObject): Network candidates.
            optimizer (str, autogluon.AutoGluonObject, or ag.Choice of AutoGluonObject): optimizer candidates.
            metric (str or object): observation metric.
            loss (object): training loss function.
            num_trials (int): number of trials in the experiment.
            time_limits (int): training time limits in seconds.
            resources_per_trial (dict): Machine resources to allocate per trial.
            savedir (str): Local dir to save training results to.
            search_strategy (str): Search Algorithms ('random', 'bayesopt' and 'hyperband')
            resume (bool): If checkpoint exists, the experiment will resume from there.


        Example:
            >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> predictor = task.fit(dataset,
            >>>                      nets=ag.Choice['resnet18_v1', 'resnet34_v1'],
            >>>                      time_limits=time_limits,
            >>>                      num_gpus=1,
            >>>                      num_trials = 4)
        """
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        train_text_classification.register_args(
            dataset=dataset,
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            metric=metric,
            num_gpus=ngpus_per_trial,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            final_fit=False,
            **kwargs)

        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'classification_reward',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'searcher': 'random',
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})
        results = BaseTask.run_fit(train_text_classification, search_strategy,
                                   scheduler_options)
        args = sample_config(train_text_classification.args, results['best_config'])
        model = get_network(args.net, results['num_classes'], mx.cpu(0))
        update_params(model, results.pop('model_params'))
        return TextClassificationPredictor(model, results, evaluate, checkpoint, args)

    # @classmethod
    # def predict(cls, sentence):
    #     """The task predict function given an input.
    #      Args:
    #         img: the input
    #      Example:
    #         >>> ind, prob = task.predict('this is cool')
    #     """
    #     pass
    #
    # @classmethod
    # def evaluate(cls, dataset):
    #     """The task evaluation function given the test dataset.
    #      Args:
    #         dataset: test dataset
    #      Example:
    #         >>> from autogluon import ImageClassification as task
    #         >>> dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
    #         >>> test_reward = task.evaluate(dataset)
    #     """
    #     logger.info('Start evaluating.')
    #     args = cls.scheduler.train_fn.args
    #     batch_size = args.batch_size * max(args.num_gpus, 1)
    #
    #     metric = get_metric_instance(args.metric)
    #     ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    #     cls.results.model.collect_params().reset_ctx(ctx)
    #
    #     if isinstance(dataset, AutoGluonObject):
    #         dataset = dataset._lazy_init()
    #     test_data = gluon.data.DataLoader(
    #         dataset.test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    #
    #     for i, batch in enumerate(test_data):
    #         data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx,
    #                                              batch_axis=0, even_split=False)
    #         label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx,
    #                                               batch_axis=0, even_split=False)
    #         outputs = [cls.results.model(X) for X in data]
    #         metric.update(label, outputs)
    #     _, test_reward = metric.get()
    #     logger.info('Finished.')
    #     return test_reward


# import json
# import logging
# import numpy as np
# from typing import AnyStr
# import matplotlib.pyplot as plt
#
# import ConfigSpace as CS
# import mxnet as mx
# import gluonnlp as nlp
#
# from autogluon.optimizer import Optimizers
# from .dataset import *
# from .losses import *
# from .metrics import *
# from .model_zoo import *
# from .optimizers import get_optim
# from .pipeline import *
# # from .transforms import TextDataTransform, BERTDataTransform
# from ..base import BaseTask, Results
# from ...loss import Losses
# from ...metric import Metrics
# from ...network import Nets
# from ...space import List
#
# __all__ = ['TextClassification']
# logger = logging.getLogger(__name__)
#
# # default_nets = Nets([
# #     get_model('standard_lstm_lm_200'),
# #     get_model('standard_lstm_lm_650'),
# #     get_model('standard_lstm_lm_1500'),
# #     get_model('awd_lstm_lm_600'),
# #     get_model('awd_lstm_lm_1150'),
# #     get_model('bert_12_768_12'),
# #     get_model('bert_24_1024_16')
# # ])
# #
# # default_optimizers = Optimizers([
# #     get_optim('adam'),
# #     get_optim('sgd'),
# #     get_optim('ftml'),
# #     get_optim('bertadam')
# # ])
# #
# # default_stop_criterion = {
# #     'time_limits': 1 * 60 * 60,
# #     'max_metric': 0.80,  # TODO Should be place a bound on metric?
# #     'max_trial_count': 1
# # }
# #
# # default_resources_per_trial = {
# #     'max_num_gpus': 4,
# #     'max_num_cpus': 4,
# #     'max_training_epochs': 5
# # }
#
#
# class TextClassification(BaseTask):
#     class Dataset(BaseTask.Dataset):
#         """
#         Python class to represent TextClassification Datasets.
#         This is a lightweight version of the dataset. This class downloads the dataset if not found, and is used to
#         pass on the attributes to pipeline.py where actual dataset objects are constructed.
#         """
#
#         def __init__(self, name: AnyStr, train_path: AnyStr = None, val_path: AnyStr = None,
#                      batch_size: int = 32, num_workers=4,
#                      transform_train_fn=None, transform_val_fn=None, transform_train_list=None, transform_val_list=None,
#                      batchify_train_fn=None, batchify_val_fn=None,
#                      data_format='tsv', **kwargs):
#
#             super(TextClassification.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
#                                                              transform_train_fn, transform_val_fn,
#                                                              transform_train_list, transform_val_list,
#                                                              batchify_train_fn, batchify_val_fn, **kwargs)
#
#             # self.data_format = data_format
#             # self._label_set = set()
#             # self.class_labels = None
#             # self.train_field_indices = None
#             # self.val_field_indices = None
#             # self.pair = False
#             #
#             # if kwargs:
#             #     if 'train_field_indices' in kwargs:
#             #         self.train_field_indices = kwargs['train_field_indices']
#             #         self.pair = True if len(self.train_field_indices) == 3 else False
#             #     if 'val_field_indices' in kwargs:
#             #         self.val_field_indices = kwargs['val_field_indices']
#             #     else:
#             #         self.val_field_indices = self.train_field_indices
#             #
#             #     if 'class_labels' in kwargs:
#             #         self.class_labels = kwargs['class_labels']
#             #
#             # if data_format == 'tsv' and self.train_field_indices is None:
#             #     raise ValueError('Specified tsv, but found the field indices empty.')
#             #
#             # self._read_dataset(**kwargs)
#             # self._add_search_space()
#             pass
#
#         def _read_dataset(self, **kwargs):
#             try:
#                 # self.train = get_dataset(self.name, train=True)
#                 # self.val = None
#                 # self.test = get_dataset(self.name, train=False)
#                 # self.num_classes = len(np.unique(self.train._label))
#                 pass
#             except ValueError:
#                 if self.train_path is not None:
#                     if '.tsv' or '.csv' in self.train_path:
#                         self.train = nlp.data.TSVDataset(self.train_path)
#                         if self.val_path is not None:
#                             self.val = nlp.data.TSVDataset(self.val_path)
#                         else:
#                             self.val = None
#                     else:
#                         raise NotImplementedError
#                 elif 'test_path' in kwargs:
#                     if '.tsv' or '.csv' in kwargs['test_path']:
#                         self.test = nlp.data.TSVDataset(kwargs['test_path'])
#                     else:
#                         raise NotImplementedError
#                 else:
#                     raise NotImplementedError
#
#         def _add_search_space(self, **kwargs):
#             cs = CS.ConfigurationSpace()
#             batch_size_hps = List(name='batch_size', choices=[8, 16, 32, 64]).get_hyper_param()
#             seq_length_hps = List(name='max_sequence_length',
#                                   choices=[50, 100, 150, 200]).get_hyper_param()
#             transform_train_hps = List(name='transform_train',
#                                        choices=self.transform_train_list).get_hyper_param()
#             transform_val_hps = List(name='transform_val',
#                                      choices=self.transform_val_list).get_hyper_param()
#             hps = [batch_size_hps, seq_length_hps, transform_train_hps, transform_val_hps]
#             extra_hps = []
#             for k, v in kwargs:
#                 extra_hps.append(List(name=k, choices=v))
#             hps = hps + extra_hps
#             cs.add_hyperparameters(hps)
#             self.search_space = cs
#
#         # def _read_dataset(self, **kwargs):
#         #     self._load_dataset(**kwargs)
#         #     self._label_set = self._infer_labels()
#         #     self.num_classes = len(self._label_set)
#         #
#         # def _infer_labels(self):
#         #     label_set = set()
#         #
#         #     for elem in self.train:
#         #         label_set.add(elem[-1])  # Assuming label is at the end.
#         #
#         #     lbl_dict = dict([(y, x) for x, y in enumerate(label_set)])
#         #     for elem in self.train:
#         #         elem[-1] = lbl_dict[elem[-1]]
#         #
#         #     if self.val:
#         #         for elem in self.val:
#         #             elem[-1] = lbl_dict[elem[-1]]
#         #
#         #     # Also map the labels back to the dataset
#         #     return label_set
#         #
#         # def __repr__(self):
#         #     # TODO
#         #     return "AutoGluon Dataset %s" % self.name
#         #
#         # def _load_dataset(self, **kwargs):
#         #     """
#         #     Loads data from a given data path. If a url is passed, it downloads the data in the init method
#         #     :return:
#         #     """
#         #     if self.train_path is None:
#         #         # Read dataset from gluonnlp.
#         #         import os
#         #         root = os.path.join(os.getcwd(), 'data', self.name)
#         #         self.train_path = '{}/{}.{}'.format(root, 'train', self.data_format)
#         #         self.val_path = '{}/{}.{}'.format(root, 'dev', self.data_format)
#         #
#         #         get_dataset(self.name, root=root, segment='train')
#         #         get_dataset(self.name, root=root, segment='dev')
#         #
#         #     if self.data_format == 'json':
#         #
#         #         if self.val_path is None:
#         #             # Read the training data and perform split on it.
#         #             dataset = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
#         #             self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)
#         #
#         #         else:
#         #             self.train = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
#         #             self.val = TextClassification._Dataset(path=self.val_path, data_format=self.data_format)
#         #
#         #     elif self.data_format == 'tsv':
#         #
#         #         if self.val_path is None:
#         #             # Read the training data and perform split on it.
#         #             dataset = nlp.data.TSVDataset(filename=self.train_path, num_discard_samples=1,
#         #                                           field_indices=self.train_field_indices)
#         #
#         #             self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)
#         #
#         #         else:
#         #             self.train = nlp.data.TSVDataset(filename=self.train_path, num_discard_samples=1,
#         #                                              field_indices=self.train_field_indices)
#         #             self.val = nlp.data.TSVDataset(filename=self.val_path, num_discard_samples=1,
#         #                                            field_indices=self.val_field_indices)
#         #
#         #     elif self.data_format == 'txt':
#         #
#         #         if self.val_path is None:
#         #             dataset = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
#         #             self.train, self.val = nlp.data.utils.train_valid_split(dataset, valid_ratio=0.2)
#         #
#         #         else:
#         #             self.train = TextClassification._Dataset(path=self.train_path, data_format=self.data_format)
#         #             self.val = TextClassification._Dataset(path=self.val_path, data_format=self.data_format)
#         #
#         #     else:
#         #         raise NotImplementedError("Error. Different formats are not supported yet")
#         #     pass
#
#     class _Dataset(nlp.data.TextLineDataset):
#         """
#         Internal class needed to read the files into a Dataset object.
#         This wraps over gluon nlp functions.
#         """
#
#         def __init__(self, path, data_format):
#             self.path = path
#             self.data_format = data_format
#             self._data = None
#
#             if self.data_format == 'json':
#                 with open(self.path) as f:
#                     self._data = json.load(f)
#
#             elif self.data_format == 'txt':
#                 super().__init__(filename=path)
#
#             if self.data_format == 'txt':
#                 # Need to convert the data into desired format of [TEXT, LABEL] pair.
#                 # It is input as [LABEL, TEXT] pair.
#                 self._data = self.transform(lambda line: line.lower().strip().split(' ', 1), lazy=False)
#                 self._data = self.transform(lambda line: [line[1], line[0]], lazy=False)
#
#     # TODO: fix
#     @staticmethod
#     def predict(img):
#         """The task predict function given an input.
#
#          Args:
#             img: the input
#
#          Example:
#             >>> ind, prob = task.predict('example.jpg')
#         """
#         logger.info('Start predicting.')
#         ctx = [mx.gpu(i) for i in range(BaseTask.result.metadata['resources_per_trial']['num_gpus'])] \
#             if BaseTask.result.metadata['resources_per_trial']['num_gpus'] > 0 else [mx.cpu()]
#         # img = utils.download(img)
#         img = mx.image.imread(img)
#         plt.imshow(img.asnumpy())
#         plt.show()
#         transform_fn = transforms.Compose(BaseTask.result.metadata['data'].transform_val_list)
#         img = transform_fn(img)
#         pred = BaseTask.result.model(img.expand_dims(axis=0).as_in_context(ctx[0]))
#         ind = mx.nd.argmax(pred, axis=1).astype('int')
#         logger.info('Finished.')
#         mx.nd.waitall()
#         return ind, mx.nd.softmax(pred)[0][ind]
#
#
#     @staticmethod
#     def evaluate(data):
#         """The task evaluation function given the test data.
#
#          Args:
#             data: test data
#
#          Example:
#             >>> test_acc = task.evaluate(data)
#         """
#         logger.info('Start evaluating.')
#         L = get_loss_instance(BaseTask.result.metadata['losses'].loss_list[0].name)
#         metric = get_metric_instance(BaseTask.result.metadata['metrics'].metric_list[0].name)
#         ctx = [mx.gpu(i) for i in range(BaseTask.result.metadata['resources_per_trial']['num_gpus'])] \
#             if BaseTask.result.metadata['resources_per_trial']['num_gpus'] > 0 else [mx.cpu()]
#         def _init_dataset(dataset, transform_fn, transform_list):
#             if transform_fn is not None:
#                 dataset = dataset.transform(transform_fn)
#             if transform_list is not None:
#                 dataset = dataset.transform_first(transforms.Compose(transform_list))
#             return dataset
#
#         test_dataset = _init_dataset(data.test,
#                                      data.transform_val_fn,
#                                      data.transform_val_list)
#         test_data = mx.gluon.data.DataLoader(
#             test_dataset,
#             batch_size=data.batch_size,
#             shuffle=False,
#             num_workers=data.num_workers)
#         test_loss = 0
#         for i, batch in enumerate(test_data):
#             data = mx.gluon.utils.split_and_load(batch[0],
#                                               ctx_list=ctx,
#                                               batch_axis=0,
#                                               even_split=False)
#             label = mx.gluon.utils.split_and_load(batch[1],
#                                                ctx_list=ctx,
#                                                batch_axis=0,
#                                                even_split=False)
#             outputs = [BaseTask.result.model(X) for X in data]
#             loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
#
#             test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
#             metric.update(label, outputs)
#         _, test_acc = metric.get()
#         test_loss /= len(test_data)
#         logger.info('Finished.')
#         mx.nd.waitall()
#         return test_acc
#
#     @staticmethod
#     def fit(data,
#             nets=Nets(['bert_12_768_12']),
#             optimizers=Optimizers(['adam', 'bertadam']),
#             metrics=Metrics([get_metric('Accuracy')]),
#             losses=Losses([get_loss('SoftmaxCrossEntropyLoss')]),
#             searcher='random',
#             trial_scheduler='fifo',
#             resume=False,
#             savedir='checkpoint/exp1.ag',
#             visualizer='tensorboard',
#             stop_criterion={
#                 'time_limits': 24 * 60 * 60,
#                 'max_metric': 1.0,
#                 'num_trials': 100
#             },
#             resources_per_trial={
#                 'num_gpus': 1,
#                 'num_training_epochs': 100
#             },
#             backend='default',
#             **kwargs):
#         """
#         Fit networks on dataset
#
#         Args:
#             data: Input data. task.Datasets.
#             nets: autogluon.Nets.
#             optimizers: autogluon.Optimizers.
#             metrics: autogluon.Metrics.
#             losses: autogluon.Losses.
#             stop_criterion (dict): The stopping criteria.
#             resources_per_trial (dict): Machine resources to allocate per trial.
#             savedir (str): Local dir to save training results to.
#             searcher: Search Algorithm.
#             trial_scheduler: Scheduler for executing the experiment. Choose among FIFO (default) and HyperBand.
#             resume (bool): If checkpoint exists, the experiment will resume from there.
#             backend: support autogluon default backend.
#
#         Example:
#             >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
#             >>>                         test_path='data/test')
#             >>> net_list = ['resnet18_v1', 'resnet34_v1']
#             >>> nets = ag.Nets(net_list)
#             >>> adam_opt = ag.optims.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
#             >>>                           wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
#             >>> sgd_opt = ag.optims.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
#             >>>                         momentum=ag.space.Linear('momentum', 0.85, 0.95),
#             >>>                         wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
#             >>> optimizers = ag.Optimizers([adam_opt, sgd_opt])
#             >>> searcher = 'random'
#             >>> trial_scheduler = 'fifo'
#             >>> savedir = 'checkpoint/demo.ag'
#             >>> resume = False
#             >>> time_limits = 3*60
#             >>> max_metric = 1.0
#             >>> num_trials = 4
#             >>> stop_criterion = {
#             >>>       'time_limits': time_limits,
#             >>>       'max_metric': max_metric,
#             >>>       'num_trials': num_trials
#             >>> }
#             >>> num_gpus = 1
#             >>> num_training_epochs = 2
#             >>> resources_per_trial = {
#             >>>       'num_gpus': num_gpus,
#             >>>       'num_training_epochs': num_training_epochs
#             >>> }
#             >>> results = task.fit(dataset,
#             >>>                     nets,
#             >>>                     optimizers,
#             >>>                     searcher=searcher,
#             >>>                     trial_scheduler=trial_scheduler,
#             >>>                     resume=resume,
#             >>>                     savedir=savedir,
#             >>>                     stop_criterion=stop_criterion,
#             >>>                     resources_per_trial=resources_per_trial)
#         """
#         # if kwargs is None:
#         #     kwargs = dict()
#         # kwargs['data_format'] = data.data_format
#         # kwargs['data_name'] = data.name
#         # kwargs['train_path'] = data.train_path
#         # kwargs['val_path'] = data.val_path
#         # kwargs['log_dir'] = savedir
#
#         return BaseTask.fit(data,  nets, optimizers, metrics, losses, searcher, trial_scheduler,
#                             resume, savedir, visualizer, stop_criterion, resources_per_trial,
#                             backend,
#                             reward_attr='accuracy',
#                             train_func=train_text_classification,
#                             **kwargs)