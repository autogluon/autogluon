import os
import copy
import shutil
import logging
import argparse
import time
import json
from abc import ABC
import matplotlib.pyplot as plt
import ConfigSpace as CS

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

import autogluon as ag
from ...optimizer import Optimizers, get_optim
from ... import dataset
from ..image_classification.losses import *
from ..image_classification.metrics import *
from ...searcher import BaseSearcher

__all__ = ['BaseTask']

logger = logging.getLogger(__name__)

class Results(object):
    """the output structure of the fit function.

    Args:
        model: the final model with parameters.
        metric: the best result regarding to the provided evaluation metric.
        config: the best config regarding to the metric.
        time: the total time cost of the fit procedure.
        metadata: all the search space information.

    Example:
        >>> results = Results(model, metric, config, time, metadata)
    """
    def __init__(self, model, metric, config, time, metadata):
        self._model = model
        self._metric = metric
        self._config = config
        self._time = time
        self._metadata = metadata

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, val):
        self._metric = val

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._time = val

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, val):
        self._metadata = val


class BaseTask(ABC):
    class Dataset(dataset.Dataset):
        """The base dataset.

        Args:
            name: the dataset name.
            train_path: the training data location
            val_path: the validation data location.
            batch_size: the batch size.
            num_workers: the number of workers used in DataLoader.
            transform_train_fn: the transformation function for training data.
            transform_val_fn: the transformation function for validation data.
            transform_train_list: the compose list of Transformations for training data.
            transform_val_list: the compose list of Transformations for validation data.
            batchify_train_fn: the batchify function defined for training data.
            batchify_val_fn: the batchify function defined for validation data.
        """
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=None,
                     num_workers=None,
                     transform_train_fn=None, transform_val_fn=None,
                     transform_train_list=None, transform_val_list=None,
                     batchify_train_fn=None, batchify_val_fn=None, **kwargs):
            super(BaseTask.Dataset, self).__init__(name, train_path, val_path, batch_size,
                                                   num_workers,
                                                   transform_train_fn, transform_val_fn,
                                                   transform_train_list, transform_val_list,
                                                   batchify_train_fn, batchify_val_fn, **kwargs)

        def _read_dataset(self, **kwargs):
            pass

        def _add_search_space(self):
            pass

        def _get_search_space_strs(self):
            pass

    def __init__(self):
        self._result = None
        self._trial_scheduler = None
        super(BaseTask, self).__init__()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, val):
        self._result = val

    @property
    def trial_scheduler (self):
        return self._trial_scheduler

    @trial_scheduler.setter
    def trial_scheduler (self, val):
        self._trial_scheduler = val


    @staticmethod
    def _set_range(obj, cs):
        if obj.search_space is not None:
            cs.add_configuration_space(prefix='',
                                       delimiter='',
                                       configuration_space=obj.search_space)

    @staticmethod
    def _assert_fit_error(obj, name):
        assert obj is not None, '%s cannot be None' % name

    @staticmethod
    def _init_args(cs, metadata):
        """Initialize arguments using configuration space and metadata.

         Args:
            cs: configuration space.
            metadata: metadata including the search space
        """
        args = argparse.Namespace()
        vars(args).update({'data': metadata['data']})
        vars(args).update({'backend': metadata['backend']})
        vars(args).update({'epochs': metadata['resources_per_trial']['num_training_epochs']})
        vars(args).update({'num_gpus': metadata['resources_per_trial']['num_gpus']})
        vars(args).update({'savedir': metadata['savedir']})
        for k, v in metadata['kwargs'].items():
            vars(args).update({k: v})
        for hparam in cs.get_hyperparameters():
            vars(args).update({hparam.name: hparam.default_value})
        return args

    @staticmethod
    def _reset_checkpoint(metadata):
        dir = os.path.splitext(metadata['savedir'])[0]
        if not metadata['resume'] and os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)

    @staticmethod
    def _construct_search_space(metadata):
        """Construct search space using metadata.

         Args:
            metadata: metadata including the search space
        """
        cs = CS.ConfigurationSpace()
        for name, obj in metadata.items():
            if hasattr(obj, 'search_space'):
                BaseTask._set_range(obj, cs)
            elif name == 'kwargs':
                for k, v in obj.items():
                    if hasattr(v, 'search_space'):
                        BaseTask._set_range(v, cs)
            elif obj is None and name == 'data':
                BaseTask._assert_fit_error(obj, name)
        args = BaseTask._init_args(cs, metadata)
        return cs, args

    @staticmethod
    def _reinitialize_stop_resources(metadata):
        for k, _ in metadata['stop_criterion'].items():
            if k in metadata['kwargs']:
                metadata['stop_criterion'][k] = metadata['kwargs'][k]
        for k, _ in metadata['resources_per_trial'].items():
            if k in metadata['kwargs']:
                metadata['resources_per_trial'][k] = metadata['kwargs'][k]

    @staticmethod
    def _run_backend(cs, args, metadata, start_time):
        """Construct search space using metadata.

         Args:
            cs: configuration space.
            args: the argument parser.
            metadata: the metadata including the search space.
            start_time: the start time of the fit function.
        """

        if metadata['searcher'] is None or metadata['searcher'] == 'random':
            searcher = ag.searcher.RandomSampling(cs)
        elif isinstance(metadata['searcher'], BaseSearcher):
            searcher = metadata['searcher']
        else:
            raise NotImplementedError
        if metadata['trial_scheduler'] == 'hyperband':
            print('hyperband!!!')
            time.sleep(10)
            BaseTask.trial_scheduler = ag.distributed.DistributedHyperbandScheduler(
                metadata['kwargs']['train_func'],
                args,
                {'num_cpus': 1,
                 'num_gpus': int(metadata['resources_per_trial']['num_gpus'])},
                searcher,
                checkpoint=metadata['savedir'],
                resume=metadata['resume'],
                time_attr='epoch',
                reward_attr=metadata['kwargs']['reward_attr'],
                max_t=metadata['resources_per_trial'][
                    'num_training_epochs'],
                grace_period=metadata['resources_per_trial'][
                                 'num_training_epochs'] // 4,
                visualizer=metadata['visualizer'])
            # TODO (cgraywang): use empiral val now
        else:
            print('fifo!!!')
            time.sleep(10)
            BaseTask.trial_scheduler = ag.distributed.DistributedFIFOScheduler(
                metadata['kwargs']['train_func'],
                args,
                {'num_cpus': 1,
                 'num_gpus': int(metadata['resources_per_trial']['num_gpus'])},
                searcher,
                checkpoint=metadata['savedir'],
                resume=metadata['resume'],
                time_attr='epoch',
                reward_attr=metadata['kwargs']['reward_attr'],
                visualizer=metadata['visualizer'])

        BaseTask.trial_scheduler.run_with_stop_criterion(start_time, metadata['stop_criterion'])
        BaseTask.trial_scheduler.join_tasks()
        final_metric = BaseTask.trial_scheduler.searcher.get_best_reward()
        final_config = BaseTask.trial_scheduler.searcher.get_best_config()

        BaseTask.final_fit(args, final_config, metadata)#metadata
        ctx = [mx.gpu(i) for i in range(metadata['resources_per_trial']['num_gpus'])] \
            if metadata['resources_per_trial']['num_gpus'] > 0 else [mx.cpu()]

        net = get_model(final_config['model'], pretrained=final_config['pretrained'])
        with net.name_scope():
            num_classes = metadata['data'].num_classes
            if hasattr(net, 'output'):
                net.output = nn.Dense(num_classes)
            else:
                net.fc = nn.Dense(num_classes)
        if not final_config['pretrained']:
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        else:
            if hasattr(net, 'output'):
                net.output.initialize(mx.init.Xavier(), ctx=ctx)
            else:
                net.fc.initialize(mx.init.Xavier(), ctx=ctx)
            net.collect_params().reset_ctx(ctx)
        net_path = os.path.join(os.path.splitext(args.savedir)[0], 'net.params')
        net.load_parameters(net_path, ctx=ctx)
        return net, final_metric, final_config

    # TODO: fix
    @staticmethod
    def predict(img):
        """The task predict function given an input.

         Args:
            img: the input

         Example:
            >>> ind, prob = task.predict('example.jpg')
        """
        logger.info('Start predicting.')
        ctx = [mx.gpu(i) for i in range(BaseTask.result.metadata['resources_per_trial']['num_gpus'])] \
            if BaseTask.result.metadata['resources_per_trial']['num_gpus'] > 0 else [mx.cpu()]
        # img = utils.download(img)
        img = mx.image.imread(img)
        plt.imshow(img.asnumpy())
        plt.show()
        transform_fn = transforms.Compose(BaseTask.result.metadata['data'].transform_val_list)
        img = transform_fn(img)
        pred = BaseTask.result.model(img.expand_dims(axis=0).as_in_context(ctx[0]))
        ind = mx.nd.argmax(pred, axis=1).astype('int')
        logger.info('Finished.')
        mx.nd.waitall()
        return ind, mx.nd.softmax(pred)[0][ind]

    @staticmethod
    def evaluate(data):
        """The task evaluation function given the test data.

         Args:
            data: test data

         Example:
            >>> test_acc = task.evaluate(data)
        """
        logger.info('Start evaluating.')
        L = get_loss_instance(BaseTask.result.metadata['losses'].loss_list[0].name)
        metric = get_metric_instance(BaseTask.result.metadata['metrics'].metric_list[0].name)
        ctx = [mx.gpu(i) for i in range(BaseTask.result.metadata['resources_per_trial']['num_gpus'])] \
            if BaseTask.result.metadata['resources_per_trial']['num_gpus'] > 0 else [mx.cpu()]
        def _init_dataset(dataset, transform_fn, transform_list):
            if transform_fn is not None:
                dataset = dataset.transform(transform_fn)
            if transform_list is not None:
                dataset = dataset.transform_first(transforms.Compose(transform_list))
            return dataset

        test_dataset = _init_dataset(data.test,
                                     data.transform_val_fn,
                                     data.transform_val_list)
        test_data = mx.gluon.data.DataLoader(
            test_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers)
        test_loss = 0
        for i, batch in enumerate(test_data):
            data = mx.gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
            outputs = [BaseTask.result.model(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)
        _, test_acc = metric.get()
        test_loss /= len(test_data)
        logger.info('Finished.')
        mx.nd.waitall()
        return test_acc

    @staticmethod
    def final_fit(args, config, metadata):
        args = copy.deepcopy(args)
        logger.info('Start final fitting.')
        vars(args).update({'savedir': metadata['savedir']})
        args.data.split = 0
        args.train_func(args, config, reporter=None)
        mx.nd.waitall()

    @staticmethod
    def fit(data,
            nets=None,
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=None,
            losses=None,
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 10*60,
                'max_metric': 1.0,
                'num_trials': 40
            },
            resources_per_trial={
                'num_gpus': 1,
                'num_training_epochs': 10
            },
            backend='default',
            **kwargs):
        """
        Fit networks on dataset

        Args:
            data: Input data. task.Datasets.
            nets: autogluon.Nets.
            optimizers: autogluon.Optimizers.
            metrics: autogluon.Metrics.
            losses: autogluon.Losses.
            stop_criterion (dict): The stopping criteria.
            resources_per_trial (dict): Machine resources to allocate per trial.
            savedir (str): Local dir to save training results to.
            searcher: Search Algorithm.
            trial_scheduler: Scheduler for executing the experiment. Choose among FIFO (default) and HyperBand.
            resume (bool): If checkpoint exists, the experiment will resume from there.
            backend: support autogluon default backend.

        Example:
            >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> net_list = ['resnet18_v1', 'resnet34_v1']
            >>> nets = ag.Nets(net_list)
            >>> adam_opt = ag.optims.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
            >>>                           wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
            >>> sgd_opt = ag.optims.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
            >>>                         momentum=ag.space.Linear('momentum', 0.85, 0.95),
            >>>                         wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
            >>> optimizers = ag.Optimizers([adam_opt, sgd_opt])
            >>> searcher = 'random'
            >>> trial_scheduler = 'fifo'
            >>> savedir = 'checkpoint/demo.ag'
            >>> resume = False
            >>> time_limits = 3*60
            >>> max_metric = 1.0
            >>> num_trials = 4
            >>> stop_criterion = {
            >>>       'time_limits': time_limits,
            >>>       'max_metric': max_metric,
            >>>       'num_trials': num_trials
            >>> }
            >>> num_gpus = 1
            >>> num_training_epochs = 2
            >>> resources_per_trial = {
            >>>       'num_gpus': num_gpus,
            >>>       'num_training_epochs': num_training_epochs
            >>> }
            >>> results = task.fit(dataset,
            >>>                     nets,
            >>>                     optimizers,
            >>>                     searcher=searcher,
            >>>                     trial_scheduler=trial_scheduler,
            >>>                     resume=resume,
            >>>                     savedir=savedir,
            >>>                     stop_criterion=stop_criterion,
            >>>                     resources_per_trial=resources_per_trial)
        """
        logger.info('Start fitting')
        start_fit_time = time.time()

        logger.info('Start constructing search space')
        metadata = locals()
        BaseTask._reinitialize_stop_resources(metadata)
        cs, args = BaseTask._construct_search_space(metadata)
        logger.info('Finished.')

        logger.info('Start running trials')
        BaseTask._reset_checkpoint(metadata)
        final_model, final_metric, final_config = BaseTask._run_backend(cs, args, metadata,
                                                                        start_fit_time)
        logger.info('Finished.')

        logger.info('Finished.')
        BaseTask.result = Results(copy.deepcopy(final_model), copy.deepcopy(final_metric),
                                  copy.deepcopy(final_config),
                                  copy.deepcopy(time.time() - start_fit_time),
                                  copy.deepcopy(metadata))
        # BaseTask.trial_scheduler.shutdown()
        mx.nd.waitall()
        return BaseTask.result
