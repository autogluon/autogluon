import logging
<<<<<<< HEAD
import matplotlib.pyplot as plt

import mxnet as mx
=======
import numpy as np

>>>>>>> awslabs/master
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

<<<<<<< HEAD
from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ..base import BaseTask

from .nets import get_built_in_network
from .dataset import ImageClassificationDataset
from .pipeline import train_image_classification
from .metrics import get_metric_instance
=======
from .model_zoo import *
from .losses import *
from .metrics import *
from ...network import Nets
from ...optimizer import Optimizers, get_optim
from ...loss import Losses
from ...metric import Metrics
from ...utils.data_analyzer import DataAnalyzer
from ..base import *
from .dataset import *
from .pipeline import *
>>>>>>> awslabs/master

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

<<<<<<< HEAD
class ImageClassification(BaseTask):
    Dataset = ImageClassificationDataset
    @staticmethod
    def fit(dataset='cifar10',
            net=List('ResNet34_v1b', 'ResNet50_v1b'),
            optimizer=List(
                SGD(learning_rate=LogLinear(1e-4, 1e-2),
                          momentum=LogLinear(0.85, 0.95),
                          wd=LogLinear(1e-5, 1e-3)),
                Adam(learning_rate=LogLinear(1e-4, 1e-2),
                     wd=LogLinear(1e-5, 1e-3)),
            ),
            lr_scheduler='cosine',
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=64,
            epochs=20,
            metric='accuracy',
            num_cpus=4,
            num_gpus=1,
            algorithm='random',
            time_limits=None,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=[],
            grace_period=None,
            auto_search=True):

        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        train_image_classification.update(
            dataset=dataset,
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            metric=metric,
            num_gpus=num_gpus,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_cpus,
            final_fit=False)

        scheduler_options = {
            'resource': {'num_cpus': num_cpus, 'num_gpus': num_gpus},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'reward',
            'dist_ip_addrs': dist_ip_addrs,
        }
        if algorithm == 'hyperband':
            scheduler_options.update({
                'max_t': args.epochs,
                'grace_period': grace_period if grace_period else args.epochs//4})

        return BaseTask.run_fit(train_image_classification, algorithm, scheduler_options)

    @classmethod
    def predict(cls, img):
        """The task predict function given an input.
         Args:
            img: the input
         Example:
            >>> ind, prob = task.predict('example.jpg')
        """
        logger.info('Start predicting.')
        # load and display the image
        img = mx.image.imread(img)
        plt.imshow(img.asnumpy())
        plt.show()
        # model inference
        args = cls.scheduler.train_fn.args
        dataset = args.dataset
        if isinstance(dataset, AutoGluonObject):
            dataset = dataset._lazy_init()
        transform_fn = dataset.transform_val
        img = transform_fn(img)
        ctx = mx.gpu(0)if args.num_gpus > 0 else mx.cpu()
        pred = cls.results.model(img.expand_dims(0).as_in_context(ctx))
        ind = mx.nd.argmax(pred, axis=1).astype('int')
        logger.info('Finished.')
        mx.nd.waitall()
        return ind, mx.nd.softmax(pred)[0][ind]

    @classmethod
    def evaluate(cls, dataset):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import ImageClassification as task
            >>> dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
            >>> test_reward = task.evaluate(dataset)
        """
        logger.info('Start evaluating.')
        args = cls.scheduler.train_fn.args
        batch_size = args.batch_size * max(args.num_gpus, 1)

        metric = get_metric_instance(args.metric)
        ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        cls.results.model.collect_params().reset_ctx(ctx)

        if isinstance(dataset, AutoGluonObject):
            dataset = dataset._lazy_init()
        test_data = gluon.data.DataLoader(
            dataset.test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        for i, batch in enumerate(test_data):
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                                 batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx,
                                                  batch_axis=0, even_split=False)
            outputs = [cls.results.model(X) for X in data]
            metric.update(label, outputs)
        _, test_reward = metric.get()
        logger.info('Finished.')
        return test_reward
=======

class ImageClassification(BaseTask):
    class Dataset(BaseTask.Dataset):
        """The image classification dataset.

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
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=64, num_workers=4,
                     transform_train_fn=None, transform_val_fn=None,
                     transform_train_list=[
                         transforms.Resize(480),
                         transforms.RandomResizedCrop(224),
                         transforms.RandomFlipLeftRight(),
                         transforms.RandomColorJitter(brightness=0.4,
                                                      contrast=0.4,
                                                      saturation=0.4),
                         transforms.RandomLighting(0.1),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                     transform_val_list=[
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                     batchify_train_fn=None, batchify_val_fn=None,
                     **kwargs):
            super(ImageClassification.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                          transform_train_fn, transform_val_fn,
                                          transform_train_list, transform_val_list,
                                          batchify_train_fn, batchify_val_fn, **kwargs)
            self._split = None
            # self._kwargs = kwargs
            self._read_dataset(**kwargs)
            # TODO (cgraywang): add search space, handle batch_size, num_workers
            self._add_search_space()

        @property
        def split(self):
            return self._split

        @split.setter
        def split(self, value):
            self._split = value

        def _read_dataset(self, **kwargs):
            import time
            # TODO (cgraywang): put transform in the search space
            try:
                self.train = get_dataset(self.name, train=True)
                self.val = None
                self.test = get_dataset(self.name, train=False)
                self.num_classes = len(np.unique(self.train._label))
                # self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
                # DataAnalyzer.check_dataset(self.train, self.val)
            except ValueError:
                if self.train_path is not None:
                    if '.rec' not in self.train_path:
                        self.train = gluon.data.vision.ImageFolderDataset(self.train_path)
                        if self.val_path is not None:
                            self.val = gluon.data.vision.ImageFolderDataset(self.val_path)
                        else:
                            self.val = None
                        if 'test_path' in kwargs:
                            self.test = gluon.data.vision.ImageFolderDataset(kwargs['test_path'])
                        self.num_classes = len(np.unique([e[1] for e in self.train]))
                    elif '.rec' in self.train_path:
                        self.train = gluon.data.vision.ImageRecordDataset(self.train_path)
                        if self.val_path is not None:
                            self.val = gluon.data.vision.ImageRecordDataset(self.val_path)
                        else:
                            self.val = None
                        if 'test_path' in kwargs:
                            self.test = gluon.data.vision.ImageRecordDataset(kwargs['test_path'])
                        self.num_classes = len(np.unique([e[1] for e in self.train]))
                elif 'test_path' in kwargs:
                    if '.rec' not in kwargs['test_path']:
                        self.test = gluon.data.vision.ImageFolderDataset(kwargs['test_path'])
                    elif '.rec' in kwargs['test_path']:
                        self.test = gluon.data.vision.ImageRecordDataset(kwargs['test_path'])
                else:
                    raise NotImplementedError
            self.split = np.random.choice(range(1, 10), 1)[0]

        def _add_search_space(self):
            pass

        def _get_search_space_strs(self):
            pass

        def __repr__(self):
            repr_str = ""
            # try:
            #     train_stats = DataAnalyzer.stat_dataset(self.train)
            #     val_stats = DataAnalyzer.stat_dataset(self.val)
            #     repr_str = "AutoGluon Dataset: " \
            #                "\n ======== " \
            #                "\n name = %s" \
            #                "\n ======== " \
            #                "\n Train data statistic " \
            #                "\n number of classes = %d" \
            #                "\n number of samples = %d" \
            #                "\n mean (label) = %.2f" \
            #                "\n std (label) = %.2f" \
            #                "\n var (label) = %.2f" \
            #                "\n ======== " \
            #                "\n Val data statistic " \
            #                "\n number of classes = %d" \
            #                "\n number of samples = %d" \
            #                "\n mean (label) = %.2f" \
            #                "\n std (label) = %.2f" \
            #                "\n var (label) = %.2f" % (self.name,
            #                                           train_stats[0], train_stats[1],
            #                                           train_stats[2], train_stats[3],
            #                                           train_stats[4],
            #                                           val_stats[0], val_stats[1],
            #                                           val_stats[2], val_stats[3],
            #                                           val_stats[4])
            # except AttributeError:
            #     # TODO: add more info for folder dataset
            #     repr_str = "AutoGluon Dataset: " \
            #                "\n ======== " \
            #                "\n name = %s" \
            #                "\n ======== " % self.name
            return repr_str

    @staticmethod
    def fit(data,
            nets=Nets([
                get_model('resnet18_v1'),
                get_model('resnet34_v1')]),
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=Metrics([
                get_metric('Accuracy')]),
            losses=Losses([
                get_loss('SoftmaxCrossEntropyLoss')]),
            searcher='random',
            trial_scheduler='fifo',
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 24*60*60,
                'max_metric': 1.0,
                'num_trials': 100
            },
            resources_per_trial={
                'num_gpus': 1,
                'num_training_epochs': 100
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
        return BaseTask.fit(data, nets, optimizers, metrics, losses, searcher, trial_scheduler,
                            resume, savedir, visualizer, stop_criterion, resources_per_trial,
                            backend,
                            reward_attr='accuracy',
                            train_func=train_image_classification,
                            **kwargs)
>>>>>>> awslabs/master
