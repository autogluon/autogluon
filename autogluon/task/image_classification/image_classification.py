import logging
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ..base import BaseTask

from .nets import get_built_in_network
from .dataset import ImageClassificationDataset
from .pipeline import train_image_classification
from .metrics import get_metric_instance

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(BaseTask):
    """AutoGluon ImageClassification Task
    """
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

        """
        Fit networks on dataset

        Args:
            dataset (str or autogluon.task.ImageClassification.Dataset): Training dataset.
            net (str, autogluon.AutoGluonObject, or ag.List of AutoGluonObject): Network candidates.
            optimizer (str, autogluon.AutoGluonObject, or ag.List of AutoGluonObject): optimizer candidates.
            metric (str or object): observation metric.
            loss (object): training loss function.
            num_trials (int): number of trials in the experiment.
            time_limits (int): training time limits in seconds.
            resources_per_trial (dict): Machine resources to allocate per trial.
            savedir (str): Local dir to save training results to.
            algorithm (str): Search Algorithms ('random', 'bayesian' and 'hyperband')
            resume (bool): If checkpoint exists, the experiment will resume from there.


        Example:
            >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> results = task.fit(dataset,
            >>>                    nets=ag.List['resnet18_v1', 'resnet34_v1'],
            >>>                    time_limits=time_limits,
            >>>                    num_gpus=1,
            >>>                    num_trials = 4)
        """
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
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})

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
