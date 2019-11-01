import logging

import mxnet as mx
from mxnet import gluon, nd

from ...core.optimizer import *
from .dataset import *
from ..base import BaseTask

__all__ = ['ObjectDetection']

logger = logging.getLogger(__name__)

class ObjectDetection(BaseTask):
    r"""
    AutoGluon Object Detection Task
    """
    
    @staticmethod
    def Dataset(*args, **kwargs):
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset='voc',
            net=Choice('ResNet34_v1b', 'ResNet50_v1b'),
            optimizer=Choice(
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
            search_strategy='random',
            search_options={},
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
            >>> dataset = taskDataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> results = task.fit(dataset,
            >>>                    nets=ag.Choice['resnet18_v1', 'resnet34_v1'],
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
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})

        return BaseTask.run_fit(train_image_classification, search_strategy, scheduler_options)