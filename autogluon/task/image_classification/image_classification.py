import logging

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from .losses import *
from .metrics import *
from ...optim import *#Optimizers, get_optim
from ...utils.data_analyzer import DataAnalyzer
from ..base import *
from .dataset import *
from .pipeline import *
from ...basic.space import *

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(BaseTask):
    @staticmethod
    def fit(data,
            net=ListSpace('resnet50v1b', 'ResNet50_v1d'),
        optimizer=ListSpace(
            SGD(learning_rate=LogLinearSpace(1e-4, 1e-2),
                momentum=0.9,
                wd=LogLinearSpace(1e-5, 1e-3)),
            NAG(learning_rate=LogLinearSpace(1e-4, 1e-2),
                momentum=0.9,
                wd=LogLinearSpace(1e-5, 1e-3))),
            metrics='Accuracy',
            losses=gluon.loss.SoftmaxCrossEntropyLoss,
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='none',
            stop_criterion={
                'time_limits': 1 * 60 * 60,
                'max_metric': 1.0,
                'max_trial_count': 2
            },
            resources_per_trial={
                'max_num_gpus': 0,
                'max_num_cpus': 4,
                'max_training_epochs': 3
            },
            backend='default',
            **kwargs):
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
            model: the parameters associated with the best model. (TODO:)
            val_accuracy: validation set accuracy
            config: best configuration
            time: total time cost
        """
        return BaseTask.fit(data, nets, optimizers, metrics, losses, searcher, trial_scheduler,
                            resume, savedir, visualizer, stop_criterion, resources_per_trial,
                            backend,
                            reward_attr='accuracy',
                            train_func=train_image_classification,
                            **kwargs)
