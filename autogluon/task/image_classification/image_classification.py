import copy
import logging

import mxnet as mx
from mxnet import gluon, nd

from ...core.optimizer import *
from ...core.loss import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .dataset import get_dataset
from .pipeline import train_image_classification
from .utils import *
from .nets import *
from .classifier import Classifier

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(BaseTask):
    """AutoGluon Task for classifying images based on their content
    """
    @staticmethod
    def Dataset(*args, **kwargs):
        """Dataset for AutoGluon image classification tasks. Can either be a 
    :class:`ImageFolderDataset`, :class:`RecordDataset`, or a 
    popular dataset already built into AutoGluon ('mnist', 'cifar10', 'cifar100', 'imagenet').

        Parameters
        ----------
        name : str, optional
            Which built-in dataset to use, will override all other options if specified.
            The options are ('mnist', 'cifar', 'cifar10', 'cifar100', 'imagenet')
        train : bool, default = True
            Whether this dataset should be used for training or validation.
        train_path : str
            The training data location. If using :class:`ImageFolderDataset`,
            image folder`path/to/the/folder` should be provided. 
            If using :class:`RecordDataset`, the `path/to/*.rec` should be provided.
        input_size : int
            The input image size.
        crop_ratio : float
            Center crop ratio (for evaluation only)
        """
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset,
            net=Categorical('ResNet50_v1b', 'ResNet18_v1b'),
            optimizer= SGD(learning_rate=Real(1e-3, 1e-2, log=True),
                           wd=Real(1e-4, 1e-3, log=True)),
            lr_scheduler='cosine',
            loss=SoftmaxCrossEntropyLoss(),
            split_ratio=0.8,
            batch_size=64,
            input_size=224,
            epochs=20,
            metric='accuracy',
            nthreads_per_trial=4,
            ngpus_per_trial=1,
            hybridize=True,
            search_strategy='random',
            plot_results=False,
            verbose=False,
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
        Fit image classification models to a given dataset.

        Parameters
        ----------
        dataset : str or :meth:`autogluon.task.ImageClassification.Dataset`
            Training dataset containing images and their associated class labels.
        input_size : int
            Size of images in the dataset (pixels).
        net : str or :class:`autogluon.space.AutoGluonObject`
            Which existing neural network models to consider as candidates.
        optimizer : str or :class:`autogluon.space.AutoGluonObject`
            Which optimizers to consider as candidates for learning the neural network weights.
        lr_scheduler : str
            Describes how learning rate should be adjusted over the course of training. Options include: 'cosine', 'poly'.
        batch_size : int
            How many images to group in each mini-batch during gradient computations in training.
        epochs: int
            How many epochs to train the neural networks for at most.
        metric : str or object
            Evaluation metric by which predictions will be ulitmately evaluated on test data.
        loss : mxnet.gluon.loss
            Loss function used during training of the neural network weights.
        num_trials : int
            Maximal number of hyperparameter configurations to try out.
        split_ratio : float, defaut = 0.8
            Fraction of dataset to use for training (rest of data is held-out for tuning hyperparameters).
            The final returned model may be fit to all of the data (after hyperparameters have been selected).
        time_limits : int
            Approximately how long should `fit()` should run for (wallclock time in seconds).
            `fit()` will stop training new models after this amount of time has elapsed (but models which have already started training will continue to completion). 
        nthreads_per_trial : (int)
            How many CPUs to use in each trial (ie. single training run of a model).
        ngpus_per_trial : (int)
            How many GPUs to use in each trial (ie. single training run of a model). 
        checkpoint:
            The path to local directory where trained models will be saved.
        search_strategy : str
            Which hyperparameter search algorithm to use. 
            Options include: 'random' (random search), 'skopt' (SKopt Bayesian optimization), 'grid' (grid search), 'hyperband' (Hyperband), 'rl' (reinforcement learner)
        search_options : (dict)
            Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization. 
        resume : bool
            If a model checkpoint file exists, model training will resume from there when specified.
        dist_ip_addrs : list
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        verbose : bool
            Whether or not to print out intermediate information during training.
        plot_results : bool
            Whether or not to generate plots summarizing training process.
        visualizer : str
            Describes method to visualize training progress during `fit()`. Options: ['mxboard', 'tensorboard', 'none']. 
        grace_period : int
            The grace period in early stopping when using Hyperband to tune hyperparameters. If None, this is set automatically.
        auto_search : bool
            If True, enables automatic suggestion of network types and hyper-parameter ranges adaptively based on provided dataset.
        
        Returns
        -------
            :class:`autogluon.task.image_classification.Classifier` object which can make predictions on new data and summarize what happened during `fit()`.
        
        Examples
        --------
        >>> from autogluon import ImageClassification as task
        >>> train_data = task.Dataset(train_path='~/data/train')
        >>> classifier = task.fit(train_data,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limits=600, ngpus_per_trial=1, num_trials=4)
        >>> test_data = task.Dataset('~/data/test', train=False)
        >>> test_acc = classifier.evaluate(test_data)
        """
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            net = auto_suggest_network(dataset, net)

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        train_image_classification.register_args(
            dataset=dataset,
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            metric=metric,
            num_gpus=ngpus_per_trial,
            split_ratio=split_ratio,
            batch_size=batch_size,
            input_size=input_size,
            epochs=epochs,
            verbose=verbose,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            final_fit=False)

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
            'plot_results': plot_results,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'searcher': 'random',
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})

        results = BaseTask.run_fit(train_image_classification, search_strategy,
                                   scheduler_options)
        args = sample_config(train_image_classification.args, results['best_config'])

        model = get_network(args.net, results['num_classes'], mx.cpu(0))
        update_params(model, results.pop('model_params'))
        return Classifier(model, results, default_val_fn, checkpoint, args)
