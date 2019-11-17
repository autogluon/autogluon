import copy
import logging

import mxnet as mx
from mxnet import gluon, nd

from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .dataset import get_dataset
from .pipeline import train_image_classification
from .utils import *
from .classifier import Classifier

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(BaseTask):
    """AutoGluon ImageClassification Task
    """
    @staticmethod
    def Dataset(*args, **kwargs):
        """A convenient function for image classification dataset, supported datasets given by
        built-in datasets ('mnist', 'cifar10', 'cifar100', 'imagenet'),
        :class:`ImageFolderDataset` and :class:`RecordioDataset`.

        Parameters
        ----------
            name : str, optional
                The name for built-in dataset, overrite other options.
                The options are ('mnist', 'cifar', 'cifar10', 'cifar100', 'imagenet')
            train : bool, default True
                Train or validation mode
            train_path : str
                The training data location
            input_size : int
                The input image size.
            crop_ratio : float
                Center crop ratio for evaluation only
        """
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset,
            net=Categorical('ResNet18_v1b', 'ResNet50_v1b'),
            optimizer= SGD(learning_rate=Real(1e-4, 1e-2, log=True),
                           wd=Real(1e-5, 1e-3, log=True)),
            lr_scheduler='cosine',
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=64,
            input_size=224,
            epochs=20,
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
            auto_search=True):

        """
        Auto fit on image classification dataset

        Parameters
        ----------
        dataset : (str or autogluon.task.ImageClassification.Dataset)
            Training dataset.
        net : (str or :class:`autogluon.AutoGluonObject`)
            Network candidates.
        optimizer : (str or :class:`autogluon.AutoGluonObject`)
            optimizer candidates.
        metric : (str or object)
            observation metric.
        loss : (mxnet.gluon.loss)
            training loss function.
        num_trials : (int)
            number of trials in the experiment.
        time_limits : (int)
            training time limits in seconds.
        resources_per_trial : (dict)
            Machine resources to allocate per trial.
        savedir : (str)
            Local dir to save training results to.
        search_strategy : (str)
            Search Algorithms ('random', 'bayesopt' and 'hyperband')
        resume : (bool)
            If checkpoint exists, the experiment will resume from there.

        Examples
        --------
        >>> dataset = task.Dataset(train_path='~/data/train',
        >>>                        test_path='data/test')
        >>> results = task.fit(dataset,
        >>>                    nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                    time_limits=time_limits,
        >>>                    ngpus_per_trial=1,
        >>>                    num_trials = 4)
        """
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

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
            batch_size=batch_size,
            input_size=input_size,
            epochs=epochs,
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
