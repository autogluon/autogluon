import logging

import mxnet as mx
from mxnet import gluon, nd

from ...core.optimizer import *
from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask

from .dataset import *
from .pipeline import train_object_detection
from .utils import *
from ...utils import update_params

from .detector import Detector

__all__ = ['ObjectDetection']

logger = logging.getLogger(__name__)

class ObjectDetection(BaseTask):
    """AutoGluon ImageClassification Task
    """
    @staticmethod
    def Dataset(*args, **kwargs):
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset='voc',
            net=Categorical('mobilenet1.0'),
            lr=Categorical(5e-4, 1e-4),
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=16,
            epochs=200,
            num_trials=2,
            nthreads_per_trial=12,
            num_workers=32,
            ngpus_per_trial=1,
            hybridize=True,
            search_strategy='random',
            search_options={},
            time_limits=None,
            verbose=False,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            dist_ip_addrs=[],
            grace_period=None,
            auto_search=True,
            seed=223,
            data_shape=416,
            start_epoch=0,
            lr_mode='step',
            lr_decay=0.1,
            lr_decay_period=0,
            lr_decay_epoch='160,180',
            warmup_lr=0.0,
            warmup_epochs=2,
            momentum=0.9,
            wd=0.0005,
            log_interval=100,
            save_prefix='',
            save_interval=10,
            val_interval=1,
            num_samples=-1,
            no_random_shape=False,
            no_wd=False,
            mixup=False,
            no_mixup_epochs=20,
            label_smooth=False,
            syncbn=False,
            ):

        """
        Auto fit on object detection dataset

        Parameters
        ----------
        dataset : str or :meth:`autogluon.task.ObjectDectection.Dataset`
            Training dataset.
        net : str, :class:`autogluon.AutoGluonObject`
            Network candidates.
        optimizer : str, :class:`autogluon.AutoGluonObject`
            optimizer candidates.
         metric : str or object
            observation metric.
        loss : mxnet.gluon.loss
            training loss function.
        num_trials : int
            number of trials in the experiment.
        time_limits : int
            training time limits in seconds.
        resources_per_trial : dict
            Machine resources to allocate per trial.
        savedir : str
            Local dir to save training results to.
        search_strategy : str or callable
            Search Algorithms ('random', 'bayesopt' and 'hyperband')
        resume : bool, default False
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

        train_object_detection.register_args(
            dataset=dataset,
            net=net,
            lr = lr,
            loss=loss,
            num_gpus=ngpus_per_trial,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            verbose=verbose,
            final_fit=False,
            seed=seed,
            data_shape=data_shape,
            start_epoch=0,
            lr_mode=lr_mode,
            lr_decay=lr_decay,
            lr_decay_period=lr_decay_period,
            lr_decay_epoch=lr_decay_epoch,
            warmup_lr=warmup_lr,
            warmup_epochs=warmup_epochs,
            momentum=momentum,
            wd=wd,
            log_interval=log_interval,
            save_prefix=save_prefix,
            save_interval=save_interval,
            val_interval=val_interval,
            num_samples=num_samples,
            no_random_shape=no_random_shape,
            no_wd=no_wd,
            mixup=mixup,
            no_mixup_epochs=no_mixup_epochs,
            label_smooth=label_smooth,
            resume=resume,
            syncbn=syncbn)

        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'map_reward',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'searcher': 'random',
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})
                
        results = BaseTask.run_fit(train_object_detection, search_strategy,
                                   scheduler_options)
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        args = sample_config(train_object_detection.args, results['best_config'])
        logger.info('The best config:\n', results['best_config'])

        model = get_network(args.net, dataset.init().get_classes(), mx.cpu(0))
        update_params(model, results.pop('model_params'))
        return Detector(model, results, checkpoint, args)
