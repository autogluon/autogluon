import logging

import mxnet as mx
from mxnet import gluon
import copy

from .dataset import get_dataset
from .detector import Detector
from .pipeline import train_object_detection
from .utils import get_network
from ..base import BaseTask, compile_scheduler_options
from ...core.decorator import sample_config
from ...core.space import Categorical
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ...utils import update_params

__all__ = ['ObjectDetection']

logger = logging.getLogger(__name__)


class ObjectDetection(BaseTask):
    """AutoGluon Task for detecting and locating objects in images
    """

    @staticmethod
    def Dataset(*args, **kwargs):
        """ Dataset of images in which to detect objects.
        """
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset='voc',
            net=Categorical('mobilenet1.0'),
            meta_arch='yolo3',
            lr=Categorical(5e-4, 1e-4),
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            split_ratio=0.8,
            batch_size=16,
            epochs=50,
            num_trials=None,
            time_limits=None,
            nthreads_per_trial=12,
            num_workers=32,
            ngpus_per_trial=1,
            hybridize=True,
            scheduler_options=None,
            search_strategy='random',
            search_options=None,
            verbose=False,
            transfer='coco',
            resume='',
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            dist_ip_addrs=None,
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
            warmup_iters=1000,
            warmup_factor=1. / 3.,
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
            reuse_pred_weights=True,
            **kwargs):

        """
        Fit object detection models.

        Parameters
        ----------
        dataset : str or :class:`autogluon.task.ObjectDectection.Dataset`
            Training dataset containing images and corresponding object bounding boxes.
        net : str, :class:`autogluon.space.AutoGluonObject`
            Which existing neural network base models to consider as candidates.
        meta_arch : str
            Meta architecture of the model. Currently support YoloV3 (Default) and FasterRCNN.
            YoloV3 is faster, while FasterRCNN is more accurate.
        lr : float or :class:`autogluon.space`
            The learning rate to use in each update of the neural network weights during training.
        loss : mxnet.gluon.loss
            Loss function used during training of the neural network weights.
        split_ratio : float
            Fraction of dataset to hold-out during training in order to tune hyperparameters (i.e. validation data).
            The final returned model may be fit to all of the data (after hyperparameters have been selected).
        batch_size : int
            How many images to group in each mini-batch during gradient computations in training.
        epochs : int
            How many epochs to train the neural networks for at most.
        num_trials : int
            Maximal number of hyperparameter configurations to try out.
        time_limits : int
            Approximately how long should `fit()` should run for (wallclock time in seconds).
            `fit()` will stop training new models after this amount of time has elapsed (but models which have already started training will continue to completion).
        nthreads_per_trial : int
            How many CPUs to use in each trial (ie. single training run of a model).
        num_workers : int
            How many CPUs to use for data loading during training of a model.
        ngpus_per_trial : int
            How many GPUs to use in each trial (ie. single training run of a model). 
        hybridize : bool
            Whether or not the MXNet neural network should be hybridized (for increased efficiency).
        scheduler_options : dict
            Extra arguments passed to __init__ of scheduler, to configure the
            orchestration of training jobs during hyperparameter-tuning.
        search_strategy : str
            Which hyperparameter search algorithm to use.
            Options include: 'random' (random search), 'skopt' (SKopt Bayesian
            optimization), 'grid' (grid search), 'hyperband' (Hyperband random),
            'rl' (reinforcement learner).
        search_options : dict
            Auxiliary keyword arguments to pass to the searcher that performs
            hyperparameter optimization.
        verbose : bool
            Whether or not to print out intermediate information during training.
        resume : str
            Path to checkpoint file of existing model, from which model training
            should resume.
            If not empty, we also start the hyperparameter search from the state
            loaded from checkpoint.
        checkpoint : str or None
            State of hyperparameter search is stored to this local file
        visualizer : str
            Describes method to visualize training progress during `fit()`. Options: ['mxboard', 'tensorboard', 'none']. 
        dist_ip_addrs : list
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        auto_search : bool
            If True, enables automatic suggestion of network types and hyper-parameter ranges adaptively based on provided dataset.
        seed : int
            Random seed to set for reproducibility.
        data_shape : int
            Shape of the image data.
        start_epoch : int
            Which epoch we begin training from 
            (eg. if we resume training of an existing model, then this
            argument may be set to the number of epochs the model has already been trained for previously).
        lr_mode : str
            What sort of learning rate schedule should be followed during training.
        lr_decay : float
            How much learning rate should be decayed during training.
        lr_decay_period : int
            How often learning rate should be decayed during training.
        warmup_lr : float
            Learning rate to use during warm up period at the start of training.
        warmup_epochs : int
            How many initial epochs constitute the "warm up" period of model training.
        warmup_iters : int
            How many initial iterations constitute the "warm up" period of model training.
            This is used by R-CNNs
        warmup_factor : float
            warmup factor of target lr. initial lr starts from target lr * warmup_factor
        momentum : float or  :class:`autogluon.space`
            Momentum to use in optimization of neural network weights during training.
        wd : float or :class:`autogluon.space`
            Weight decay to use in optimization of neural network weights during training.
        log_interval : int
            Log results every so many epochs during training.
        save_prefix : str
            Prefix to append to file name for saved model.
        save_interval : int
            Save a copy of model every so many epochs during training.
        val_interval : int
            Evaluate performance on held-out validation data every so many epochs during training.
        no_random_shape : bool
            Whether random shapes should not be used.
        no_wd : bool
            Whether weight decay should be turned off.
        mixup : bool
            Whether or not to utilize mixup data augmentation strategy.
        no_mixup_epochs : int
            If using mixup, we first train model for this many epochs without mixup data augmentation.
        label_smooth : bool
            Whether or not to utilize label smoothing.
        syncbn : bool
            Whether or not to utilize synchronized batch normalization.
        
        Returns
        -------
            :class:`autogluon.task.object_detection.Detector` object which can make predictions on new data and summarize what happened during `fit()`.
        
        Examples
        --------
        >>> from autogluon import ObjectDetection as task
        >>> detector = task.fit(dataset = 'voc', net = 'mobilenet1.0',
        >>>                     time_limits = 600, ngpus_per_trial = 1, num_trials = 1)
        """
        assert search_strategy not in {'bayesopt', 'bayesopt_hyperband'}, \
            "search_strategy == 'bayesopt' or 'bayesopt_hyperband' not yet supported"
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        if ngpus_per_trial > get_gpu_count():
            logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs.")
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        # If only time_limits is given, the scheduler starts trials until the
        # time limit is reached
        if num_trials is None and time_limits is None:
            num_trials = 2

        train_object_detection.register_args(
            meta_arch=meta_arch,
            dataset=dataset,
            net=net,
            lr=lr,
            loss=loss,
            num_gpus=ngpus_per_trial,
            batch_size=batch_size,
            split_ratio=split_ratio,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            verbose=verbose,
            final_fit=False,
            seed=seed,
            data_shape=data_shape,
            start_epoch=0,
            transfer=transfer,
            lr_mode=lr_mode,
            lr_decay=lr_decay,
            lr_decay_period=lr_decay_period,
            lr_decay_epoch=lr_decay_epoch,
            warmup_lr=warmup_lr,
            warmup_epochs=warmup_epochs,
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
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
            syncbn=syncbn,
            reuse_pred_weights=reuse_pred_weights)

        # Backward compatibility:
        grace_period = kwargs.get('grace_period')
        if grace_period is not None:
            if scheduler_options is None:
                scheduler_options = {'grace_period': grace_period}
            else:
                assert 'grace_period' not in scheduler_options, \
                    "grace_period appears both in scheduler_options and as direct argument"
                scheduler_options = copy.copy(scheduler_options)
                scheduler_options['grace_period'] = grace_period
            logger.warning(
                "grace_period is deprecated, use "
                "scheduler_options={'grace_period': ...} instead")
        scheduler_options = compile_scheduler_options(
            scheduler_options=scheduler_options,
            search_strategy=search_strategy,
            search_options=search_options,
            nthreads_per_trial=nthreads_per_trial,
            ngpus_per_trial=ngpus_per_trial,
            checkpoint=checkpoint,
            num_trials=num_trials,
            time_out=time_limits,
            resume=(len(resume) > 0),
            visualizer=visualizer,
            time_attr='epoch',
            reward_attr='map_reward',
            dist_ip_addrs=dist_ip_addrs,
            epochs=epochs)
        results = BaseTask.run_fit(
            train_object_detection, search_strategy, scheduler_options)
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        args = sample_config(train_object_detection.args, results['best_config'])
        logger.info('The best config: {}'.format(results['best_config']))

        ctx = [mx.gpu(i) for i in range(get_gpu_count())]
        model = get_network(args.meta_arch, args.net, dataset.init().get_classes(), transfer, ctx,
                            syncbn=args.syncbn)
        update_params(model, results.pop('model_params'))
        return Detector(model, results, checkpoint, args)
