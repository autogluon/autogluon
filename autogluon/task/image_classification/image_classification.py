import logging
import os
import copy
import mxnet as mx

from .classifier import Classifier
from .dataset import get_dataset
from .nets import *
from .pipeline import train_image_classification
from .utils import *
from ..base import BaseTask, compile_scheduler_options, create_scheduler
from ...core import *
from ...core.loss import *
from ...core.optimizer import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ...utils import update_params

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)


class ImageClassification(BaseTask):
    """AutoGluon Task for classifying images based on their content
    """
    Classifier = Classifier

    @staticmethod
    def Dataset(path=None, name=None, train=True, input_size=224, crop_ratio=0.875, *args, **kwargs):
        """Dataset for AutoGluon image classification tasks.
           May either be a :class:`autogluon.task.image_classification.ImageFolderDataset`, :class:`autogluon.task.image_classification.RecordDataset`,
           or a popular dataset already built into AutoGluon ('mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagenet').

        Parameters
        ----------
        path : str, optional
            The data location. If using :class:`ImageFolderDataset`,
            image folder`path/to/the/folder` should be provided.
            If using :class:`RecordDataset`, the `path/to/*.rec` should be provided.
        name : str, optional
            Which built-in dataset to use, will override all other options if specified.
            The options are: 'mnist', 'fashionmnist', 'cifar', 'cifar10', 'cifar100', 'imagenet'
        train : bool, optional, default = True
            Whether this dataset should be used for training or validation.
        input_size : int
            The input image size.
        crop_ratio : float
            Center crop ratio (for evaluation only).

        Returns
        -------
        Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`.
        To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
        """
        if name is None:
            if path is None:
                raise ValueError("Either `path` or `name` must be present in Dataset(). "
                                 "If `name` is provided, it will override the rest of the arguments.")
        return get_dataset(path=path, train=train, name=name,
                           input_size=input_size, crop_ratio=crop_ratio,
                           *args, **kwargs)

    @staticmethod
    def fit(dataset,
            net=Categorical('ResNet50_v1b', 'ResNet18_v1b'),
            optimizer=NAG(
                learning_rate=Real(1e-3, 1e-2, log=True),
                wd=Real(1e-4, 1e-3, log=True),
                multi_precision=False
            ),
            loss=SoftmaxCrossEntropyLoss(),
            split_ratio=0.8,
            batch_size=64,
            input_size=224,
            epochs=20,
            final_fit_epochs=None,
            ensemble=1,
            metric='accuracy',
            nthreads_per_trial=60,
            ngpus_per_trial=1,
            hybridize=True,
            scheduler_options=None,
            search_strategy='random',
            search_options=None,
            plot_results=False,
            verbose=False,
            time_limits=None,
            resume=False,
            output_directory='checkpoint/',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=None,
            auto_search=True,
            lr_config=Dict(
                lr_mode='cosine',
                lr_decay=0.1,
                lr_decay_period=0,
                lr_decay_epoch='40,80',
                warmup_lr=0.0,
                warmup_epochs=0
            ),
            tricks=Dict(
                last_gamma=False,
                use_pretrained=True,
                use_se=False,
                mixup=False,
                mixup_alpha=0.2,
                mixup_off_epoch=0,
                label_smoothing=False,
                no_wd=False,
                teacher_name=None,
                temperature=20.0,
                hard_weight=0.5,
                batch_norm=False,
                use_gn=False),
            **kwargs):
        # TODO: ensemble and hybridize are not in docstring
        """
        Fit image classification models to a given dataset.

        Parameters
        ----------
        dataset : str or :meth:`autogluon.task.ImageClassification.Dataset`
            Training dataset containing images and their associated class labels.
            Popular image datasets built into AutoGluon can be used by specifying their name as a string (options: ‘mnist’, ‘fashionmnist’, ‘cifar’, ‘cifar10’, ‘cifar100’, ‘imagenet’).
        input_size : int
            Size of images in the dataset (pixels).
        net : str or :class:`autogluon.space.Categorical`
            Which existing neural network models to consider as candidates.
        optimizer : str or :class:`autogluon.space.AutoGluonObject`
            Which optimizers to consider as candidates for learning the neural network weights.
        batch_size : int
            How many images to group in each mini-batch during gradient computations in training.
        epochs: int
            How many epochs to train the neural networks for at most.
        final_fit_epochs: int, default None
            Final fit epochs, the same number of epochs will be used as during the HPO if not specified.
        metric : str or callable object
            Evaluation metric by which predictions will be ulitmately evaluated on test data.
        loss : `mxnet.gluon.loss`
            Loss function used during training of the neural network weights.
        num_trials : int
            Maximal number of hyperparameter configurations to try out.
        split_ratio : float, default = 0.8
            Fraction of dataset to use for training (rest of data is held-out for tuning hyperparameters).
            The final returned model may be fit to all of the data (after hyperparameters have been selected).
        time_limits : int
            Approximately how long `fit()` should run for (wallclock time in seconds).
            `fit()` will stop training new models after this amount of time has elapsed (but models which have already started training will continue to completion).
        nthreads_per_trial : int
            How many CPUs to use in each trial (ie. single training run of a model).
        ngpus_per_trial : int
            How many GPUs to use in each trial (ie. single training run of a model).
        output_directory : str
            Checkpoints of the search state are written to
            os.path.join(output_directory, 'exp1.ag')
        scheduler_options : dict
            Extra arguments passed to __init__ of scheduler, to configure the
            orchestration of training jobs during hyperparameter-tuning.
        search_strategy : str
            Which hyperparameter search algorithm to use.
            Options include: 'random' (random search), 'skopt' (SKopt Bayesian optimization), 'grid' (grid search), 'hyperband' (Hyperband), 'rl' (reinforcement learner)
        search_options : dict
            Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization.
        resume : bool
            If True, the hyperparameter search is started from state loaded from
            os.path.join(output_directory, 'exp1.ag')
        dist_ip_addrs : list
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        verbose : bool
            Whether or not to print out intermediate information during training.
        plot_results : bool
            Whether or not to generate plots summarizing training process.
        visualizer : str
            Describes method to visualize training progress during `fit()`. Options: ['mxboard', 'tensorboard', 'none'].
        auto_search : bool
            If True, enables automatic suggestion of network types and hyper-parameter ranges adaptively based on provided dataset.

        Returns
        -------
            :class:`autogluon.task.image_classification.Classifier` object which can make predictions on new data and summarize what happened during `fit()`.

        Examples
        --------
        >>> from autogluon import ImageClassification as task
        >>> dataset = task.Dataset(train_path='data/train',
        >>>                        test_path='data/test')
        >>> classifier = task.fit(dataset,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limits=time_limits,
        >>>                       ngpus_per_trial=1,
        >>>                       num_trials = 4)
        >>> test_data = task.Dataset('~/data/test', train=False)
        >>> test_acc = classifier.evaluate(test_data)


        Bag of tricks are used on image classification dataset

        lr_config
        ----------
        lr-mode : type=str, default='step'.
            describes how learning rate should be adjusted over the course of training. Options include: 'cosine', 'poly'.
        lr-decay : type=float, default=0.1.
            decay rate of learning rate. default is 0.1.
        lr-decay-period : type=int, default=0.
            interval for periodic learning rate decays. default is 0 to disable.
        lr-decay-epoch : type=str, default='10,20,30'.
            epochs at which learning rate decays. epochs=40, default is 10, 20, 30.
        warmup-lr : type=float, default=0.0.
            starting warmup learning rate. default is 0.0.
        warmup-epochs : type=int, default=0.
            number of warmup epochs.

        tricks
        ----------
        last-gamma', default= True.
            whether to init gamma of the last BN layer in each bottleneck to 0.
        use-pretrained', default= True.
            enable using pretrained model from gluon.
        use_se', default= False.
            use SE layers or not in resnext. default is false.
        mixup', default= False.
            whether train the model with mix-up. default is false.
        mixup-alpha', type=float, default=0.2.
            beta distribution parameter for mixup sampling, default is 0.2.
        mixup-off-epoch', type=int, default=0.
            how many last epochs to train without mixup, default is 0.
        label-smoothing', default= True.
            use label smoothing or not in training. default is false.
        no-wd', default= True.
            whether to remove weight decay on bias, and beta/gamma for batchnorm layers.
        teacher', type=str, default=None.
            teacher model for distillation training
        temperature', type=float, default=20.
            temperature parameter for distillation teacher model
        hard-weight', type=float, default=0.5.
            weight for the loss of one-hot label for distillation training
        batch-norm', default= True.
            enable batch normalization or not in vgg. default is false.
        use-gn', default= False.
            whether to use group norm.
        """
        checkpoint = os.path.join(output_directory, 'exp1.ag')
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            net = auto_suggest_network(dataset, net)

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        final_fit_epochs = final_fit_epochs if final_fit_epochs else epochs
        train_image_classification.register_args(
            dataset=dataset,
            net=net,
            optimizer=optimizer,
            loss=loss,
            metric=metric,
            num_gpus=ngpus_per_trial,
            split_ratio=split_ratio,
            batch_size=batch_size,
            input_size=input_size,
            epochs=epochs,
            final_fit_epochs=final_fit_epochs,
            verbose=verbose,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            final_fit=False,
            tricks=tricks,
            lr_config=lr_config
        )

        # Backward compatibility:
        grace_period = kwargs.get('grace_period')
        if grace_period is not None:
            if scheduler_options is None:
                scheduler_options = {'grace_period': grace_period}
            else:
                assert 'grace_period' not in scheduler_options, \
                    "grace_period appears both in scheduler_options and as direct argument"
                logger.warning(
                    "grace_period is deprecated, use "
                    "scheduler_options={'grace_period': ...} instead")
                scheduler_options = copy.copy(scheduler_options)
                scheduler_options['grace_period'] = grace_period
        scheduler_options = compile_scheduler_options(
            scheduler_options=scheduler_options,
            search_strategy=search_strategy,
            search_options=search_options,
            nthreads_per_trial=nthreads_per_trial,
            ngpus_per_trial=ngpus_per_trial,
            checkpoint=checkpoint,
            num_trials=num_trials,
            time_out=time_limits,
            resume=resume,
            visualizer=visualizer,
            time_attr='epoch',
            reward_attr='classification_reward',
            dist_ip_addrs=dist_ip_addrs,
            epochs=epochs)
        results = BaseTask.run_fit(
            train_image_classification, search_strategy, scheduler_options,
            plot_results=plot_results)
        args = sample_config(train_image_classification.args, results['best_config'])

        kwargs = {'num_classes': results['num_classes'], 'ctx': mx.cpu(0)}
        model = get_network(args.net, **kwargs)
        multi_precision = optimizer.kwvars['multi_precision'] if 'multi_precision' in optimizer.kwvars else False
        update_params(model, results.pop('model_params'), multi_precision)

        if ensemble > 1:
            models = [model]
            scheduler = create_scheduler(
                train_image_classification, search_strategy, scheduler_options)
            for i in range(1, ensemble):
                resultsi = scheduler.run_with_config(results['best_config'])
                kwargs = {
                    'num_classes': resultsi['num_classes'], 'ctx': mx.cpu(0)}
                model = get_network(args.net, **kwargs)
                update_params(
                    model, resultsi.pop('model_params'), multi_precision)
                models.append(model)
            model = Ensemble(models)

        results.pop('args')
        args.pop('optimizer')
        args.pop('dataset')
        args.pop('loss')
        return Classifier(model, results, default_val_fn, checkpoint, args)
