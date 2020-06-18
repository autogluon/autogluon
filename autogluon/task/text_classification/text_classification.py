import logging

import mxnet as mx
from ...utils.try_import import try_import_gluonnlp

import copy

from ...core import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask, compile_scheduler_options
from ...utils import update_params

from .network import get_network
from .dataset import get_dataset
from .pipeline import *
from .predictor import TextClassificationPredictor

__all__ = ['TextClassification']

logger = logging.getLogger(__name__)

class TextClassification(BaseTask):
    """AutoGluon Task for classifying text snippets based on their content
    """
    @staticmethod
    def Dataset(*args, **kwargs):
        """Dataset of text examples to make predictions for. 
           See :meth:`autogluon.task.TextClassification.get_dataset`
        """
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset='SST',
            net=Categorical('bert_12_768_12'),
            pretrained_dataset=Categorical('book_corpus_wiki_en_uncased',
                                           'openwebtext_book_corpus_wiki_en_uncased'),
            lr=Real(2e-05, 2e-04, log=True),
            warmup_ratio=0.01,
            lr_scheduler='cosine',
            log_interval=100,
            seed=0,
            batch_size=32,
            dev_batch_size=32,
            max_len=128,
            dtype='float32',
            epochs=3,
            epsilon=1e-6,
            accumulate=1,
            early_stop=False,
            nthreads_per_trial=4,
            ngpus_per_trial=1,
            hybridize=True,
            scheduler_options=None,
            search_strategy='random',
            search_options=None,
            num_trials=None,
            time_limits=None,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            dist_ip_addrs=None,
            auto_search=True,
            verbose=False,
            **kwargs):

        """Fit neural networks on text dataset.

        Parameters
        ----------
        dataset : str or :class:`autogluon.task.TextClassification.Dataset`
            The Training dataset. You can specify a string to use a popular built-in text dataset.
        net : str or :class:`autogluon.space.Categorical`
            Which existing neural network models to consider as candidates.
        pretrained_dataset : str, :class:`autogluon.space.Categorical`
            Which existing datasets to consider as candidates for transfer learning from.
        lr : float or :class:`autogluon.space`
            The learning rate to use in each update of the neural network weights during training.
        warmup_ratio : float
            Ratio of overall training period considered as "warm up".
        lr_scheduler : str
            Describes how learning rate should be adjusted over the course of training. Options include: 'cosine', 'poly'.
        log_interval : int
            Log results every so many epochs during training.
        seed : int
            Random seed to set for reproducibility.
        batch_size : int
            How many examples to group in each mini-batch during gradient computations in training.
        dev_batch_size : int
            How many examples to group in each mini-batch during performance evalatuion over validation dataset.
        max_len : int
            Maximum number of words in a single training example (i.e. one text snippet).
        dtype : str
            Dtype used to represent data fed to neural networks.
        epochs: int
            How many epochs to train the neural networks for at most.
        epsilon : float
            Small number.
        accumulate : int
            How often to accumulate losses.
        early_stop : bool
            Whether to utilize early stopping during training to avoid overfitting.
        num_trials : int
            Maximal number of hyperparameter configurations to try out.
        time_limits : int
            Approximately how long should `fit()` should run for (wallclock time in seconds).
            `fit()` will stop training new models after this amount of time has elapsed (but models which have already started training will continue to completion).
        nthreads_per_trial : int
            How many CPUs to use in each trial (ie. single training run of a model).
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
        checkpoint : str or None
            State of hyperparameter search is stored to this local file
        resume : bool
            If True, the hyperparameter search is started from state loaded
            from checkpoint
        visualizer : str
            Describes method to visualize training progress during `fit()`. Options: ['mxboard', 'tensorboard', 'none']. 
        dist_ip_addrs : list
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        auto_search : bool
            If True, enables automatic suggestion of network types and hyper-parameter ranges adaptively based on provided dataset.
        
         Returns
        -------
        :class:`autogluon.task.text_classification.TextClassificationPredictor` object which can make predictions on new data and summarize what happened during `fit()`.
        
        Examples
        --------
        >>> from autogluon import TextClassification as task
        >>> dataset = task.Dataset(name='ToySST')
        >>> predictor = task.fit(dataset)
        """
        assert search_strategy not in {'bayesopt', 'bayesopt_hyperband'}, \
            "search_strategy == 'bayesopt' or 'bayesopt_hyperband' not yet supported"

        logger.warning('`TextClassification` is in preview mode.'
                       'Please feel free to request new features in issues '
                       'if it is not covered in the current implementation. '
                       'If your dataset is in tabular format, you could also try out our `TabularPrediction` module.')

        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        # If only time_limits is given, the scheduler starts trials until the
        # time limit is reached
        if num_trials is None and time_limits is None:
            num_trials = 2

        train_text_classification.register_args(
            dataset=dataset,
            pretrained_dataset=pretrained_dataset,
            net=net,
            lr=lr,
            warmup_ratio=warmup_ratio,
            early_stop=early_stop,
            dtype=dtype,
            max_len=max_len,
            log_interval=log_interval,
            epsilon=epsilon,
            accumulate=accumulate,
            seed=seed,
            lr_scheduler=lr_scheduler,
            num_gpus=ngpus_per_trial,
            batch_size=batch_size,
            dev_batch_size=dev_batch_size,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            verbose=verbose,
            final_fit=False,
            **kwargs)

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
            resume=resume,
            visualizer=visualizer,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs=dist_ip_addrs,
            epochs=epochs)
        results = BaseTask.run_fit(
            train_text_classification, search_strategy, scheduler_options)
        args = sample_config(train_text_classification.args, results['best_config'])
        get_model_params = results.pop('get_model_args')
        get_model_params['ctx'] = mx.cpu(0)
        nlp = try_import_gluonnlp()
        bert, _ = nlp.model.get_model(**get_model_params)
        model = get_network(bert, results.pop('class_labels'), 'roberta' in args.net)
        update_params(model, results.pop('model_params'))
        transform = results.pop('transform')
        test_transform = results.pop('test_transform')
        return TextClassificationPredictor(model, transform, test_transform, results, checkpoint, args)
