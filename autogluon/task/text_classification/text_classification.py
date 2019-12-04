import logging
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp

from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .network import get_network
from .dataset import get_dataset
from .pipeline import *
from .predictor import TextClassificationPredictor

__all__ = ['TextClassification']

logger = logging.getLogger(__name__)

class TextClassification(BaseTask):
    """AutoGluon TextClassification Task
    """
    @staticmethod
    def Dataset(*args, **kwargs):
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
            search_strategy='random',
            search_options={},
            time_limits=None,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=[],
            grace_period=None,
            auto_search=True,
            verbose=False,
            **kwargs):

        """
        Fit networks on dataset

        Parameters
        ----------
        dataset (str or autogluon.task.TextClassification.Dataset): Training dataset.
        net (str): Network candidates.
        num_trials (int): number of trials in the experiment.
        time_limits (int): training time limits in seconds.
        resources_per_trial (dict): Machine resources to allocate per trial.
        savedir (str): Local dir to save training results to.
        search_strategy (str): Search Algorithms ('random', 'bayesopt' and 'hyperband')
        resume (bool): If checkpoint exists, the experiment will resume from there.


        Examples
        --------
        >>> dataset = task.Dataset(name='ToySST')
        >>> predictor = task.fit(dataset)
        """
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

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

        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'accuracy',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'searcher': 'random',
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})
        results = BaseTask.run_fit(train_text_classification, search_strategy,
                                   scheduler_options)
        args = sample_config(train_text_classification.args, results['best_config'])
        get_model_params = results.pop('get_model_args')
        get_model_params['ctx'] = mx.cpu(0)
        bert, _ = nlp.model.get_model(**get_model_params)
        model = get_network(bert, results.pop('class_labels'), 'roberta' in args.net)
        update_params(model, results.pop('model_params'))
        transform = results.pop('transform')
        test_transform = results.pop('test_transform')
        return TextClassificationPredictor(model, transform, test_transform, results, checkpoint, args)
