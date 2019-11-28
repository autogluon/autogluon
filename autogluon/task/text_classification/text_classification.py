import logging
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp

from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .classification_models import get_network
from .dataset import TextClassificationDataset
from .pipeline import *
from .metrics import get_metric_instance
from .optimizers import *
from .predictor import TextClassificationPredictor

__all__ = ['TextClassification']

logger = logging.getLogger(__name__)

class TextClassification(BaseTask):
    """AutoGluon TextClassification Task
    """
    Dataset = TextClassificationDataset
    @staticmethod
    def fit(dataset='SST',
            net=Categorical('bert_12_768_12'),
            pretrained_dataset=Categorical('book_corpus_wiki_en_uncased'),
            lr=Real(2e-05, 2e-04, log=True),
            warmup_ratio=0.01,
            lr_scheduler='cosine',
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            log_interval=100,
            seed=0,
            batch_size=32,
            dev_batch_size=32,
            max_len=128,
            dtype='float32',
            epochs=3,
            epsilon=1e-6,
            accumulate=1,
            metric='accuracy',
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
            **kwargs):

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
            >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> predictor = task.fit(dataset,
            >>>                      nets=ag.Choice['resnet18_v1', 'resnet34_v1'],
            >>>                      time_limits=time_limits,
            >>>                      num_gpus=1,
            >>>                      num_trials = 4)
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
            loss=loss,
            metric=metric,
            num_gpus=ngpus_per_trial,
            batch_size=batch_size,
            dev_batch_size=dev_batch_size,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
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
        results = BaseTask.run_fit(train_text_classification, search_strategy,
                                   scheduler_options)
        args = sample_config(train_text_classification.args, results['best_config'])
        model = get_network(args.net, results['num_classes'], mx.cpu(0))
        update_params(model, results.pop('model_params'))
        return TextClassificationPredictor(model, results, evaluate, checkpoint, args)
