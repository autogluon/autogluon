import numpy as np
import ConfigSpace as CS
import argparse

import autogluon as ag

from ..task.image_classification import pipeline
from ..task.image_classification.model_zoo import get_model
from ..network import Nets
from ..optim import Optimizers, get_optim
from ..loss import Losses
from ..metric import Metrics
from ..searcher import *

__all__ = ['fit']


# TODO (cgraywang): put into class that can be inherited and add readme
def fit(data,
        nets=Nets([]),
        optimizers=Optimizers([]),
        metrics=Metrics([]),
        losses=Losses([]),
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='./outputdir/',
        visualizer='tensorboard',
        stop_criterion={'time_limits': 1 * 60 * 60,
                        'max_metric': 0.80,
                        'max_trial_count': 10},
        resources_per_trial={'max_num_gpus': 1,
                             'max_num_cpus': 4,
                             'max_training_epochs': 1},
        backend='autogluon',
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
        e.g. ``{"cpu": 64, "gpu": 8}``. Note that GPUs will not be
        assigned unless you specify them here. Defaults to 1 CPU and 0
        GPUs in ``Trainable.default_resource_request()``.
    savedir (str): Local dir to save training results to.
    checkpoint_freq (int): How many training iterations between
        checkpoints. A value of 0 (default) disables checkpointing.
    checkpoint_at_end (bool): Whether to checkpoint at the end of the
        experiment regardless of the checkpoint_freq. Default is False.
    searcher (SearchAlgorithm): Search Algorithm. Defaults to
        BasicVariantGenerator.
    trial_scheduler (TrialScheduler): Scheduler for executing
        the experiment. Choose among FIFO (default), MedianStopping,
        AsyncHyperBand, and HyperBand.
    resume (bool|"prompt"): If checkpoint exists, the experiment will
        resume from there. If resume is "prompt", Tune will prompt if
        checkpoint detected.
    **kwargs: Used for backwards compatibility.

    Returns
    ----------
    List of Trial objects.
    """

    trials = None
    best_result = None
    cs = CS.ConfigurationSpace()
    return trials, best_result, cs
