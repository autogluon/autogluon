""" Runs autogluon.tabular on multiple benchmark datasets.
    Run this benchmark with fast_benchmark=False to assess whether major chances make autogluon better or worse overall.
    Lower performance-values = better, normalized to [0,1] for each dataset to enable cross-dataset comparisons.
    Classification performance = error-rate, Regression performance = 1 - R^2
"""
import warnings, shutil, os
import numpy as np
import mxnet as mx
from random import seed

import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from test_tabular import run_tabular_benchmarks


def test_tabularHPO():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 99 # random seed
    subsample_size = None
    hyperparameter_tune = True
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limits = None
    num_trials = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 2}
        gbm_options = {'num_boost_round': 20}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options}
        time_limits = 60
        num_trials = 5

    fit_args = {
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
    if num_trials is not None:
        fit_args['num_trials'] = num_trials
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


if __name__ == '__main__':
    test_tabularHPO()
