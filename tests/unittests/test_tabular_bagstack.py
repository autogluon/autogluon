""" Runs autogluon.tabular on multiple benchmark datasets. 
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


def test_tabular_bag():
    ############ Benchmark options you can set: ########################
    num_bagging_folds = 3
    stack_ensemble_levels = 0
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 123 # random seed
    subsample_size = None
    hyperparameter_tune = False
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limits = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 120
        nn_options = {'num_epochs': 1} 
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options}
        time_limits = 60

    fit_args = {
        'num_bagging_folds': num_bagging_folds,
        'stack_ensemble_levels': stack_ensemble_levels,
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
        fit_args['num_bagging_sets'] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


def test_tabular_stack1():
    ############ Benchmark options you can set: ########################
    stack_ensemble_levels = 1
    num_bagging_folds = 0
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 32 # random seed
    subsample_size = None
    hyperparameter_tune = False
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limits = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 3} 
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options}
        time_limits = 60

    fit_args = {
        'num_bagging_folds': num_bagging_folds,
        'stack_ensemble_levels': stack_ensemble_levels,
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val = seed_val, fit_args=fit_args)


def test_tabular_stack2():
    ############ Benchmark options you can set: ########################
    stack_ensemble_levels = 2
    num_bagging_folds = 0
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 66 # random seed
    subsample_size = None
    hyperparameter_tune = False
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limits = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 3} 
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options}
        time_limits = 60

    fit_args = {
        'num_bagging_folds': num_bagging_folds,
        'stack_ensemble_levels': stack_ensemble_levels,
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


def test_tabular_bagstack():
    ############ Benchmark options you can set: ########################
    stack_ensemble_levels = 2
    num_bagging_folds = 3
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 53 # random seed
    subsample_size = None
    hyperparameter_tune = False
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limits = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 105
        nn_options = {'num_epochs': 2} 
        gbm_options = {'num_boost_round': 40}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options, 'custom': ['GBM']}
        time_limits = 60

    fit_args = {
        'num_bagging_folds': num_bagging_folds,
        'stack_ensemble_levels': stack_ensemble_levels,
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
        fit_args['num_bagging_sets'] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


if __name__ == '__main__':
    test_tabular_bag()
    # test_tabular_stack1() # TODO: Ignored for now, since stacking is disabled without bagging.
    # test_tabular_stack2()
    test_tabular_bagstack()
