""" Runs autogluon.tabular on multiple benchmark datasets.
    Run this benchmark with fast_benchmark=False to assess whether major chances make autogluon better or worse overall.
    Lower performance-values = better, normalized to [0,1] for each dataset to enable cross-dataset comparisons.
    Classification performance = error-rate, Regression performance = 1 - R^2

    # TODO: assess that Autogluon correctly inferred the type of each feature (continuous vs categorical vs text)

    # TODO: may want to take allowed run-time of AutoGluon into account? Eg. can produce performance vs training time curves for each dataset.

    # TODO: We'd like to add extra benchmark datasets with the following properties:
    - parquet file format
    - poker hand data: https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    - test dataset with just one data point
    - test dataset where order of columns different than in training data (same column names)
    - extreme-multiclass classification (500+ classes)
    - high-dimensional features + low-sample size
    - high levels of missingness in test data only, no missingness in train data
    - classification w severe class imbalance
    - regression with severely skewed Y-values (eg. predicting count data)
    - text features in dataset
"""
import warnings, shutil, os
import numpy as np
import mxnet as mx
from random import seed

from networkx.exception import NetworkXError
import pandas as pd
import pytest

import autogluon.core as ag
from autogluon.tabular import TabularPrediction as task
from autogluon.tabular.utils import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.task.tabular_prediction.predictor_v2 import TabularPredictorV2


def test_tabular():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 0 # random seed
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
        time_limits = 60

    fit_args = {
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limits is not None:
        fit_args['time_limits'] = time_limits
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)
    run_tabular_benchmark_toy(fit_args=fit_args)


def test_advanced_functionality():
    fast_benchmark = True
    dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
                      'name': 'AdultIncomeBinaryClassification',
                      'problem_type': BINARY}
    label = 'class'
    directory_prefix = './datasets/'
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
    if fast_benchmark:  # subsample for fast_benchmark
        subsample_size = 100
        train_data = train_data.head(subsample_size)
        test_data = test_data.head(subsample_size)
    print(f"Evaluating Advanced Functionality on Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + 'advanced/' + dataset['name'] + "/"
    savedir = directory + 'AutogluonOutput/'
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    predictor = TabularPredictorV2(label=label, path=savedir).fit(train_data)
    leaderboard = predictor.leaderboard(dataset=test_data)
    leaderboard_extra = predictor.leaderboard(dataset=test_data, extra_info=True)
    assert set(predictor.get_model_names()) == set(leaderboard['model'])
    assert set(predictor.get_model_names()) == set(leaderboard_extra['model'])
    assert set(leaderboard_extra.columns).issuperset(set(leaderboard.columns))
    assert len(leaderboard) == len(leaderboard_extra)
    num_models = len(predictor.get_model_names())
    feature_importances = predictor.feature_importance(dataset=test_data)
    original_features = set(train_data.columns)
    original_features.remove(label)
    assert set(feature_importances.index) == original_features
    assert set(feature_importances.columns) == {'importance', 'stddev', 'p_value', 'n', 'p99_high', 'p99_low'}
    predictor.transform_features()
    predictor.transform_features(dataset=test_data)
    predictor.info()

    assert predictor.get_model_names_persisted() == []  # Assert that no models were persisted during training
    assert predictor.unpersist_models() == []  # Assert that no models were unpersisted

    persisted_models = predictor.persist_models(models='all', max_memory=None)
    assert set(predictor.get_model_names_persisted()) == set(persisted_models)  # Ensure all models are persisted
    assert predictor.persist_models(models='all', max_memory=None) == []  # Ensure that no additional models are persisted on repeated calls
    unpersised_models = predictor.unpersist_models()
    assert set(unpersised_models) == set(persisted_models)
    assert predictor.get_model_names_persisted() == []  # Assert that all models were unpersisted

    # Raise exception
    with pytest.raises(NetworkXError):
        predictor.persist_models(models=['UNKNOWN_MODEL_1', 'UNKNOWN_MODEL_2'])

    assert predictor.get_model_names_persisted() == []

    assert predictor.unpersist_models(models=['UNKNOWN_MODEL_1', 'UNKNOWN_MODEL_2']) == []

    predictor.persist_models(models='all', max_memory=None)
    predictor.save()  # Save predictor while models are persisted: Intended functionality is that they won't be persisted when loaded.
    predictor_loaded = TabularPredictorV2.load(predictor.output_directory)  # Assert that predictor loading works
    leaderboard_loaded = predictor_loaded.leaderboard(dataset=test_data)
    assert len(leaderboard) == len(leaderboard_loaded)
    assert predictor_loaded.get_model_names_persisted() == []  # Assert that models were not still persisted after loading predictor

    assert(predictor.get_model_full_dict() == dict())
    predictor.refit_full()
    assert(len(predictor.get_model_full_dict()) == num_models)
    assert(len(predictor.get_model_names()) == num_models * 2)
    for model in predictor.get_model_names():
        predictor.predict(dataset=test_data, model=model)
    predictor.refit_full()  # Confirm that refit_models aren't further refit.
    assert(len(predictor.get_model_full_dict()) == num_models)
    assert(len(predictor.get_model_names()) == num_models * 2)
    predictor.delete_models(models_to_keep=[])  # Test that dry-run doesn't delete models
    assert(len(predictor.get_model_names()) == num_models * 2)
    predictor.predict(dataset=test_data)
    predictor.delete_models(models_to_keep=[], dry_run=False)  # Test that dry-run deletes models
    assert len(predictor.get_model_names()) == 0
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(extra_info=True)) == 0
    try:
        predictor.predict(dataset=test_data)
    except:
        pass
    else:
        raise AssertionError('predictor.predict should raise exception after all models are deleted')
    print('Tabular Advanced Functionality Test Succeeded.')


def load_data(directory_prefix, train_file, test_file, name, url=None):
    if not os.path.exists(directory_prefix):
        os.mkdir(directory_prefix)
    directory = directory_prefix + name + "/"
    train_file_path = directory + train_file
    test_file_path = directory + test_file
    if (not os.path.exists(train_file_path)) or (not os.path.exists(test_file_path)):
        # fetch files from s3:
        print("%s data not found locally, so fetching from %s" % (name, url))
        zip_name = ag.download(url, directory_prefix)
        ag.unzip(zip_name, directory_prefix)
        os.remove(zip_name)

    train_data = task.Dataset(file_path=train_file_path)
    test_data = task.Dataset(file_path=test_file_path)
    return train_data, test_data


def run_tabular_benchmark_toy(fit_args):
    dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyClassification.zip',
                          'name': 'toyClassification',
                          'problem_type': MULTICLASS,
                          'label_column': 'y',
                          'performance_val': 0.436}
    # 2-D toy noisy, imbalanced 4-class classification task with: feature missingness, out-of-vocabulary feature categories in test data, out-of-vocabulary labels in test data, training column missing from test data, extra distraction columns in test data
    # toyclassif_dataset should produce 1 warning and 1 error during inference:
    # Warning: Ignoring 181 (out of 1000) training examples for which the label value in column 'y' is missing
    # ValueError: Required columns are missing from the provided dataset. Missing columns: ['lostcolumn']

    # Additional warning that would have occurred if ValueError was not triggered:
    # UserWarning: These columns from this dataset were not present in the training dataset (AutoGluon will ignore them):  ['distractioncolumn1', 'distractioncolumn2']

    directory_prefix = './datasets/'
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + dataset['name'] + "/"
    savedir = directory + 'AutogluonOutput/'
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    predictor = TabularPredictorV2(label=dataset['label_column'], path=savedir).fit(train_data, **fit_args)
    print(predictor.feature_metadata)
    print(predictor.feature_metadata.type_map_raw)
    print(predictor.feature_metadata.type_group_map_special)
    try:
        predictor.predict(test_data)
    except KeyError:  # KeyError should be raised because test_data has missing column 'lostcolumn'
        pass
    else:
        raise AssertionError(f'{dataset["name"]} should raise an exception.')


def run_tabular_benchmarks(fast_benchmark, subsample_size, perf_threshold, seed_val, fit_args, dataset_indices=None, run_distill=False):
    print("Running fit with args:")
    print(fit_args)
    # Each train/test dataset must be located in single directory with the given names.
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    EPS = 1e-10

    # Information about each dataset in benchmark is stored in dict.
    # performance_val = expected performance on this dataset (lower = better),should update based on previously run benchmarks
    binary_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
                      'name': 'AdultIncomeBinaryClassification',
                      'problem_type': BINARY,
                      'label_column': 'class',
                      'performance_val': 0.129} # Mixed types of features.

    multi_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
                      'name': 'CoverTypeMulticlassClassification',
                      'problem_type': MULTICLASS,
                      'label_column': 'Cover_Type',
                      'performance_val': 0.032} # big dataset with 7 classes, all features are numeric. Runs SLOW.

    regression_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
                       'name': 'AmesHousingPriceRegression',
                      'problem_type': REGRESSION,
                      'label_column': 'SalePrice',
                      'performance_val': 0.076} # Regression with mixed feature-types, skewed Y-values.

    toyregres_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip',
                         'name': 'toyRegression',
                         'problem_type': REGRESSION,
                        'label_column': 'y',
                        'performance_val': 0.183}
    # 1-D toy deterministic regression task with: heavy label+feature missingness, extra distraction column in test data

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    datasets = [toyregres_dataset, binary_dataset, regression_dataset, multi_dataset]
    if dataset_indices is not None: # only run some datasets
        datasets = [datasets[i] for i in dataset_indices]

    # Aggregate performance summaries obtained in previous benchmark run:
    prev_perf_vals = [dataset['performance_val'] for dataset in datasets]
    previous_avg_performance = np.mean(prev_perf_vals)
    previous_median_performance = np.median(prev_perf_vals)
    previous_worst_performance = np.max(prev_perf_vals)

    # Run benchmark:
    performance_vals = [0.0] * len(datasets) # performance obtained in this run
    directory_prefix = './datasets/'
    with warnings.catch_warnings(record=True) as caught_warnings:
        for idx in range(len(datasets)):
            dataset = datasets[idx]
            train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
            if seed_val is not None:
                seed(seed_val)
                np.random.seed(seed_val)
                mx.random.seed(seed_val)
            print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset['name'], idx+1, len(datasets)))
            directory = directory_prefix + dataset['name'] + "/"
            savedir = directory + 'AutogluonOutput/'
            shutil.rmtree(savedir, ignore_errors=True) # Delete AutoGluon output directory to ensure previous runs' information has been removed.
            label_column = dataset['label_column']
            y_test = test_data[label_column]
            test_data = test_data.drop(labels=[label_column], axis=1)
            if fast_benchmark:
                if subsample_size is None:
                    raise ValueError("fast_benchmark specified without subsample_size")
                if subsample_size < len(train_data):
                    # .sample instead of .head to increase diversity and test cases where data index is not monotonically increasing.
                    train_data = train_data.sample(n=subsample_size, random_state=seed_val)  # subsample for fast_benchmark
            predictor = TabularPredictorV2(label=label_column, path=savedir).fit(train_data, **fit_args)
            results = predictor.fit_summary(verbosity=4)
            if predictor.problem_type != dataset['problem_type']:
                warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
            predictor = TabularPredictorV2.load(savedir)  # Test loading previously-trained predictor from file
            y_pred = predictor.predict(test_data)
            perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            if dataset['problem_type'] != REGRESSION:
                perf = 1.0 - perf_dict['accuracy_score'] # convert accuracy to error-rate
            else:
                perf = 1.0 - perf_dict['r2_score'] # unexplained variance score.
            performance_vals[idx] = perf
            print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset['name'], performance_vals[idx], dataset['performance_val']))
            if (not fast_benchmark) and (performance_vals[idx] > dataset['performance_val'] * perf_threshold):
                warnings.warn("Performance on dataset %s is %s times worse than previous performance." %
                              (dataset['name'], performance_vals[idx]/(EPS+dataset['performance_val'])))
            if predictor._trainer.bagged_mode:
                # TODO: Test index alignment with original training data (first handle duplicated rows / dropped rows edge cases)
                y_train_pred_oof = predictor.get_oof_pred()
                y_train_pred_proba_oof = predictor.get_oof_pred_proba()
                y_train_pred_oof_transformed = predictor.get_oof_pred(transformed=True)
                y_train_pred_proba_oof_transformed = predictor.get_oof_pred_proba(transformed=True)

                # Assert expected type output
                assert isinstance(y_train_pred_oof, pd.Series)
                assert isinstance(y_train_pred_oof_transformed, pd.Series)
                if predictor.problem_type == MULTICLASS:
                    assert isinstance(y_train_pred_proba_oof, pd.DataFrame)
                    assert isinstance(y_train_pred_proba_oof_transformed, pd.DataFrame)
                else:
                    if predictor.problem_type == BINARY:
                        assert isinstance(predictor.get_oof_pred_proba(as_multiclass=True), pd.DataFrame)
                    assert isinstance(y_train_pred_proba_oof, pd.Series)
                    assert isinstance(y_train_pred_proba_oof_transformed, pd.Series)

                assert y_train_pred_oof_transformed.equals(predictor.transform_labels(y_train_pred_oof, proba=False))

                # Test that the transform_labels method is capable of reproducing the same output when converting back and forth, and test that oof 'transform' parameter works properly.
                y_train_pred_proba_oof_inverse = predictor.transform_labels(y_train_pred_proba_oof, proba=True)
                y_train_pred_proba_oof_inverse_inverse = predictor.transform_labels(y_train_pred_proba_oof_inverse, proba=True, inverse=True)
                y_train_pred_oof_inverse = predictor.transform_labels(y_train_pred_oof)
                y_train_pred_oof_inverse_inverse = predictor.transform_labels(y_train_pred_oof_inverse, inverse=True)

                if isinstance(y_train_pred_proba_oof_transformed, pd.DataFrame):
                    pd.testing.assert_frame_equal(y_train_pred_proba_oof_transformed, y_train_pred_proba_oof_inverse)
                    pd.testing.assert_frame_equal(y_train_pred_proba_oof, y_train_pred_proba_oof_inverse_inverse)
                else:
                    pd.testing.assert_series_equal(y_train_pred_proba_oof_transformed, y_train_pred_proba_oof_inverse)
                    pd.testing.assert_series_equal(y_train_pred_proba_oof, y_train_pred_proba_oof_inverse_inverse)
                pd.testing.assert_series_equal(y_train_pred_oof_transformed, y_train_pred_oof_inverse)
                pd.testing.assert_series_equal(y_train_pred_oof, y_train_pred_oof_inverse_inverse)

                # Test that index of both the internal training data and the oof outputs are consistent in their index values.
                X_internal, y_internal = predictor.load_data_internal()
                y_internal_index = list(y_internal.index)
                assert list(X_internal.index) == y_internal_index
                assert list(y_train_pred_oof.index) == y_internal_index
                assert list(y_train_pred_proba_oof.index) == y_internal_index
                assert list(y_train_pred_oof_transformed.index) == y_internal_index
                assert list(y_train_pred_proba_oof_transformed.index) == y_internal_index
            else:
                # Raise exception
                with pytest.raises(AssertionError):
                    predictor.get_oof_pred()
                with pytest.raises(AssertionError):
                    predictor.get_oof_pred_proba()
            if run_distill:
                predictor.distill(time_limits=60, augment_args={'size_factor':0.5})

    # Summarize:
    avg_perf = np.mean(performance_vals)
    median_perf = np.median(performance_vals)
    worst_perf = np.max(performance_vals)
    for idx in range(len(datasets)):
        print("Performance on dataset %s: %s   (previous perf=%s)" % (datasets[idx]['name'], performance_vals[idx], datasets[idx]['performance_val']))

    print("Average performance: %s" % avg_perf)
    print("Median performance: %s" % median_perf)
    print("Worst performance: %s" % worst_perf)

    if not fast_benchmark:
        if avg_perf > previous_avg_performance * perf_threshold:
            warnings.warn("Average Performance is %s times worse than previously." % (avg_perf/(EPS+previous_avg_performance)))
        if median_perf > previous_median_performance * perf_threshold:
            warnings.warn("Median Performance is %s times worse than previously." % (median_perf/(EPS+previous_median_performance)))
        if worst_perf > previous_worst_performance * perf_threshold:
            warnings.warn("Worst Performance is %s times worse than previously." % (worst_perf/(EPS+previous_worst_performance)))

    print("Ran fit with args:")
    print(fit_args)
    # List all warnings again to make sure they are seen:
    print("\n\n WARNINGS:")
    for w in caught_warnings:
        warnings.warn(w.message)


@pytest.mark.slow
def test_tabularHPObagstack():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 10000 # random seed
    subsample_size = None
    hyperparameter_tune = True
    stack_ensemble_levels = 2
    num_bagging_folds = 2
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
        nn_options = {'num_epochs': 2, 'learning_rate': ag.Real(0.001,0.01), 'lr_scheduler': ag.Categorical(None, 'cosine','step')}
        gbm_options = {'num_boost_round': 20, 'learning_rate': ag.Real(0.01,0.1)}
        hyperparameters = {'GBM': gbm_options, 'NN': nn_options}
        time_limits = 150
        num_trials = 3

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
    if num_trials is not None:
        fit_args['num_trials'] = num_trials
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


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


@pytest.mark.skip(reason="Ignored for now, since stacking is disabled without bagging.")
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


@pytest.mark.skip(reason="Ignored for now, since stacking is disabled without bagging.")
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


@pytest.mark.slow
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
                           seed_val=seed_val, fit_args=fit_args, run_distill=True)

