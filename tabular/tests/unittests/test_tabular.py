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
import os
import shutil
import warnings
from random import seed

import numpy as np
import pandas as pd
import pytest

import autogluon.core as ag
from autogluon.core.utils import download, unzip
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, PROBLEM_TYPES_CLASSIFICATION
from autogluon.tabular import TabularDataset, TabularPredictor
from networkx.exception import NetworkXError

PARALLEL_LOCAL_BAGGING = 'parallel_local'
SEQUENTIAL_LOCAL_BAGGING = 'sequential_local'

def test_tabular():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 0 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        time_limit = 60

    fit_args = {'verbosity': verbosity}
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
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
    predictor = TabularPredictor(label=label, path=savedir).fit(train_data)
    leaderboard = predictor.leaderboard(data=test_data)
    extra_metrics = ['accuracy', 'roc_auc', 'log_loss']
    leaderboard_extra = predictor.leaderboard(data=test_data, extra_info=True, extra_metrics=extra_metrics)
    assert set(predictor.get_model_names()) == set(leaderboard['model'])
    assert set(predictor.get_model_names()) == set(leaderboard_extra['model'])
    assert set(leaderboard_extra.columns).issuperset(set(leaderboard.columns))
    assert len(leaderboard) == len(leaderboard_extra)
    assert set(leaderboard_extra.columns).issuperset(set(extra_metrics))  # Assert that extra_metrics are present in output
    num_models = len(predictor.get_model_names())
    feature_importances = predictor.feature_importance(data=test_data)
    original_features = set(train_data.columns)
    original_features.remove(label)
    assert set(feature_importances.index) == original_features
    assert set(feature_importances.columns) == {'importance', 'stddev', 'p_value', 'n', 'p99_high', 'p99_low'}
    predictor.transform_features()
    test_data_transformed = predictor.transform_features(data=test_data)
    predictor.info()

    # Assert that transform_features=False works correctly
    y_pred = predictor.predict(test_data)
    y_pred_from_transform = predictor.predict(test_data_transformed, transform_features=False)
    assert y_pred.equals(y_pred_from_transform)
    y_pred_proba = predictor.predict_proba(test_data)
    y_pred_proba_from_transform = predictor.predict_proba(test_data_transformed, transform_features=False)
    assert y_pred_proba.equals(y_pred_proba_from_transform)

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
    predictor_loaded = TabularPredictor.load(predictor.path)  # Assert that predictor loading works
    leaderboard_loaded = predictor_loaded.leaderboard(data=test_data)
    assert len(leaderboard) == len(leaderboard_loaded)
    assert predictor_loaded.get_model_names_persisted() == []  # Assert that models were not still persisted after loading predictor

    assert(predictor.get_model_full_dict() == dict())
    predictor.refit_full()
    assert(len(predictor.get_model_full_dict()) == num_models)
    assert(len(predictor.get_model_names()) == num_models * 2)
    for model in predictor.get_model_names():
        predictor.predict(data=test_data, model=model)
    predictor.refit_full()  # Confirm that refit_models aren't further refit.
    assert(len(predictor.get_model_full_dict()) == num_models)
    assert(len(predictor.get_model_names()) == num_models * 2)
    predictor.delete_models(models_to_keep=[])  # Test that dry-run doesn't delete models
    assert(len(predictor.get_model_names()) == num_models * 2)
    predictor.predict(data=test_data)
    predictor.delete_models(models_to_keep=[], dry_run=False)  # Test that dry-run deletes models
    assert len(predictor.get_model_names()) == 0
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(extra_info=True)) == 0
    try:
        predictor.predict(data=test_data)
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
        zip_name = download(url, directory_prefix)
        unzip(zip_name, directory_prefix)
        os.remove(zip_name)

    train_data = TabularDataset(train_file_path)
    test_data = TabularDataset(test_file_path)
    return train_data, test_data


def run_tabular_benchmark_toy(fit_args):
    dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyClassification.zip',
               'name': 'toyClassification',
               'problem_type': MULTICLASS,
               'label': 'y',
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
    predictor = TabularPredictor(label=dataset['label'], path=savedir).fit(train_data, **fit_args)
    print(predictor.feature_metadata)
    print(predictor.feature_metadata.type_map_raw)
    print(predictor.feature_metadata.type_group_map_special)
    try:
        predictor.predict(test_data)
    except KeyError:  # KeyError should be raised because test_data has missing column 'lostcolumn'
        pass
    else:
        raise AssertionError(f'{dataset["name"]} should raise an exception.')


def get_benchmark_sets():
    # Information about each dataset in benchmark is stored in dict.
    # performance_val = expected performance on this dataset (lower = better),should update based on previously run benchmarks
    binary_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
                      'name': 'AdultIncomeBinaryClassification',
                      'problem_type': BINARY,
                      'label': 'class',
                      'performance_val': 0.129}  # Mixed types of features.

    multi_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
                     'name': 'CoverTypeMulticlassClassification',
                     'problem_type': MULTICLASS,
                     'label': 'Cover_Type',
                     'performance_val': 0.032}  # big dataset with 7 classes, all features are numeric. Runs SLOW.

    regression_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
                          'name': 'AmesHousingPriceRegression',
                          'problem_type': REGRESSION,
                          'label': 'SalePrice',
                          'performance_val': 0.076}  # Regression with mixed feature-types, skewed Y-values.

    toyregres_dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip',
                         'name': 'toyRegression',
                         'problem_type': REGRESSION,
                         'label': 'y',
                         'performance_val': 0.183}
    # 1-D toy deterministic regression task with: heavy label+feature missingness, extra distraction column in test data

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    return [toyregres_dataset, binary_dataset, regression_dataset, multi_dataset]


def run_tabular_benchmarks(fast_benchmark, subsample_size, perf_threshold, seed_val, fit_args, dataset_indices=None, run_distill=False, crash_in_oof=False):
    print("Running fit with args:")
    print(fit_args)
    # Each train/test dataset must be located in single directory with the given names.
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    EPS = 1e-10

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    datasets = get_benchmark_sets()
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
            print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset['name'], idx+1, len(datasets)))
            directory = directory_prefix + dataset['name'] + "/"
            savedir = directory + 'AutogluonOutput/'
            shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
            label = dataset['label']
            y_test = test_data[label]
            test_data = test_data.drop(labels=[label], axis=1)
            if fast_benchmark:
                if subsample_size is None:
                    raise ValueError("fast_benchmark specified without subsample_size")
                if subsample_size < len(train_data):
                    # .sample instead of .head to increase diversity and test cases where data index is not monotonically increasing.
                    train_data = train_data.sample(n=subsample_size, random_state=seed_val)  # subsample for fast_benchmark
            predictor = TabularPredictor(label=label, path=savedir).fit(train_data, **fit_args)
            results = predictor.fit_summary(verbosity=4)
            if predictor.problem_type != dataset['problem_type']:
                warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
            predictor = TabularPredictor.load(savedir)  # Test loading previously-trained predictor from file
            y_pred_empty = predictor.predict(test_data[0:0])
            assert len(y_pred_empty) == 0
            y_pred = predictor.predict(test_data)
            perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            if dataset['problem_type'] != REGRESSION:
                perf = 1.0 - perf_dict['accuracy']  # convert accuracy to error-rate
            else:
                perf = 1.0 - perf_dict['r2']  # unexplained variance score.
            performance_vals[idx] = perf
            print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset['name'], performance_vals[idx], dataset['performance_val']))
            if (not fast_benchmark) and (performance_vals[idx] > dataset['performance_val'] * perf_threshold):
                warnings.warn("Performance on dataset %s is %s times worse than previous performance." %
                              (dataset['name'], performance_vals[idx]/(EPS+dataset['performance_val'])))
            if predictor._trainer.bagged_mode and not crash_in_oof:
                # TODO: Test index alignment with original training data (first handle duplicated rows / dropped rows edge cases)
                y_pred_oof = predictor.get_oof_pred()
                y_pred_proba_oof = predictor.get_oof_pred_proba(as_multiclass=False)
                y_pred_oof_transformed = predictor.get_oof_pred(transformed=True)
                y_pred_proba_oof_transformed = predictor.get_oof_pred_proba(as_multiclass=False, transformed=True)

                # Assert expected type output
                assert isinstance(y_pred_oof, pd.Series)
                assert isinstance(y_pred_oof_transformed, pd.Series)
                if predictor.problem_type == MULTICLASS:
                    assert isinstance(y_pred_proba_oof, pd.DataFrame)
                    assert isinstance(y_pred_proba_oof_transformed, pd.DataFrame)
                else:
                    if predictor.problem_type == BINARY:
                        assert isinstance(predictor.get_oof_pred_proba(), pd.DataFrame)
                    assert isinstance(y_pred_proba_oof, pd.Series)
                    assert isinstance(y_pred_proba_oof_transformed, pd.Series)

                assert y_pred_oof_transformed.equals(predictor.transform_labels(y_pred_oof, proba=False))

                # Test that the transform_labels method is capable of reproducing the same output when converting back and forth, and test that oof 'transform' parameter works properly.
                y_pred_proba_oof_inverse = predictor.transform_labels(y_pred_proba_oof, proba=True)
                y_pred_proba_oof_inverse_inverse = predictor.transform_labels(y_pred_proba_oof_inverse, proba=True, inverse=True)
                y_pred_oof_inverse = predictor.transform_labels(y_pred_oof)
                y_pred_oof_inverse_inverse = predictor.transform_labels(y_pred_oof_inverse, inverse=True)

                if isinstance(y_pred_proba_oof_transformed, pd.DataFrame):
                    pd.testing.assert_frame_equal(y_pred_proba_oof_transformed, y_pred_proba_oof_inverse)
                    pd.testing.assert_frame_equal(y_pred_proba_oof, y_pred_proba_oof_inverse_inverse)
                else:
                    pd.testing.assert_series_equal(y_pred_proba_oof_transformed, y_pred_proba_oof_inverse)
                    pd.testing.assert_series_equal(y_pred_proba_oof, y_pred_proba_oof_inverse_inverse)
                pd.testing.assert_series_equal(y_pred_oof_transformed, y_pred_oof_inverse)
                pd.testing.assert_series_equal(y_pred_oof, y_pred_oof_inverse_inverse)

                # Test that index of both the internal training data and the oof outputs are consistent in their index values.
                X_internal, y_internal = predictor.load_data_internal()
                y_internal_index = list(y_internal.index)
                assert list(X_internal.index) == y_internal_index
                assert list(y_pred_oof.index) == y_internal_index
                assert list(y_pred_proba_oof.index) == y_internal_index
                assert list(y_pred_oof_transformed.index) == y_internal_index
                assert list(y_pred_proba_oof_transformed.index) == y_internal_index
            else:
                # Raise exception
                with pytest.raises(AssertionError):
                    predictor.get_oof_pred()
                with pytest.raises(AssertionError):
                    predictor.get_oof_pred_proba()
            if run_distill:
                predictor.distill(time_limit=60, augment_args={'size_factor':0.5})

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


def test_pseudolabeling():
    datasets = get_benchmark_sets()
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    directory_prefix = './datasets/'
    hyperparam_setting = {
        'GBM': {'num_boost_round': 10},
        'XGB': {'n_estimators': 10},
    }

    fit_args = dict(
        hyperparameters=hyperparam_setting,
        time_limit=20,
    )

    fit_args_best = dict(
        presets='best_quality',
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble=dict(fold_fitting_strategy='sequential_local'),
    )
    for idx in range(len(datasets)):
        dataset = datasets[idx]
        label = dataset['label']
        problem_type = dataset['problem_type']
        name = dataset['name']
        train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file,
                                          name=dataset['name'], url=dataset['url'])

        print(f"Testing dataset with name: {name}, problem type: {problem_type}")

        train_data = train_data.sample(50, random_state=1)
        test_data = test_data[test_data[label].notna()]

        if problem_type in PROBLEM_TYPES_CLASSIFICATION:
            valid_class_idxes = test_data[label].isin(train_data[label].unique())
            test_data = test_data[valid_class_idxes]

        test_data = test_data.sample(50, random_state=1)

        error_msg_og = f'pseudolabel threw an exception during fit, it should have ' \
                       f'succeeded on problem type:{problem_type} with dataset name:{name}, ' \
                       f'with problem_type: {problem_type}. Under settings:'

        # Test label already given. If test label already given doesn't use pseudo labeling filter.
        try:
            print("Pseudolabel Testing: Pre-labeled data 'fit_pseudolabel'")
            _, y_pred_proba = TabularPredictor(label=label, problem_type=problem_type).fit_pseudolabel(
                pseudo_data=test_data,
                return_pred_prob=True,
                train_data=train_data,
                **fit_args,
            )
        except Exception as e:
            assert False, error_msg_og + 'labeled test data'

        try:
            print("Pseudolabel Testing: Pre-labeled data, best quality 'fit_pseudolabel'")
            _, y_pred_proba = TabularPredictor(label=label, problem_type=problem_type).fit_pseudolabel(
                pseudo_data=test_data,
                return_pred_prob=True,
                train_data=train_data,
                **fit_args_best,
                **fit_args,
            )
        except Exception as e:
            assert False, error_msg_og + 'labeled test data, best quality'

        # Test unlabeled pseudo data
        unlabeled_test_data = test_data.drop(columns=label)
        for flag_ensemble in [True, False]:
            error_prefix = 'ensemble ' if flag_ensemble else ''
            error_msg = error_prefix + error_msg_og
            for is_weighted_ensemble in [True, False]:
                error_suffix = ' with pseudo label model weighted ensembling' if is_weighted_ensemble else ''

                try:
                    print("Pseudolabel Testing: Unlabeled data 'fit_pseudolabel'")
                    _, y_pred_proba = TabularPredictor(label=label, problem_type=problem_type).fit_pseudolabel(
                        pseudo_data=unlabeled_test_data,
                        return_pred_prob=True,
                        train_data=train_data,
                        use_ensemble=flag_ensemble,
                        fit_ensemble=is_weighted_ensemble,
                        **fit_args,
                    )
                except Exception as e:
                    assert False, error_msg + 'unlabeled test data' + error_suffix

                try:
                    print("Pseudolabel Testing: Unlabeled data, best quality 'fit_pseudolabel'")
                    _, y_pred_proba = TabularPredictor(label=label, problem_type=problem_type).fit_pseudolabel(
                        pseudo_data=unlabeled_test_data,
                        return_pred_prob=True,
                        train_data=train_data,
                        use_ensemble=flag_ensemble,
                        fit_ensemble=is_weighted_ensemble,
                        **fit_args_best,
                        **fit_args,
                    )
                except Exception as e:
                    assert False, error_msg + 'unlabeled test data, best quality' + error_suffix


@pytest.mark.slow
def test_tabularHPObagstack():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 10000 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = {'scheduler': 'local', 'searcher': 'auto'}
    num_stack_levels = 2
    num_bag_folds = 2
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 2, 'learning_rate': ag.Real(0.001, 0.01)}
        gbm_options = {'num_boost_round': 20, 'learning_rate': ag.Real(0.01, 0.1)}
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 150
        hyperparameter_tune_kwargs['num_trials'] = 3

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
        fit_args['num_bag_sets'] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)



def test_tabularHPO():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 99 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = {'scheduler': 'local', 'searcher': 'auto'}
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 2}
        gbm_options = {'num_boost_round': 20}
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60
        hyperparameter_tune_kwargs['num_trials'] = 5

    fit_args = {'verbosity': verbosity,}
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


@pytest.mark.slow
def test_tabular_feature_prune():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 99  # random seed
    subsample_size = None
    ag_args = {
        'feature_prune_kwargs': {
            'stop_threshold': 3,
            'prune_ratio': 0.05,
            'prune_threshold': None,
            'n_train_subsample': 1000,
            'n_fi_subsample': 5000,
            'min_fi_samples': 5000,
            'feature_prune_time_limit': 10,
            'raise_exception': True
        }
    }
    verbosity = 2  # how much output to print
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 1000
        gbm_options = {'num_boost_round': 20}
        hyperparameters = {'GBM': gbm_options}
        time_limit = 60

    fit_args = {'verbosity': verbosity, }
    fit_args['ag_args'] = ag_args
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


def _construct_tabular_bag_test_config(fold_fitting_strategy):
    ############ Benchmark options you can set: ########################
    num_bag_folds = 3
    num_stack_levels = 0
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 123  # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2  # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 120
        nn_options = {'num_epochs': 1}
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
        'ag_args_ensemble': {
            'fold_fitting_strategy': fold_fitting_strategy,
        },
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
        fit_args['num_bag_sets'] = 2
    ###################################################################
    config = dict(fast_benchmark=fast_benchmark, subsample_size=subsample_size,
                  perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)
    return config


def test_tabular_parallel_local_bagging():
    config = _construct_tabular_bag_test_config(PARALLEL_LOCAL_BAGGING)
    run_tabular_benchmarks(**config)


def test_tabular_sequential_local_bagging():
    config = _construct_tabular_bag_test_config(SEQUENTIAL_LOCAL_BAGGING)
    run_tabular_benchmarks(**config)


def test_sample_weight():
    dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip',
               'name': 'toyRegression',
               'problem_type': REGRESSION,
               'label': 'y',
               'performance_val': 0.183}
    directory_prefix = './datasets/'
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + dataset['name'] + "/"
    savedir = directory + 'AutogluonOutput/'
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    sample_weight = 'sample_weights'
    weights = np.abs(np.random.rand(len(train_data),))
    test_weights = np.abs(np.random.rand(len(test_data),))
    train_data[sample_weight] = weights
    test_data_weighted = test_data.copy()
    test_data_weighted[sample_weight] = test_weights
    fit_args = {'time_limit': 20}
    predictor = TabularPredictor(label=dataset['label'], path=savedir, problem_type=dataset['problem_type'], sample_weight=sample_weight).fit(train_data, **fit_args)
    ldr = predictor.leaderboard(test_data)
    perf = predictor.evaluate(test_data)
    # Run again with weight_evaluation:
    # FIXME: RMSE doesn't support sample_weight, this entire call doesn't make sense
    predictor = TabularPredictor(label=dataset['label'], path=savedir, problem_type=dataset['problem_type'], sample_weight=sample_weight, weight_evaluation=True).fit(train_data, **fit_args)
    # perf = predictor.evaluate(test_data_weighted)  # TODO: Doesn't work without implementing sample_weight in evaluate
    predictor.distill(time_limit=10)
    ldr = predictor.leaderboard(test_data_weighted)


def test_quantile():
    quantile_levels = [0.01, 0.02, 0.05, 0.98, 0.99]
    dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip',
               'name': 'toyRegression',
               'problem_type': QUANTILE,
               'label': 'y'}
    directory_prefix = './datasets/'
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + dataset['name'] + "/"
    savedir = directory + 'AutogluonOutput/'
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    fit_args = {'time_limit': 20}
    predictor = TabularPredictor(label=dataset['label'], path=savedir, problem_type=dataset['problem_type'],
                                 quantile_levels=quantile_levels).fit(train_data, **fit_args)
    ldr = predictor.leaderboard(test_data)
    perf = predictor.evaluate(test_data)


@pytest.mark.skip(reason="Ignored for now, since stacking is disabled without bagging.")
def test_tabular_stack1():
    ############ Benchmark options you can set: ########################
    num_stack_levels = 1
    num_bag_folds = 0
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 32 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 3}
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val = seed_val, fit_args=fit_args)


@pytest.mark.skip(reason="Ignored for now, since stacking is disabled without bagging.")
def test_tabular_stack2():
    ############ Benchmark options you can set: ########################
    num_stack_levels = 2
    num_bag_folds = 0
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 66 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {'num_epochs': 3}
        gbm_options = {'num_boost_round': 30}
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


@pytest.mark.slow
def test_tabular_bagstack():
    ############ Benchmark options you can set: ########################
    num_stack_levels = 2
    num_bag_folds = 3
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 53 # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2 # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 105
        nn_options = {'num_epochs': 2}
        gbm_options = [{'num_boost_round': 40}, 'GBMLarge']
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
        fit_args['num_bag_sets'] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args, run_distill=True)


@pytest.mark.slow
def test_tabular_bagstack_use_bag_holdout():
    ############ Benchmark options you can set: ########################
    num_stack_levels = 2
    num_bag_folds = 3
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 53  # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2  # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 105
        nn_options = {'num_epochs': 2}
        gbm_options = [{'num_boost_round': 40}, 'GBMLarge']
        hyperparameters = {'GBM': gbm_options, 'NN_TORCH': nn_options}
        time_limit = 60

    fit_args = {
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'verbosity': verbosity,
        'use_bag_holdout': True,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters
    if time_limit is not None:
        fit_args['time_limit'] = time_limit
        fit_args['num_bag_sets'] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args, run_distill=True, crash_in_oof=True)
