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

import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


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
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold,
                           seed_val=seed_val, fit_args=fit_args)


def run_tabular_benchmarks(fast_benchmark, subsample_size, perf_threshold, seed_val, fit_args, dataset_indices=None):
    print("Running fit with args:")
    print(fit_args)
    # Each train/test dataset must be located in single directory with the given names.
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    EPS = 1e-10

    # Information about each dataset in benchmark is stored in dict.
    # performance_val = expected performance on this dataset (lower = better),should update based on previously run benchmarks
    binary_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
                      'name': 'AdultIncomeBinaryClassification',
                      'problem_type': BINARY,
                      'label_column': 'class',
                      'performance_val': 0.129} # Mixed types of features.

    multi_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
                      'name': 'CoverTypeMulticlassClassification',
                      'problem_type': MULTICLASS,
                      'label_column': 'Cover_Type',
                      'performance_val': 0.032} # big dataset with 7 classes, all features are numeric. Runs SLOW.

    regression_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
                       'name': 'AmesHousingPriceRegression',
                      'problem_type': REGRESSION,
                      'label_column': 'SalePrice',
                      'performance_val': 0.076} # Regression with mixed feature-types, skewed Y-values.

    toyregres_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/toyRegression.zip', 
                         'name': 'toyRegression',
                         'problem_type': REGRESSION, 
                        'label_column': 'y', 
                        'performance_val': 0.183}
    # 1-D toy deterministic regression task with: heavy label+feature missingness, extra distraction column in test data

    toyclassif_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/toyClassification.zip',
                         'name': 'toyClassification',
                         'problem_type': MULTICLASS, 
                        'label_column': 'y', 
                        'performance_val': 0.436}
    # 2-D toy noisy, imbalanced 4-class classification task with: feature missingness, out-of-vocabulary feature categories in test data, out-of-vocabulary labels in test data, training column missing from test data, extra distraction columns in test data
    # toyclassif_dataset should produce 3 warnings:
    # UserWarning: These columns from this dataset were not present in the training dataset (AutoGluon will ignore them):  ['distractioncolumn1', 'distractioncolumn2']
    # UserWarning: The columns listed below from the training data are no longer in the given dataset. (AutoGluon will proceed assuming their values are missing, but you should remove these columns from training dataset and train a new model):  ['lostcolumn']
    # UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    datasets = [toyregres_dataset, toyclassif_dataset, binary_dataset, regression_dataset, multi_dataset]
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
    if not os.path.exists(directory_prefix):
        os.mkdir(directory_prefix)
    with warnings.catch_warnings(record=True) as caught_warnings:
        for idx in range(len(datasets)):
            if seed_val is not None:
                seed(seed_val)
                np.random.seed(seed_val)
                mx.random.seed(seed_val)
            dataset = datasets[idx]
            print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset['name'], idx+1, len(datasets)))
            directory = directory_prefix + dataset['name'] + "/"
            train_file_path = directory + train_file
            test_file_path = directory + test_file
            if (not os.path.exists(train_file_path)) or (not os.path.exists(test_file_path)):
                # fetch files from s3:
                print("%s data not found locally, so fetching from %s" % (dataset['name'],  dataset['url']))
                zip_name = ag.download(dataset['url'], directory_prefix)
                ag.unzip(zip_name, directory_prefix)
                os.remove(zip_name)

            savedir = directory + 'AutogluonOutput/'
            shutil.rmtree(savedir, ignore_errors=True) # Delete AutoGluon output directory to ensure previous runs' information has been removed.
            label_column = dataset['label_column']
            train_data = task.Dataset(file_path=train_file_path)
            test_data = task.Dataset(file_path=test_file_path)
            y_test = test_data[label_column]
            test_data = test_data.drop(labels=[label_column], axis=1)
            if fast_benchmark:
                if subsample_size is None:
                    raise ValueError("fast_benchmark specified without subsample_size")
                train_data = train_data.head(subsample_size) # subsample for fast_benchmark
            predictor = None # reset from last Dataset
            predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, **fit_args)
            results = predictor.fit_summary(verbosity=0)
            if predictor.problem_type != dataset['problem_type']:
                warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
            predictor = None  # We delete predictor here to test loading previously-trained predictor from file
            predictor = task.load(savedir)
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


if __name__ == '__main__':
    test_tabular()
