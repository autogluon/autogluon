""" Runs autogluon.tabular on multiple benchmark datasets.
    Lower performance-values = better, normalized to [0,1] for each dataset to enable cross-dataset comparisons.
    Classification performance = error-rate

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
    - text features in dataset
"""
import warnings, shutil, os
import numpy as np
import mxnet as mx
from random import seed

import pytest
import autogluon as ag

from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS

supervised=True
semi_supervised=True

def test_tabular():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 0 # random seed
    subsample_size = None
    hyperparameter_tune = False
    verbosity = 2 # how much output to print
    hyperparameters = None

    subsample_size = 50 #50
    unlabeled_subsample_size = 50000


    fit_args = {
        'hyperparameter_tune': hyperparameter_tune,
        'verbosity': verbosity,
    }
    if hyperparameters is not None:
        fit_args['hyperparameters'] = hyperparameters

    ###################################################################
    run_tabular_benchmarks(subsample_size=subsample_size, unlabeled_subsample_size=unlabeled_subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)



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


def run_tabular_benchmarks(subsample_size, unlabeled_subsample_size, perf_threshold, seed_val, fit_args, dataset_indices=None, run_distill=False):
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


    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    datasets = [multi_dataset] #multi_dataset]

    if dataset_indices is not None: # only run some datasets
        datasets = [datasets[i] for i in dataset_indices]

    # Aggregate performance summaries obtained in previous benchmark run:
    prev_perf_vals = [dataset['performance_val'] for dataset in datasets]
    previous_avg_performance = np.mean(prev_perf_vals)
    previous_median_performance = np.median(prev_perf_vals)
    previous_worst_performance = np.max(prev_perf_vals)

    # Run benchmark:
    sup_performance_vals = [0.0] * len(datasets) # performance obtained in this run
    semi_sup_performance_vals = [0.0] * len(datasets)
    directory_prefix = './datasets/'
    with warnings.catch_warnings(record=True) as caught_warnings:
        for idx in range(len(datasets)):
            dataset = datasets[idx]
            data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset['name'], url=dataset['url'])
          
            if seed_val is not None:
                seed(seed_val)
                np.random.seed(seed_val)
                mx.random.seed(seed_val)
            print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset['name'], idx+1, len(datasets)))
            directory = directory_prefix + dataset['name'] + "/"
            savedir = directory + 'AutogluonOutput/'
            shutil.rmtree(savedir, ignore_errors=True) # Delete AutoGluon output directory to ensure previous runs' information has been removed.
            label_column = dataset['label_column']

            test_data = test_data.sample(frac=1).head(10000)
            y_test = test_data[label_column]
            test_data = test_data.drop(labels=[label_column], axis=1)

        
            if subsample_size is None:
                raise ValueError("subsample_size not specified")
            train_data = data.head(subsample_size) 
            unlabeled_data = data.head(unlabeled_subsample_size).drop(columns=[label_column])


            custom_hyperparameters = {"Transf": {}}
            
            if supervised:
                predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters, problem_type=dataset['problem_type'], output_directory=savedir, **fit_args)
                results = predictor.fit_summary(verbosity=0)
                if predictor.problem_type != dataset['problem_type']:
                    warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
                predictor = task.load(savedir)  # Test loading previously-trained predictor from file
                y_pred = predictor.predict(test_data)
                perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
                if dataset['problem_type'] != REGRESSION:
                    perf = 1.0 - perf_dict['accuracy_score'] # convert accuracy to error-rate
                else:
                    perf = 1.0 - perf_dict['r2_score'] # unexplained variance score.
                sup_performance_vals[idx] = perf
                print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset['name'], sup_performance_vals[idx], dataset['performance_val']))

                if run_distill:
                    predictor.distill(time_limits=60, augment_args={'size_factor':0.5})


            if semi_supervised:
                predictor = task.fit(train_data=train_data, label=label_column, problem_type=dataset['problem_type'], unlabeled_data=unlabeled_data, hyperparameters=custom_hyperparameters, output_directory=savedir, **fit_args)
                results = predictor.fit_summary(verbosity=0)
                if predictor.problem_type != dataset['problem_type']:
                    warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
                predictor = task.load(savedir)  # Test loading previously-trained predictor from file
                y_pred = predictor.predict(test_data)
                perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
                if dataset['problem_type'] != REGRESSION:
                    perf = 1.0 - perf_dict['accuracy_score'] # convert accuracy to error-rate
                else:
                    perf = 1.0 - perf_dict['r2_score'] # unexplained variance score.
                semi_sup_performance_vals[idx] = perf
                print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset['name'], semi_sup_performance_vals[idx], dataset['performance_val']))

                if run_distill:
                    predictor.distill(time_limits=60, augment_args={'size_factor':0.5})

    # Summarize:
    avg_perf = np.mean(sup_performance_vals)
    median_perf = np.median(sup_performance_vals)
    worst_perf = np.max(sup_performance_vals)
    for idx in range(len(datasets)):
        print("Performance on dataset %s: %s   (previous perf=%s)" % (datasets[idx]['name'], sup_performance_vals[idx], datasets[idx]['performance_val']))

    print("Supervised Average performance: %s" % avg_perf)
    print("Supervised Median performance: %s" % median_perf)
    print("Supervised Worst performance: %s" % worst_perf)

    avg_perf = np.mean(semi_sup_performance_vals)
    median_perf = np.median(semi_sup_performance_vals)
    worst_perf = np.max(semi_sup_performance_vals)
    for idx in range(len(datasets)):
        print("Performance on dataset %s: %s   (previous perf=%s)" % (datasets[idx]['name'], semi_sup_performance_vals[idx], datasets[idx]['performance_val']))
    
    print("Semi-supervised Average performance: %s" % avg_perf)
    print("Semi-supervised Median performance: %s" % median_perf)
    print("Semi-supervised Worst performance: %s" % worst_perf)


    print("Ran fit with args:")
    print(fit_args)
    # List all warnings again to make sure they are seen:
    print("\n\n WARNINGS:")
    for w in caught_warnings:
        warnings.warn(w.message)



if __name__ == "__main__":
    test_tabular()

