""" Runs autogluon.tabular on multiple benchmark datasets. 
    Run this benchmark to assess whether major chances make autogluon better or worse overall.
    Lower performance-values = better, normalized to [0,1] for each dataset to enable cross-dataset comparisons.
    Classification performance = error-rate, Regression performance = 1 - R^2
    
    # TODO: assess that Autogluon has correctly inferred the type of each feature (continuous vs categorical vs text)
    
    # TODO: suppress internal AutoGluon print statements, so that only benchmark-info is printed
    
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

import warnings, shutil
import numpy as np
import mxnet as mx
from random import seed

from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon import PredictTableColumn as task
import autogluon as ag
 
############ Benchmark options you can set: ########################
hyperparameter_tune = False
fast_benchmark = True # False # If True, run a faster benchmark (subsample training sets, less epochs, etc.)
                       # Please disregard performance_value warnings when fast_benchmark = True.
subsample_size = 1000
perf_threshold = 1.1 # How much worse can performance on each dataset be vs previous performance without warning
# nn_options =  {} # or set = None to omit neural net
# gbm_options = {} # or set = None to omit gradient boosting
nn_options = {'num_epochs': 5} # Can control model training time here.
gbm_options = {'num_boost_round': 100}
###################################################################

# Each train/test dataset must be located in single directory with the given names.
train_file = 'train_data.csv'
test_file = 'test_data.csv'
seed_val = 0 # random seed
EPS = 1e-10

# Information about each dataset in benchmark is stored in dict.
# performance_val = expected performance on this dataset (lower = better), Should update based on previously run benchmarks
binary_dataset = {'folder': '~/WorkDocs/Datasets/AdultIncomeBinaryClassification/',
                  'name': 'AdultIncomeBinary',
                  'problem_type': BINARY,
                  'label_column': 'class',
                  'performance_val': 0.129} # Mixed types of features.

multi_dataset = {'folder': '~/WorkDocs/Datasets/CoverTypeMulticlassClassification/',
                  'name': 'CoverTypeMulticlass',
                  'problem_type': MULTICLASS,
                  'label_column': 'Cover_Type',
                  'performance_val': 0.032} # big dataset with 7 classes, all features are numeric. Runs SLOW.

regression_dataset = {'folder': '~/WorkDocs/Datasets/AmesHousingPriceRegression/',
                   'name': 'AmesHousingRegression',
                  'problem_type': REGRESSION,
                  'label_column': 'SalePrice',
                  'performance_val': 0.076} # Regression with mixed feature-types, skewed Y-values.

toyregres_dataset = {'folder': '~/WorkDocs/Datasets/AutogluonToyBenchmark/toyRegression/', 
                     'name': 'ToyRegression',
                     'problem_type': REGRESSION, 
                    'label_column': 'y', 
                    'performance_val': 0.183}
# 1-D toy deterministic regression task with: heavy label+feature missingness, extra distraction column in test data

toyclassif_dataset = {'folder': '~/WorkDocs/Datasets/AutogluonToyBenchmark/toyClassification/', 
                     'name': 'ToyClassification',
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

# Aggregate performance summaries obtained in previous benchmark run:
prev_perf_vals = [dataset['performance_val'] for dataset in datasets]
previous_avg_performance = np.mean(prev_perf_vals)
previous_median_performance = np.median(prev_perf_vals)
previous_worst_performance = np.max(prev_perf_vals)

# Run benchmark:
performance_vals = [0.0] * len(datasets) # performance obtained in this run
with warnings.catch_warnings(record=True) as caught_warnings:
    for idx in range(len(datasets)):
        seed(seed_val)
        np.random.seed(seed_val)
        mx.random.seed(seed_val)
        dataset = datasets[idx]
        print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset['name'], idx+1, len(datasets)))
        directory = dataset['folder']
        train_file_path = directory + train_file
        test_file_path = directory + test_file
        savedir = directory + 'AutogluonOutput/'
        shutil.rmtree(savedir, ignore_errors=True) # Delete AutoGluon output directory to ensure previous runs' information has been removed.
        label_column = dataset['label_column']
        train_data = task.Dataset(file_path=train_file_path)
        test_data = task.Dataset(file_path=test_file_path)
        y_test = test_data[label_column]
        test_data = test_data.drop(labels=[label_column], axis=1)
        if fast_benchmark:
            train_data = train_data.head(subsample_size) # subsample for fast_benchmark
        predictor = None # reset from last Dataset
        predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, 
                             hyperparameter_tune=hyperparameter_tune, 
                             nn_options=nn_options, gbm_options=gbm_options)
        if predictor.problem_type != dataset['problem_type']:
            warnings.warn("For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset['name'], predictor.problem_type, dataset['problem_type']))
        predictor = None  # We delete predictor here to test loading previously-trained predictor from file
        predictor = task.load(savedir)
        y_pred = predictor.predict(test_data)
        perf_dict = predictor.evaluate(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        if dataset['problem_type'] != REGRESSION:
            perf = 1.0 - perf_dict['accuracy_score'] # convert accuracy to error-rate
        else:
            perf = 1.0 - perf_dict['r2_score'] # unexplained variance score.
        performance_vals[idx] = perf
        print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset['name'], performance_vals[idx], dataset['performance_val']))
        if performance_vals[idx] > dataset['performance_val'] * perf_threshold:
            warnings.warn("Performance on dataset %s is %s times worse than previous performance." % (dataset['name'], performance_vals[idx]/(EPS+dataset['performance_val'])))

# Summarize:
avg_perf = np.mean(performance_vals)
median_perf = np.median(performance_vals)
worst_perf = np.max(performance_vals)
if avg_perf > previous_avg_performance * perf_threshold:
    warnings.warn("Average Performance is %s times worse than previously." % (avg_perf/(EPS+previous_avg_performance)))

if median_perf > previous_median_performance * perf_threshold:
    warnings.warn("Median Performance is %s times worse than previously." % (median_perf/(EPS+previous_median_performance)))

if worst_perf > previous_worst_performance * perf_threshold:
    warnings.warn("Worst Performance is %s times worse than previously." % (worst_perf/(EPS+previous_worst_performance)))

for idx in range(len(datasets)):
    print("Performance on dataset %s: %s   (previous perf=%s)" % (datasets[idx]['name'], performance_vals[idx], datasets[idx]['performance_val']))

print("Average performance: %s" % avg_perf)
print("Median performance: %s" % median_perf)
print("Worst performance: %s" % worst_perf)

# List all warnings again to make sure they are seen:
print("\n\n WARNINGS:")
for w in caught_warnings:
    warnings.warn(w.message)
