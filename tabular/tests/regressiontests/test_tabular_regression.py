""" Runs autogluon.tabular on synthetic classification and regression datasets.
    We test various parameters to TabularPredictor() and fit()
    We then check the leaderboard:
       Did we run the expected list of models
       Did each model have the expected score (within given range)
       Did the ensembling produce the expected score (within given range)
    This helps us spot any change in model performance.
    If any changes are spotted, the script does its best to dump out new proposed score ranges
    that you can cut and paste into the tests.  Only do this once you've identified the cause!

    Potential naming confusion: 
        - this is a *regression* test, to make sure no functionality has accidentally got worse (testing terminology)
        - it runs two types of TabularPredictor tests : *regression* and classification (ML terminology)

    These tests are designed to run fast, to permit them to be run on a github hook.
    Currently the 11 tests, calling TabularPredictor.fit() 20 times, run in ~8 minutes on an 8 vcore machine with no GPU.

    Format of a test:  

    {   # Default regression model on a small dataset
        'name':             # some unique name
        'type':             # either 'regression' or 'classification'.  We make a dataset using 
                            # scikit-learn.make_{regression,classification}
        'n_samples':        # number of rows in the training dataset.  With TEST_SIZE default of 0.5, 
                            # we make an additional n_samples for testing.
        'n_features': 2,    # number of columns.
        'n_categorical': 0, # number of categorical (discrete not continuous) columns.
        ''dataset_hash' :   # Hash of synthetic dataset to ensure the dataset itself didn't change.
        'params' : [ { 'predict' : {}, 'fit' : {} },   # If an array, we call TabularPredictor multiple times with 
                                                       # different parameters.
                     { 'predict' : {}, 'fit' : {} },   # Pass the additional parameters to predict(), fit() or both 
                                                       # in the dicts.
                                                       # If a scalar, we only call TabularPredictor once.
                   ],
        'expected_score_range' : {                     # A list of models we expect to run, and a valid score range we 
                                                       # expect from each model.
                  'CatBoost': (-7.86, 0.01),           # The first value is the lower bound, the 2nd value is a delta 
                  'ExtraTreesMSE': (-7.88, 0.01),      # to compute the upper bound, e.g. ( -8.12, 0.01 ) means we 
                                                       # expect the score to be from -8.12 to -8.11 inclusive.
                  'CatBoost_BAG_L1': (np.nan, np.nan), # If np.nan, we expect this model to return np.nan as the score.
        },
    },

    Testing by @willsmithorg on master AG as of 2022-02-22 - 2022-02-23:
    Tested on AWS Linux instance m5.2xlarge, amzn2-ami-kernel-5.10-hvm-2.0.20211223.0-x86_64-gp2 with 
                                               (8  vcore, no GPU, Python==3.7.10, scikit-learn==1.0.2, torch==1.10.2), 
    Tested on Github jenkins Linux:
                                               (?  vcore,  0 GPU, Python==3.9.10, scikit-learn==1.0.2, torch==1.10.2), 
    Tested on AWS Windows instance t3.xlarge, 
                                               (4  vcore,  0 GPU, Python==3.9.7 , scikit-learn==1.0.2, torch==1.10.2), 
                                               - Pytorch scores are slighty different, all else same.

"""
import sys
import math
import hashlib
import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import pytest

from autogluon.tabular import TabularDataset, TabularPredictor


tests = [
    # 
    # Regressions
    # 
    {   # Default regression model on a small dataset
        'name': 'small regression',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : [ { 'predict' : {}, 'fit' : {} },          # All of the followiing params should return same results because they're defaults
                     { 'predict' : {}, 'fit' : { 'presets' : 'medium_quality_faster_train' } }, 
                     { 'predict' : {}, 'fit' : { 'presets' : 'ignore_text' } }, 
                     { 'predict' : {}, 'fit' : { 'hyperparameters' : 'default' } }, 
                     { 'predict' : { 'eval_metric' : 'root_mean_squared_error'}, 'fit' : { } }, 
                   ], 
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'ExtraTreesMSE': (-7.88, 0.01),
                  'KNeighborsDist': (-8.69, 0.01),
                  'KNeighborsUnif': (-9.06, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'RandomForestMSE': (-9.63, 0.01),
                  'WeightedEnsemble_L2': (-5.66, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # If we explictly exclude some models the others should return unchanged and the ensemble result will be changed.
        'name': 'small regression excluded models',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'excluded_model_types' : [ 'KNN', 'RF', 'XT', 'GBM', 'CAT', 'XGB' ] } },
        'expected_score_range' : {
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'WeightedEnsemble_L2': (-5.09, 0.01),
        },
    },
    {   # Small regression, hyperparameters = light removes some models
        'name': 'small regression light hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'light' } },
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'ExtraTreesMSE': (-7.87, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'RandomForestMSE': (-9.63, 0.01),
                  'WeightedEnsemble_L2': (-5.66, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # Small regression, hyperparameters = very_light removes some models
        'name': 'small regression very light hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'very_light' } },
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'WeightedEnsemble_L2': (-5.58, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # Small regression, hyperparameters = toy removes almost all models and runs very fast
        'name': 'small regression toy hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'toy' } },
        'expected_score_range' : { 
                  'CatBoost': (-28.39, 0.01),
                  'LightGBM': (-27.81, 0.01),
                  'NeuralNetTorch': (-27.11, 0.01),
                  'WeightedEnsemble_L2': (-19.12, 0.01),
                  'XGBoost': (-19.12, 0.01),
        },
    },
    {   # High quality preset on small datset.
        'name': 'small regression high quality',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'presets' : 'high_quality_fast_inference_only_refit' } }, 
        'expected_score_range' : {
                  'CatBoost_BAG_L1': (np.nan, np.nan),
                  'CatBoost_BAG_L1_FULL': (-7.75, 0.01),
                  'ExtraTreesMSE_BAG_L1': (-7.52, 0.01),
                  'ExtraTreesMSE_BAG_L1_FULL': (-7.52, 0.01),
                  'KNeighborsDist_BAG_L1': (-8.21, 0.01),
                  'KNeighborsDist_BAG_L1_FULL': (-8.21, 0.01),
                  'KNeighborsUnif_BAG_L1': (-8.7, 0.01),
                  'KNeighborsUnif_BAG_L1_FULL': (-8.7, 0.01),
                  'LightGBMLarge_BAG_L1': (np.nan, np.nan),
                  'LightGBMLarge_BAG_L1_FULL': (-9.94, 0.01),
                  'LightGBMXT_BAG_L1': (np.nan, np.nan),
                  'LightGBMXT_BAG_L1_FULL': (-13.03, 0.01),
                  'LightGBM_BAG_L1': (np.nan, np.nan),
                  'LightGBM_BAG_L1_FULL': (-14.17, 0.01),
                  'NeuralNetFastAI_BAG_L1': (np.nan, np.nan),
                  'NeuralNetFastAI_BAG_L1_FULL': (-5.48, 0.01),
                  'NeuralNetTorch_BAG_L1': (np.nan, np.nan),
                  'NeuralNetTorch_BAG_L1_FULL': (-5.29, 0.01),
                  'RandomForestMSE_BAG_L1': (-9.5, 0.01),
                  'RandomForestMSE_BAG_L1_FULL': (-9.5, 0.01),
                  'WeightedEnsemble_L2': (np.nan, np.nan),
                  'WeightedEnsemble_L2_FULL': (-5.29, 0.01),
                  'XGBoost_BAG_L1': (np.nan, np.nan),
                  'XGBoost_BAG_L1_FULL': (-9.76, 0.01),
        }
    },
    {   # Best quality preset on small datset.
        'name': 'small regression best quality',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'presets' : 'best_quality' } }, 
        'expected_score_range' : {
                  'CatBoost_BAG_L1': (-7.85, 0.01),
                  'ExtraTreesMSE_BAG_L1' : (-7.52, 0.01),
                  'KNeighborsDist_BAG_L1' : (-8.21, 0.01),
                  'KNeighborsUnif_BAG_L1' : (-8.70, 0.01),
                  'LightGBMLarge_BAG_L1' : (-9.44, 0.01),
                  'LightGBMXT_BAG_L1' : (-14.78, 0.01),
                  'LightGBM_BAG_L1' : (-14.92, 0.01),
                  'NeuralNetFastAI_BAG_L1' : (-5.55, 0.01),
                  'NeuralNetTorch_BAG_L1' : (-5.07, 0.01),
                  'RandomForestMSE_BAG_L1' : (-9.5, 0.01),
                  'WeightedEnsemble_L2' : (-5.05, 0.01),   # beats default, as expected
                  'XGBoost_BAG_L1' : (-9.74, 0.01),
        }
    },
    {   # Default regression model, add some categorical features.
        'name': 'small regression with categorical',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 1,
        'dataset_hash' : '3e26d128e0',
        'params' : { 'predict' : {}, 'fit' : {} },          # Default params
        'expected_score_range' : {
                 'CatBoost': (-22.58, 0.01),
                 'ExtraTreesMSE': (-25.09, 0.01),
                 'KNeighborsDist': (-39.45, 0.01),
                 'KNeighborsUnif': (-35.64, 0.01),
                 'LightGBM': (-32.96, 0.01),
                 'LightGBMLarge': (-34.86, 0.01),
                 'LightGBMXT': (-32.69, 0.01),
                 'NeuralNetFastAI': (-22.11, 0.01),
                 'NeuralNetTorch': (-19.76, 0.01),
                 'RandomForestMSE': (-27.49, 0.01),
                 'WeightedEnsemble_L2': (-19.76, 0.01),
                 'XGBoost': (-24.93, 0.01),

        }
    },
    {   # Default regression model different metric
        'name': 'small regression metric mae',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : { 'eval_metric' : 'mean_absolute_error'}, 'fit' : { } }, 
        'expected_score_range' : {
                  'CatBoost': (-5.23, 0.01),
                  'ExtraTreesMSE': (-5.48, 0.01),
                  'KNeighborsDist': (-6.16, 0.01),
                  'KNeighborsUnif': (-6.61, 0.01),
                  'LightGBM': (-11.97, 0.01),
                  'LightGBMLarge': (-7.69, 0.01),
                  'LightGBMXT': (-12.37, 0.01),
                  'NeuralNetFastAI': (-4.74, 0.01),
                  'NeuralNetTorch': (-3.77, 0.01),
                  'RandomForestMSE': (-6.96, 0.01),
                  'WeightedEnsemble_L2': (-4.03, 0.01),
                  'XGBoost': (-8.32, 0.01),

        },
    },
    # 
    # Classifications
    # 
    {   # Default classification model on a small dataset
        'name': 'small classification',
        'type': 'classification',
        'n_samples': 400,  # With only 8 classes it's hard to compare model quality unless we have a biggest test set (and therefore train set).
        'n_features': 10,
        'n_informative': 5,
        'n_classes': 8,
        'n_categorical': 0,
        'dataset_hash' : 'be1f16df80',
        'params' : [ { 'predict' : {}, 'fit' : {} },     # All of the followiing params should return same results
                     { 'predict' : {}, 'fit' : { 'presets' : 'medium_quality_faster_train' } }, 
                     { 'predict' : {}, 'fit' : { 'presets' : 'ignore_text' } }, 
                     { 'predict' : {}, 'fit' : { 'hyperparameters' : 'default' } }, 
                     { 'predict' : { 'eval_metric' : 'accuracy'}, 'fit' : { } }, 
                   ], 
        'expected_score_range' : {
                 'CatBoost': (0.245, 0.001),            # Classification scores are low numbers so we decrease the
                 'ExtraTreesEntr': (0.327, 0.001),      # tolerance to 0.001 to make sure we pick up changes.
                 'ExtraTreesGini': (0.32, 0.001),
                 'KNeighborsDist': (0.337, 0.001),
                 'KNeighborsUnif': (0.322, 0.001),
                 'LightGBM': (0.197, 0.001),
                 'LightGBMLarge': (0.265, 0.001),
                 'LightGBMXT': (0.23, 0.001),
                 'NeuralNetFastAI': (0.34, 0.001),
                 'NeuralNetTorch': (0.232, 0.001),
                 'RandomForestEntr': (0.305, 0.001),
                 'RandomForestGini': (0.295, 0.001),
                 'WeightedEnsemble_L2': (0.34, 0.001),
                 'XGBoost': (0.227, 0.001),
        }
    },
    {   # There's different logic for boolean classification so let's test that with n_classes = 2.
        'name': 'small classification boolean',
        'type': 'classification',
        'n_samples': 400,  
        'n_features': 10,
        'n_informative': 5,
        'n_classes': 2,
        'n_categorical': 0,
        'dataset_hash' : '79e634aac3',
        'params' : [ { 'predict' : {}, 'fit' : {} },      # All of the followiing params should return same results
                     { 'predict' : { 'eval_metric' : 'accuracy'}, 'fit' : { } }, 
                   ], 
        'expected_score_range' : {
                  'CatBoost': (0.61, 0.001),
                  'ExtraTreesEntr': (0.607, 0.001),
                  'ExtraTreesGini': (0.6, 0.001),
                  'KNeighborsDist': (0.61, 0.001),
                  'KNeighborsUnif': (0.61, 0.001),
                  'LightGBM': (0.632, 0.001),
                  'LightGBMLarge': (0.552, 0.001),
                  'LightGBMXT': (0.612, 0.001),
                  'NeuralNetFastAI': (0.62, 0.001),
                  'NeuralNetTorch': (0.597, 0.001),
                  'RandomForestEntr': (0.607, 0.001),
                  'RandomForestGini': (0.582, 0.001),
                  'WeightedEnsemble_L2': (0.61, 0.001),
                  'XGBoost': (0.58, 0.001),
        }
    },
]


# Lots of test data since inference is fast and we want a score that's very reflective of model quality, 
# despite very fast training times.
TEST_SIZE=0.5 
def make_dataset(request, seed):
    # Ensure our datasets and model calls remain deterministic.
    random.seed(seed)
    np.random.seed(seed)
    if request['type'] == 'regression':

        x, y = make_regression(n_samples = int(request['n_samples']*(1/(1-TEST_SIZE))),
                               n_features = request['n_features'], 
                               noise=4) # To make it hard enough that we get better performance on slower models
    elif request['type'] == 'classification':
        x, y = make_classification(n_samples = int(request['n_samples']*(1/(1-TEST_SIZE))),
                               n_features = request['n_features'], 
                               n_informative = request['n_informative'], 
                               n_redundant = request['n_classes'] - request['n_informative'],
                               n_classes = request['n_classes'],
                               class_sep=0.4) # To make it hard enough that we get better performance on slower models
    else:
        assert False, "Unrecognised request type '{request['type'}'"

 
    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y, columns=['label'])

    # Make some columns categorical if required.
    if request['n_categorical'] > 0:
        cols_to_convert = random.sample(set(dfx.columns.values), k=request['n_categorical'])
        for col in cols_to_convert:
            dfx[col] = dfx[col].astype(int)
            vals = np.unique(dfx[col])        
            # Shuffle the categoricals so there's no pattern in their ordering. 
            vals2 = vals.copy()-min(vals)
            np.random.shuffle(vals2)
            mapper = dict(zip(vals, vals2))
            dfx[col] = dfx[col].map(mapper)
            dfx[col] = dfx[col].astype("category")

    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=TEST_SIZE)
    dftrain = pd.concat([x_train, y_train], axis=1)
    dftest  = pd.concat([x_test,  y_test],  axis=1)

    return (dftrain, dftest)

# Round to given accuracy.  The 8 is to remove floating point rounding errors.
def myfloor(x, base=.01):
  return round(base * math.floor(float(x)/base),8)

@pytest.mark.regression
def inner_test_tabular(testname):

    # Find the named test
    test = None
    for t in tests:
        if t['name'] == testname:
            test = t
    assert test is not None, f"Could not find test {testname}"
 
    # Build the dataset
    (dftrain, dftest) = make_dataset(request=test, seed=0)

    # Check the synthetic dataset itself hasn't changed.  We round it to 3dp otherwise tiny floating point differences  
    # between platforms can give a different hash that still yields same prediction scores.
    # Ultimately it doesn't matter how we do this as long as the same dataset gives the same hash function on
    # different python versions and architectures.
    current_hash = hashlib.sha256(dftrain.round(decimals=3).values.tobytes()).hexdigest()[0:10]
    proposedconfig = "Proposed new config:\n"
    proposedconfig += f"'dataset_hash' : '{current_hash}',"
    assert current_hash == test['dataset_hash'], f"Test '{testname}' input dataset has changed.  All scores will change.\n" + proposedconfig

    # Now run the Predictor 1 or more times with various parameters, and make sure we get
    # back the expected results.

    # Params can either omitted, or a single run, or a list of runs.
    if 'params' not in test:
        test['params'] = { 'predict' : {}, 'fit' : {} }
    if not isinstance(test['params'], list):
        test['params'] = [ test['params'] ]
    for params in test['params']:

        # Run this model and set of params		
        predictor = TabularPredictor(label='label', **params['predict'])
        predictor.fit(dftrain, **params['fit'])
        leaderboard = predictor.leaderboard(dftest, silent=True)
        leaderboard = leaderboard.sort_values(by='model') # So we can pre-generate sample config in alphabetical order

        # Store proposed new config based on the current run, in case the developer wants to keep thee results (just cut and paste).
        proposedconfig = "Proposed new config:\n"
        proposedconfig += "'expected_score_range' : {\n";
        for model in leaderboard['model']:
            midx_in_leaderboard = leaderboard.index.values[leaderboard['model'] == model][0]
            if np.isnan(leaderboard['score_test'][midx_in_leaderboard]): 
                 values = "np.nan, np.nan"
            else:
                 if model in test['expected_score_range'] and not np.isnan(test['expected_score_range'][model][1]):
                     currentprecision = test['expected_score_range'][model][1]
                 else:
                     currentprecision = 0.01
                 values = "{}, {}".format(myfloor(leaderboard['score_test'][midx_in_leaderboard], currentprecision), currentprecision)
            proposedconfig += f"    '{model}': ({values}),\n"
        proposedconfig += "},\n"

        # First validate the model list was as expected.
        assert set(leaderboard['model']) == set(test['expected_score_range'].keys()), (f"Test '{testname}' params {params} got unexpected model list.\n" + proposedconfig)

        # Now validate the scores for each model were as expected.
        all_assertions_met = True
        currentconfig = "Existing config:\n"
        currentconfig += "'expected_score_range' : {\n";
        for model in sorted(test['expected_score_range']):
            midx_in_leaderboard = leaderboard.index.values[leaderboard['model'] == model][0]
            assert leaderboard['model'][midx_in_leaderboard] == model
            expectedrange = test['expected_score_range'][model][1]
            expectedmin = test['expected_score_range'][model][0]
            expectedmax = expectedmin + expectedrange

            if np.isnan(expectedmin):
                 values = "np.nan, np.nan"
            else:
                 values = "{}, {}".format(expectedmin, expectedrange)
            
            if (((leaderboard['score_test'][midx_in_leaderboard] >= expectedmin) and 
               (leaderboard['score_test'][midx_in_leaderboard] <= expectedmax)) or  
               (np.isnan(leaderboard['score_test'][midx_in_leaderboard]) and np.isnan(expectedmin))):
                currentconfig += f"    '{model}': ({values}),\n"
            else:
                currentconfig += f"    '{model}': ({values}), # <--- not met, got {leaderboard['score_test'][midx_in_leaderboard]} \n"
                all_assertions_met = False
        currentconfig += "},\n"

        assert all_assertions_met, f"Test '{testname}', params {params} had unexpected scores:\n" + currentconfig + proposedconfig

        # Clean up this model created with specific params.
        predictor.delete_models(models_to_keep=[], dry_run=False)  

	

# The tests are all run individually rather than in 1 big loop that simply goes through the tests dictionary.
# This is so we easily remove some tests if necessary.
@pytest.mark.parametrize("testname", [
    'small regression',
    'small regression excluded models',
    'small regression light hyperparameters',
    'small regression very light hyperparameters',
    'small regression toy hyperparameters',
    'small regression high quality',
    'small regression best quality',
    'small regression with categorical',
    'small regression metric mae',
    'small classification',
    'small classification boolean',
])

# These results have only been confirmed for Linux.  Windows is known to give different results for Pytorch.
@pytest.mark.skipif(sys.platform != 'linux', reason='Scores only confirmed on Linux')
@pytest.mark.regression
def test_tabular_score(testname):
    inner_test_tabular(testname)
