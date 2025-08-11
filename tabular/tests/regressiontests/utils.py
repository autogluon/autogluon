import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

"""Format of a test:
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
"""
tests = [
    #
    # Regressions
    #
    {  # Default regression model on a small dataset
        "name": "small regression",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": [
            {
                "predict": {},
                "fit": {},
            },  # All of the following params should return same results because they're defaults
            {"predict": {}, "fit": {"presets": "medium_quality_faster_train"}},
            {"predict": {}, "fit": {"presets": "ignore_text"}},
            {"predict": {}, "fit": {"hyperparameters": "default"}},
            {"predict": {"eval_metric": "root_mean_squared_error"}, "fit": {}},
        ],
        "expected_score_range": {
            "CatBoost": (-7.86, 0.01),
            "ExtraTreesMSE": (-7.88, 0.01),
            "KNeighborsDist": (-8.69, 0.01),
            "KNeighborsUnif": (-9.06, 0.01),
            "LightGBM": (-15.55, 0.01),
            "LightGBMLarge": (-10.43, 0.01),
            "LightGBMXT": (-16.32, 0.01),
            "NeuralNetFastAI": (-6.12, 0.01),
            "NeuralNetTorch": (-4.96, 0.01),
            "RandomForestMSE": (-9.63, 0.01),
            "WeightedEnsemble_L2": (-5.66, 0.01),
            "XGBoost": (-10.8, 0.01),
        },
    },
    {  # If we explicitly exclude some models the others should return unchanged and the ensemble result will be changed.
        "name": "small regression excluded models",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"excluded_model_types": ["KNN", "RF", "XT", "GBM", "CAT", "XGB"]}},
        "expected_score_range": {
            "NeuralNetFastAI": (-6.12, 0.01),
            "NeuralNetTorch": (-4.96, 0.01),
            "WeightedEnsemble_L2": (-5.09, 0.01),
        },
    },
    {  # Small regression, hyperparameters = light removes some models
        "name": "small regression light hyperparameters",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"hyperparameters": "light"}},
        "expected_score_range": {
            "CatBoost": (-7.86, 0.01),
            "ExtraTreesMSE": (-7.87, 0.01),
            "LightGBM": (-15.55, 0.01),
            "LightGBMLarge": (-10.43, 0.01),
            "LightGBMXT": (-16.32, 0.01),
            "NeuralNetFastAI": (-6.12, 0.01),
            "NeuralNetTorch": (-4.96, 0.01),
            "RandomForestMSE": (-9.63, 0.01),
            "WeightedEnsemble_L2": (-5.66, 0.01),
            "XGBoost": (-10.8, 0.01),
        },
    },
    {  # Small regression, hyperparameters = very_light removes some models
        "name": "small regression very light hyperparameters",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"hyperparameters": "very_light"}},
        "expected_score_range": {
            "CatBoost": (-7.86, 0.01),
            "LightGBM": (-15.55, 0.01),
            "LightGBMLarge": (-10.43, 0.01),
            "LightGBMXT": (-16.32, 0.01),
            "NeuralNetFastAI": (-6.12, 0.01),
            "NeuralNetTorch": (-4.96, 0.01),
            "WeightedEnsemble_L2": (-5.58, 0.01),
            "XGBoost": (-10.8, 0.01),
        },
    },
    {  # Small regression, hyperparameters = toy removes almost all models and runs very fast
        "name": "small regression toy hyperparameters",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"hyperparameters": "toy"}},
        "expected_score_range": {
            "CatBoost": (-28.39, 0.01),
            "LightGBM": (-27.81, 0.01),
            "NeuralNetTorch": (-27.11, 0.01),
            "WeightedEnsemble_L2": (-19.12, 0.01),
            "XGBoost": (-19.12, 0.01),
        },
    },
    {  # High quality preset on small dataset.
        "name": "small regression high quality",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"presets": "high_quality_fast_inference_only_refit"}},
        "expected_score_range": {
            "CatBoost_BAG_L1": (np.nan, np.nan),
            "CatBoost_BAG_L1_FULL": (-7.75, 0.01),
            "ExtraTreesMSE_BAG_L1": (-7.52, 0.01),
            "ExtraTreesMSE_BAG_L1_FULL": (-7.52, 0.01),
            "KNeighborsDist_BAG_L1": (-8.21, 0.01),
            "KNeighborsDist_BAG_L1_FULL": (-8.21, 0.01),
            "KNeighborsUnif_BAG_L1": (-8.7, 0.01),
            "KNeighborsUnif_BAG_L1_FULL": (-8.7, 0.01),
            "LightGBMLarge_BAG_L1": (np.nan, np.nan),
            "LightGBMLarge_BAG_L1_FULL": (-9.94, 0.01),
            "LightGBMXT_BAG_L1": (np.nan, np.nan),
            "LightGBMXT_BAG_L1_FULL": (-13.03, 0.01),
            "LightGBM_BAG_L1": (np.nan, np.nan),
            "LightGBM_BAG_L1_FULL": (-14.17, 0.01),
            "NeuralNetFastAI_BAG_L1": (np.nan, np.nan),
            "NeuralNetFastAI_BAG_L1_FULL": (-5.48, 0.01),
            "NeuralNetTorch_BAG_L1": (np.nan, np.nan),
            "NeuralNetTorch_BAG_L1_FULL": (-5.29, 0.01),
            "RandomForestMSE_BAG_L1": (-9.5, 0.01),
            "RandomForestMSE_BAG_L1_FULL": (-9.5, 0.01),
            "WeightedEnsemble_L2": (np.nan, np.nan),
            "WeightedEnsemble_L2_FULL": (-5.29, 0.01),
            "XGBoost_BAG_L1": (np.nan, np.nan),
            "XGBoost_BAG_L1_FULL": (-9.76, 0.01),
        },
    },
    {  # Best quality preset on small dataset.
        "name": "small regression best quality",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {}, "fit": {"presets": "best_quality"}},
        "expected_score_range": {
            "CatBoost_BAG_L1": (-7.85, 0.01),
            "ExtraTreesMSE_BAG_L1": (-7.52, 0.01),
            "KNeighborsDist_BAG_L1": (-8.21, 0.01),
            "KNeighborsUnif_BAG_L1": (-8.70, 0.01),
            "LightGBMLarge_BAG_L1": (-9.44, 0.01),
            "LightGBMXT_BAG_L1": (-14.78, 0.01),
            "LightGBM_BAG_L1": (-14.92, 0.01),
            "NeuralNetFastAI_BAG_L1": (-5.55, 0.01),
            "NeuralNetTorch_BAG_L1": (-5.07, 0.01),
            "RandomForestMSE_BAG_L1": (-9.5, 0.01),
            "WeightedEnsemble_L2": (-5.05, 0.01),  # beats default, as expected
            "XGBoost_BAG_L1": (-9.74, 0.01),
        },
    },
    {  # Default regression model, add some categorical features.
        "name": "small regression with categorical",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 1,
        "dataset_hash": "3e26d128e0",
        "params": {"predict": {}, "fit": {}},  # Default params
        "expected_score_range": {
            "CatBoost": (-22.58, 0.01),
            "ExtraTreesMSE": (-25.09, 0.01),
            "KNeighborsDist": (-39.45, 0.01),
            "KNeighborsUnif": (-35.64, 0.01),
            "LightGBM": (-32.96, 0.01),
            "LightGBMLarge": (-34.86, 0.01),
            "LightGBMXT": (-32.69, 0.01),
            "NeuralNetFastAI": (-22.11, 0.01),
            "NeuralNetTorch": (-19.76, 0.01),
            "RandomForestMSE": (-27.49, 0.01),
            "WeightedEnsemble_L2": (-19.76, 0.01),
            "XGBoost": (-24.93, 0.01),
        },
    },
    {  # Default regression model different metric
        "name": "small regression metric mae",
        "type": "regression",
        "n_samples": 100,
        "n_features": 2,
        "n_categorical": 0,
        "dataset_hash": "5850a1c21a",
        "params": {"predict": {"eval_metric": "mean_absolute_error"}, "fit": {}},
        "expected_score_range": {
            "CatBoost": (-5.23, 0.01),
            "ExtraTreesMSE": (-5.48, 0.01),
            "KNeighborsDist": (-6.16, 0.01),
            "KNeighborsUnif": (-6.61, 0.01),
            "LightGBM": (-11.97, 0.01),
            "LightGBMLarge": (-7.69, 0.01),
            "LightGBMXT": (-12.37, 0.01),
            "NeuralNetFastAI": (-4.74, 0.01),
            "NeuralNetTorch": (-3.77, 0.01),
            "RandomForestMSE": (-6.96, 0.01),
            "WeightedEnsemble_L2": (-4.03, 0.01),
            "XGBoost": (-8.32, 0.01),
        },
    },
    #
    # Classifications
    #
    {  # Default classification model on a small dataset
        "name": "small classification",
        "type": "classification",
        "n_samples": 400,  # With only 8 classes it's hard to compare model quality unless we have a biggest test set (and therefore train set).
        "n_features": 10,
        "n_informative": 5,
        "n_classes": 8,
        "n_categorical": 0,
        "dataset_hash": "be1f16df80",
        "params": [
            {"predict": {}, "fit": {}},  # All of the following params should return same results
            {"predict": {}, "fit": {"presets": "medium_quality_faster_train"}},
            {"predict": {}, "fit": {"presets": "ignore_text"}},
            {"predict": {}, "fit": {"hyperparameters": "default"}},
            {"predict": {"eval_metric": "accuracy"}, "fit": {}},
        ],
        "expected_score_range": {
            "CatBoost": (0.245, 0.001),  # Classification scores are low numbers so we decrease the
            "ExtraTreesEntr": (0.327, 0.001),  # tolerance to 0.001 to make sure we pick up changes.
            "ExtraTreesGini": (0.32, 0.001),
            "KNeighborsDist": (0.337, 0.001),
            "KNeighborsUnif": (0.322, 0.001),
            "LightGBM": (0.197, 0.001),
            "LightGBMLarge": (0.265, 0.001),
            "LightGBMXT": (0.23, 0.001),
            "NeuralNetFastAI": (0.34, 0.001),
            "NeuralNetTorch": (0.232, 0.001),
            "RandomForestEntr": (0.305, 0.001),
            "RandomForestGini": (0.295, 0.001),
            "WeightedEnsemble_L2": (0.34, 0.001),
            "XGBoost": (0.227, 0.001),
        },
    },
    {  # There's different logic for boolean classification so let's test that with n_classes = 2.
        "name": "small classification boolean",
        "type": "classification",
        "n_samples": 400,
        "n_features": 10,
        "n_informative": 5,
        "n_classes": 2,
        "n_categorical": 0,
        "dataset_hash": "79e634aac3",
        "params": [
            {"predict": {}, "fit": {}},  # All of the following params should return same results
            {"predict": {"eval_metric": "accuracy"}, "fit": {}},
        ],
        "expected_score_range": {
            "CatBoost": (0.61, 0.001),
            "ExtraTreesEntr": (0.607, 0.001),
            "ExtraTreesGini": (0.6, 0.001),
            "KNeighborsDist": (0.61, 0.001),
            "KNeighborsUnif": (0.61, 0.001),
            "LightGBM": (0.632, 0.001),
            "LightGBMLarge": (0.552, 0.001),
            "LightGBMXT": (0.612, 0.001),
            "NeuralNetFastAI": (0.62, 0.001),
            "NeuralNetTorch": (0.597, 0.001),
            "RandomForestEntr": (0.607, 0.001),
            "RandomForestGini": (0.582, 0.001),
            "WeightedEnsemble_L2": (0.61, 0.001),
            "XGBoost": (0.58, 0.001),
        },
    },
]


def make_dataset(request, seed):
    TEST_SIZE = 0.5
    # Ensure our datasets and model calls remain deterministic.
    random.seed(seed)
    np.random.seed(seed)
    if request["type"] == "regression":
        x, y = make_regression(
            n_samples=int(request["n_samples"] * (1 / (1 - TEST_SIZE))), n_features=request["n_features"], noise=4
        )  # To make it hard enough that we get better performance on slower models
    elif request["type"] == "classification":
        x, y = make_classification(
            n_samples=int(request["n_samples"] * (1 / (1 - TEST_SIZE))),
            n_features=request["n_features"],
            n_informative=request["n_informative"],
            n_redundant=request["n_classes"] - request["n_informative"],
            n_classes=request["n_classes"],
            class_sep=0.4,
        )  # To make it hard enough that we get better performance on slower models
    else:
        assert False, "Unrecognised request type '{request['type'}'"

    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y, columns=["label"])

    # Make some columns categorical if required.
    if request["n_categorical"] > 0:
        cols_to_convert = random.sample(set(dfx.columns.values), k=request["n_categorical"])
        for col in cols_to_convert:
            dfx[col] = dfx[col].astype(int)
            vals = np.unique(dfx[col])
            # Shuffle the categoricals so there's no pattern in their ordering.
            vals2 = vals.copy() - min(vals)
            np.random.shuffle(vals2)
            mapper = dict(zip(vals, vals2))
            dfx[col] = dfx[col].map(mapper)
            dfx[col] = dfx[col].astype("category")

    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=TEST_SIZE)
    dftrain = pd.concat([x_train, y_train], axis=1)
    dftest = pd.concat([x_test, y_test], axis=1)

    return (dftrain, dftest)
