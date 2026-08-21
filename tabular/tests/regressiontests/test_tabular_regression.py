"""Runs autogluon.tabular on synthetic classification and regression datasets.
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

Testing by @willsmithorg on master AG as of 2022-02-22 - 2022-02-23:
Tested on AWS Linux instance m5.2xlarge, amzn2-ami-kernel-5.10-hvm-2.0.20211223.0-x86_64-gp2 with
                                           (8  vcore, no GPU, Python==3.7.10, scikit-learn==1.0.2, torch==1.10.2),
Tested on Github jenkins Linux:
                                           (?  vcore,  0 GPU, Python==3.9.10, scikit-learn==1.0.2, torch==1.10.2),
Tested on AWS Windows instance t3.xlarge,
                                           (4  vcore,  0 GPU, Python==3.9.7 , scikit-learn==1.0.2, torch==1.10.2),
                                           - Pytorch scores are slightly different, all else same.

"""

import hashlib
import math
import sys

import numpy as np
import pytest

from autogluon.tabular import TabularDataset, TabularPredictor

from .utils import make_dataset, tests


# Lots of test data since inference is fast and we want a score that's very reflective of model quality,
# despite very fast training times.
# Round to given accuracy.  The 8 is to remove floating point rounding errors.
def myfloor(x, base=0.01):
    return round(base * math.floor(float(x) / base), 8)


@pytest.mark.regression
def inner_test_tabular(testname):
    # Find the named test
    test = None
    for t in tests:
        if t["name"] == testname:
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
    assert current_hash == test["dataset_hash"], (
        f"Test '{testname}' input dataset has changed.  All scores will change.\n" + proposedconfig
    )

    # Now run the Predictor 1 or more times with various parameters, and make sure we get
    # back the expected results.

    # Params can either omitted, or a single run, or a list of runs.
    if "params" not in test:
        test["params"] = {"predict": {}, "fit": {}}
    if not isinstance(test["params"], list):
        test["params"] = [test["params"]]
    for params in test["params"]:
        # Run this model and set of params
        predictor = TabularPredictor(label="label", **params["predict"])
        predictor.fit(dftrain, **params["fit"])
        leaderboard = predictor.leaderboard(dftest)
        leaderboard = leaderboard.sort_values(by="model")  # So we can pre-generate sample config in alphabetical order

        # Store proposed new config based on the current run, in case the developer wants to keep thee results (just cut and paste).
        proposedconfig = "Proposed new config:\n"
        proposedconfig += "'expected_score_range' : {\n"
        for model in leaderboard["model"]:
            midx_in_leaderboard = leaderboard.index.values[leaderboard["model"] == model][0]
            if np.isnan(leaderboard["score_test"][midx_in_leaderboard]):
                values = "np.nan, np.nan"
            else:
                if model in test["expected_score_range"] and not np.isnan(test["expected_score_range"][model][1]):
                    currentprecision = test["expected_score_range"][model][1]
                else:
                    currentprecision = 0.01
                values = "{}, {}".format(
                    myfloor(leaderboard["score_test"][midx_in_leaderboard], currentprecision), currentprecision
                )
            proposedconfig += f"    '{model}': ({values}),\n"
        proposedconfig += "},\n"

        # First validate the model list was as expected.
        assert set(leaderboard["model"]) == set(test["expected_score_range"].keys()), (
            f"Test '{testname}' params {params} got unexpected model list.\n" + proposedconfig
        )

        # Now validate the scores for each model were as expected.
        all_assertions_met = True
        currentconfig = "Existing config:\n"
        currentconfig += "'expected_score_range' : {\n"
        for model in sorted(test["expected_score_range"]):
            midx_in_leaderboard = leaderboard.index.values[leaderboard["model"] == model][0]
            assert leaderboard["model"][midx_in_leaderboard] == model
            expectedrange = test["expected_score_range"][model][1]
            expectedmin = test["expected_score_range"][model][0]
            expectedmax = expectedmin + expectedrange

            if np.isnan(expectedmin):
                values = "np.nan, np.nan"
            else:
                values = "{}, {}".format(expectedmin, expectedrange)

            if (
                (leaderboard["score_test"][midx_in_leaderboard] >= expectedmin)
                and (leaderboard["score_test"][midx_in_leaderboard] <= expectedmax)
            ) or (np.isnan(leaderboard["score_test"][midx_in_leaderboard]) and np.isnan(expectedmin)):
                currentconfig += f"    '{model}': ({values}),\n"
            else:
                currentconfig += f"    '{model}': ({values}), # <--- not met, got {leaderboard['score_test'][midx_in_leaderboard]} \n"
                all_assertions_met = False
        currentconfig += "},\n"

        assert all_assertions_met, (
            f"Test '{testname}', params {params} had unexpected scores:\n" + currentconfig + proposedconfig
        )

        # Clean up this model created with specific params.
        predictor.delete_models(models_to_keep=[], dry_run=False)


# The tests are all run individually rather than in 1 big loop that simply goes through the tests dictionary.
# This is so we easily remove some tests if necessary.
@pytest.mark.parametrize(
    "testname",
    [
        "small regression",
        "small regression excluded models",
        "small regression light hyperparameters",
        "small regression very light hyperparameters",
        "small regression toy hyperparameters",
        "small regression high quality",
        "small regression best quality",
        "small regression with categorical",
        "small regression metric mae",
        "small classification",
        "small classification boolean",
    ],
)
# These results have only been confirmed for Linux.  Windows is known to give different results for Pytorch.
@pytest.mark.skipif(sys.platform != "linux", reason="Scores only confirmed on Linux")
@pytest.mark.regression
def test_tabular_score(testname):
    inner_test_tabular(testname)
