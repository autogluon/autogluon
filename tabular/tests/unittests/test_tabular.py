"""Runs autogluon.tabular on multiple benchmark datasets.
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
import sys
import tempfile
import warnings
from random import seed

import numpy as np
import pandas as pd
import pytest

from autogluon.common import space
from autogluon.common.utils.simulation_utils import convert_simulation_artifacts_to_tabular_predictions_dict
from autogluon.core.constants import BINARY, MULTICLASS, PROBLEM_TYPES_CLASSIFICATION, QUANTILE, REGRESSION
from autogluon.core.utils import download, unzip
from autogluon.tabular import TabularDataset, TabularPredictor, __version__
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

PARALLEL_LOCAL_BAGGING = "parallel_local"
SEQUENTIAL_LOCAL_BAGGING = "sequential_local"
on_windows = os.name == "nt"


def test_tabular():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 0  # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = None
    verbosity = 2  # how much output to print
    hyperparameters = get_hyperparameter_config("default")
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        time_limit = 60

    # Catboost > 1.2 is required for python 3.11 but cannot be correctly installed on macos
    if sys.version_info >= (3, 11) and sys.platform == "darwin":
        hyperparameters.pop("CAT")

    fit_args = {"verbosity": verbosity}
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)
    run_tabular_benchmark_toy(fit_args=fit_args)


def _assert_predict_dict_identical_to_predict(predictor: TabularPredictor, data):
    """Assert that predict_multi is identical to looping calls to predict"""
    for as_pandas in [True, False]:
        for inverse_transform in [True, False]:
            predict_dict = predictor.predict_multi(data=data, as_pandas=as_pandas, inverse_transform=inverse_transform)
            assert set(predictor.model_names()) == set(predict_dict.keys())
            for m in predictor.model_names():
                if not inverse_transform:
                    model_pred = predictor._learner.predict(data, model=m, as_pandas=as_pandas, inverse_transform=inverse_transform)
                else:
                    model_pred = predictor.predict(data, model=m, as_pandas=as_pandas)
                if as_pandas:
                    # pandas default int type on Windows is int64, while on Linux it is int32
                    if model_pred.dtype in ["int64", "int32"]:
                        assert predict_dict[m].dtype in ["int64", "int32"]
                        assert model_pred.astype("int64").equals(predict_dict[m].astype("int64"))
                    else:
                        assert model_pred.equals(predict_dict[m])
                else:
                    assert np.array_equal(model_pred, predict_dict[m])


def _assert_predict_proba_dict_identical_to_predict_proba(predictor: TabularPredictor, data):
    """Assert that predict_proba_multi is identical to looping calls to predict_proba"""
    for as_pandas in [True, False]:
        for inverse_transform in [True, False]:
            for as_multiclass in [True, False]:
                predict_proba_dict = predictor.predict_proba_multi(
                    data=data, as_pandas=as_pandas, as_multiclass=as_multiclass, inverse_transform=inverse_transform
                )
                assert set(predictor.model_names()) == set(predict_proba_dict.keys())
                for m in predictor.model_names():
                    if not inverse_transform:
                        model_pred_proba = predictor._learner.predict_proba(
                            data, model=m, as_pandas=as_pandas, as_multiclass=as_multiclass, inverse_transform=inverse_transform
                        )
                    else:
                        model_pred_proba = predictor.predict_proba(data, model=m, as_pandas=as_pandas, as_multiclass=as_multiclass)
                    if as_pandas:
                        assert model_pred_proba.equals(predict_proba_dict[m])
                    else:
                        assert np.array_equal(model_pred_proba, predict_proba_dict[m])


def test_advanced_functionality():
    """
    Tests a bunch of advanced functionality, including when used in combination.
    The idea is that if this test passes, we are in good shape.
    Simpler to test all of this within one test as it avoids repeating redundant setup such as fitting a predictor.
    """
    fast_benchmark = True
    dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip",
        "name": "AdultIncomeBinaryClassification",
        "problem_type": BINARY,
    }
    label = "class"
    directory_prefix = "./datasets/"
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"])
    if fast_benchmark:  # subsample for fast_benchmark
        subsample_size = 100
        train_data = train_data.head(subsample_size)
        test_data = test_data.head(subsample_size)
    print(f"Evaluating Advanced Functionality on Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + "advanced/" + dataset["name"] + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    savedir_predictor_original = savedir + "predictor/"
    predictor: TabularPredictor = TabularPredictor(label=label, path=savedir_predictor_original).fit(train_data)

    version_in_file = predictor._load_version_file(path=predictor.path)
    assert version_in_file == __version__

    leaderboard = predictor.leaderboard(data=test_data)

    # test metric_error leaderboard
    leaderboard_error = predictor.leaderboard(data=test_data, score_format="error")
    assert sorted(leaderboard["model"].to_list()) == sorted(leaderboard_error["model"].to_list())
    leaderboard_combined = pd.merge(leaderboard, leaderboard_error[["model", "metric_error_test", "metric_error_val"]], on=["model"])
    score_test = leaderboard_combined["score_test"].to_list()
    score_val = leaderboard_combined["score_val"].to_list()
    metric_error_test = leaderboard_combined["metric_error_test"].to_list()
    metric_error_val = leaderboard_combined["metric_error_val"].to_list()
    for score, error in zip(score_test, metric_error_test):
        assert predictor.eval_metric.convert_score_to_error(score) == error
    for score, error in zip(score_val, metric_error_val):
        assert predictor.eval_metric.convert_score_to_error(score) == error

    if not on_windows:
        predictor.plot_ensemble_model()

    # Test get_simulation_artifact
    simulation_artifact_no_test = predictor.simulation_artifact()
    assert "pred_proba_dict_test" not in simulation_artifact_no_test
    assert "y_test" not in simulation_artifact_no_test
    simulation_artifact = predictor.simulation_artifact(test_data=test_data)
    assert sorted(list(simulation_artifact["pred_proba_dict_test"].keys())) == sorted(predictor.model_names(can_infer=True))
    assert simulation_artifact["y_test"].equals(predictor.transform_labels(test_data[label]))
    for sim_artifact in [simulation_artifact, simulation_artifact_no_test]:
        assert sim_artifact["label"] == predictor.label
        assert sorted(list(sim_artifact["pred_proba_dict_val"].keys())) == sorted(predictor.model_names(can_infer=True))
        assert sim_artifact["eval_metric"] == predictor.eval_metric.name
        assert sim_artifact["problem_type"] == predictor.problem_type
    simulation_artifacts = {dataset["name"]: {0: simulation_artifact}}

    # Test convert_simulation_artifacts_to_tabular_predictions_dict
    aggregated_pred_proba, aggregated_ground_truth = convert_simulation_artifacts_to_tabular_predictions_dict(simulation_artifacts=simulation_artifacts)
    assert set(aggregated_pred_proba[dataset["name"]][0]["pred_proba_dict_val"].keys()) == set(predictor.model_names(can_infer=True))
    assert set(aggregated_pred_proba[dataset["name"]][0]["pred_proba_dict_test"].keys()) == set(predictor.model_names(can_infer=True))
    ground_truth_keys_expected = set(simulation_artifact.keys())
    ground_truth_keys_expected.remove("pred_proba_dict_val")
    ground_truth_keys_expected.remove("pred_proba_dict_test")
    assert set(aggregated_ground_truth[dataset["name"]][0].keys()) == ground_truth_keys_expected

    extra_metrics = ["accuracy", "roc_auc", "log_loss"]
    test_data_no_label = test_data.drop(columns=[label])
    with pytest.raises(ValueError):
        # Error because skip_score is False and label not present
        predictor.leaderboard(data=test_data_no_label)
    with pytest.raises(ValueError):
        # Error because extra_metrics != None and label not present
        predictor.leaderboard(data=test_data_no_label, skip_score=True, extra_metrics=extra_metrics)
    # Valid because skip_score=True
    leaderboard_no_score = predictor.leaderboard(data=test_data.drop(columns=[label]), skip_score=True)
    assert len(leaderboard) == len(leaderboard_no_score)
    assert "pred_time_test" in leaderboard_no_score
    assert "pred_time_test_marginal" in leaderboard_no_score
    assert "score_test" in leaderboard_no_score
    for i in range(len(leaderboard_no_score)):
        # Assert that score_test is NaN for all models
        assert leaderboard_no_score["score_test"].isnull().iloc[i]
    leaderboard_extra = predictor.leaderboard(data=test_data, extra_info=True, extra_metrics=extra_metrics)
    _assert_predict_dict_identical_to_predict(predictor=predictor, data=test_data)
    _assert_predict_proba_dict_identical_to_predict_proba(predictor=predictor, data=test_data)
    assert set(predictor.model_names()) == set(leaderboard["model"])
    assert set(predictor.model_names()) == set(leaderboard_extra["model"])
    assert set(leaderboard_extra.columns).issuperset(set(leaderboard.columns))
    assert len(leaderboard) == len(leaderboard_extra)
    assert set(leaderboard_extra.columns).issuperset(set(extra_metrics))  # Assert that extra_metrics are present in output
    num_models = len(predictor.model_names())
    feature_importances = predictor.feature_importance(data=test_data)
    original_features = set(train_data.columns)
    original_features.remove(label)
    assert set(feature_importances.index) == original_features
    assert set(feature_importances.columns) == {"importance", "stddev", "p_value", "n", "p99_high", "p99_low"}
    predictor.transform_features()
    test_data_transformed = predictor.transform_features(data=test_data)
    predictor.info()

    # Assert that transform_features=False works correctly
    y_pred = predictor.predict(test_data)
    y_pred_from_transform = predictor.predict(test_data_transformed, transform_features=False)
    assert y_pred.equals(y_pred_from_transform)

    y_pred_proba = None
    if predictor.can_predict_proba:
        y_pred_proba = predictor.predict_proba(test_data)
        y_pred_proba_from_transform = predictor.predict_proba(test_data_transformed, transform_features=False)
        assert y_pred_proba.equals(y_pred_proba_from_transform)

    assert predictor.model_names(persisted=True) == []  # Assert that no models were persisted during training
    assert predictor.unpersist() == []  # Assert that no models were unpersisted

    persisted_models = predictor.persist(models="all", max_memory=None)
    assert set(predictor.model_names(persisted=True)) == set(persisted_models)  # Ensure all models are persisted
    assert predictor.persist(models="all", max_memory=None) == []  # Ensure that no additional models are persisted on repeated calls
    unpersised_models = predictor.unpersist()
    assert set(unpersised_models) == set(persisted_models)
    assert predictor.model_names(persisted=True) == []  # Assert that all models were unpersisted

    # Raise exception
    with pytest.raises(ValueError):
        predictor.persist(models=["UNKNOWN_MODEL_1", "UNKNOWN_MODEL_2"])

    assert predictor.model_names(persisted=True) == []

    assert predictor.unpersist(models=["UNKNOWN_MODEL_1", "UNKNOWN_MODEL_2"]) == []

    predictor.persist(models="all", max_memory=None)
    predictor.save()  # Save predictor while models are persisted: Intended functionality is that they won't be persisted when loaded.
    predictor_loaded = TabularPredictor.load(predictor.path)  # Assert that predictor loading works
    leaderboard_loaded = predictor_loaded.leaderboard(data=test_data)
    assert len(leaderboard) == len(leaderboard_loaded)
    assert predictor_loaded.model_names(persisted=True) == []  # Assert that models were not still persisted after loading predictor

    _assert_predictor_size(predictor=predictor)
    # Test cloning logic
    with pytest.raises(AssertionError):
        # Ensure don't overwrite existing predictor
        predictor.clone(path=predictor.path)
    path_clone = predictor.clone(path=predictor.path + "_clone")
    predictor_clone = TabularPredictor.load(path_clone)
    assert predictor.path != predictor_clone.path
    if predictor_clone.can_predict_proba:
        y_pred_proba_clone = predictor_clone.predict_proba(test_data)
        assert y_pred_proba.equals(y_pred_proba_clone)
    y_pred_clone = predictor_clone.predict(test_data)
    assert y_pred.equals(y_pred_clone)
    leaderboard_clone = predictor_clone.leaderboard(data=test_data)
    assert len(leaderboard) == len(leaderboard_clone)

    # Test cloning for deployment logic
    path_clone_for_deployment_og = predictor.path + "_clone_for_deployment"
    with pytest.raises(FileNotFoundError):
        # Assert that predictor does not exist originally
        TabularPredictor.load(path_clone_for_deployment_og)
    path_clone_for_deployment = predictor.clone_for_deployment(path=path_clone_for_deployment_og)
    assert path_clone_for_deployment == path_clone_for_deployment_og
    predictor_clone_for_deployment = TabularPredictor.load(path_clone_for_deployment)
    assert predictor.path != predictor_clone_for_deployment.path
    if predictor_clone_for_deployment.can_predict_proba:
        y_pred_proba_clone_for_deployment = predictor_clone_for_deployment.predict_proba(test_data)
        assert y_pred_proba.equals(y_pred_proba_clone_for_deployment)
    y_pred_clone_for_deployment = predictor_clone_for_deployment.predict(test_data)
    assert y_pred.equals(y_pred_clone_for_deployment)
    leaderboard_clone_for_deployment = predictor_clone_for_deployment.leaderboard(data=test_data)
    assert len(leaderboard) >= len(leaderboard_clone_for_deployment)
    # Raise exception due to lacking fit artifacts
    with pytest.raises(FileNotFoundError):
        predictor_clone_for_deployment.refit_full()

    assert predictor.model_refit_map() == dict()
    predictor.refit_full()
    if not on_windows:
        predictor.plot_ensemble_model()
    assert len(predictor.model_refit_map()) == num_models
    assert len(predictor.model_names()) == num_models * 2
    for model in predictor.model_names():
        predictor.predict(data=test_data, model=model)
    predictor.refit_full()  # Confirm that refit_models aren't further refit.
    assert len(predictor.model_refit_map()) == num_models
    assert len(predictor.model_names()) == num_models * 2
    predictor.delete_models(models_to_keep=[])  # Test that dry-run doesn't delete models
    assert len(predictor.model_names()) == num_models * 2
    predictor.predict(data=test_data)

    # Test refit_full with train_data_extra argument
    refit_full_models = list(predictor.model_refit_map().values())
    predictor.delete_models(models_to_delete=refit_full_models, dry_run=False)
    assert len(predictor.model_names()) == num_models
    assert predictor.model_refit_map() == dict()
    predictor.refit_full(train_data_extra=test_data)  # train_data_extra argument
    assert len(predictor.model_names()) == num_models * 2
    assert len(predictor.model_refit_map()) == num_models
    predictor.predict(data=test_data)

    predictor.delete_models(models_to_keep=[], dry_run=False)  # Test that dry_run=False deletes models
    assert len(predictor.model_names()) == 0
    assert len(predictor.leaderboard()) == 0
    assert len(predictor.leaderboard(extra_info=True)) == 0
    # Assert that predictor can no longer predict
    try:
        predictor.predict(data=test_data)
    except:
        pass
    else:
        raise AssertionError("predictor.predict should raise exception after all models are deleted")
    # Assert that clone is not impacted by changes to original
    assert len(predictor_clone.leaderboard(data=test_data)) == len(leaderboard_clone)
    if predictor_clone.can_predict_proba:
        y_pred_proba_clone_2 = predictor_clone.predict_proba(data=test_data)
        assert y_pred_proba.equals(y_pred_proba_clone_2)
    y_pred_clone_2 = predictor_clone.predict(data=test_data)
    assert y_pred.equals(y_pred_clone_2)
    print("Tabular Advanced Functionality Test Succeeded.")


def _assert_predictor_size(predictor: TabularPredictor):
    predictor_size_disk = predictor.disk_usage()
    predictor_size_disk_per_file = predictor.disk_usage_per_file()
    assert predictor_size_disk > 0  # Assert that .disk_usage() produces a >0 result and doesn't crash
    assert len(predictor_size_disk_per_file) > 0
    assert predictor_size_disk == predictor_size_disk_per_file.sum()


def test_advanced_functionality_bagging():
    fast_benchmark = True
    dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip",
        "name": "AdultIncomeBinaryClassification",
        "problem_type": BINARY,
    }
    label = "class"
    directory_prefix = "./datasets/"
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"])
    if fast_benchmark:  # subsample for fast_benchmark
        subsample_size = 500
        train_data = train_data.head(subsample_size)
        test_data = test_data.head(subsample_size)
    print(f"Evaluating Advanced Functionality (Bagging) on Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + "advanced/" + dataset["name"] + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    predictor = TabularPredictor(label=label, path=savedir).fit(
        train_data,
        num_bag_folds=2,
        hyperparameters={"GBM": {}},
    )

    expected_num_models = 2
    assert len(predictor.model_names()) == expected_num_models

    _assert_predict_dict_identical_to_predict(predictor=predictor, data=test_data)
    _assert_predict_proba_dict_identical_to_predict_proba(predictor=predictor, data=test_data)

    oof_pred_proba = predictor.predict_proba_oof()
    assert len(oof_pred_proba) == len(train_data)

    predict_proba_dict_oof = predictor.predict_proba_multi()
    for m in predictor.model_names():
        predict_proba_oof = predictor.predict_proba_oof(model=m)
        assert predict_proba_oof.equals(predict_proba_dict_oof[m])

    predict_dict_oof = predictor.predict_multi()
    for m in predictor.model_names():
        predict_oof = predictor.predict_oof(model=m)
        assert predict_oof.equals(predict_dict_oof[m])

    score_oof = predictor.evaluate_predictions(train_data[label], oof_pred_proba)
    model_best = predictor.model_best

    predictor.refit_full()
    assert len(predictor.model_refit_map()) == expected_num_models
    assert len(predictor.model_names()) == expected_num_models * 2

    model_best_refit = predictor.model_best
    assert model_best != model_best_refit

    # assert that refit model uses original model's OOF predictions
    oof_pred_proba_refit = predictor.predict_proba_oof()
    assert oof_pred_proba.equals(oof_pred_proba_refit)

    # check predict_proba_multi after refit does not raise an exception
    predict_proba_dict_oof = predictor.predict_proba_multi()
    for m in predictor.model_names():
        predict_proba_oof = predictor.predict_proba_oof(model=m)
        assert predict_proba_oof.equals(predict_proba_dict_oof[m])

    # check predict_multi after refit does not raise an exception
    predict_dict_oof = predictor.predict_multi()
    for m in predictor.model_names():
        predict_oof = predictor.predict_oof(model=m)
        assert predict_oof.equals(predict_dict_oof[m])


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
    dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/toyClassification.zip",
        "name": "toyClassification",
        "problem_type": MULTICLASS,
        "label": "y",
        "performance_val": 0.436,
    }
    # 2-D toy noisy, imbalanced 4-class classification task with: feature missingness, out-of-vocabulary feature categories in test data, out-of-vocabulary labels in test data, training column missing from test data, extra distraction columns in test data
    # toyclassif_dataset should produce 1 warning and 1 error during inference:
    # Warning: Ignoring 181 (out of 1000) training examples for which the label value in column 'y' is missing
    # ValueError: Required columns are missing from the provided dataset. Missing columns: ['lostcolumn']

    # Additional warning that would have occurred if ValueError was not triggered:
    # UserWarning: These columns from this dataset were not present in the training dataset (AutoGluon will ignore them):  ['distractioncolumn1', 'distractioncolumn2']

    directory_prefix = "./datasets/"
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + dataset["name"] + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    predictor = TabularPredictor(label=dataset["label"], path=savedir).fit(train_data, **fit_args)
    assert len(predictor._trainer._models_failed_to_train_errors.keys()) == 0
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
    binary_dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip",
        "name": "AdultIncomeBinaryClassification",
        "problem_type": BINARY,
        "label": "class",
        "performance_val": 0.129,
    }  # Mixed types of features.

    multi_dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip",
        "name": "CoverTypeMulticlassClassification",
        "problem_type": MULTICLASS,
        "label": "Cover_Type",
        "performance_val": 0.032,
    }  # big dataset with 7 classes, all features are numeric. Runs SLOW.

    regression_dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip",
        "name": "AmesHousingPriceRegression",
        "problem_type": REGRESSION,
        "label": "SalePrice",
        "performance_val": 0.076,
    }  # Regression with mixed feature-types, skewed Y-values.

    toyregres_dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip",
        "name": "toyRegression",
        "problem_type": REGRESSION,
        "label": "y",
        "performance_val": 0.183,
    }
    # 1-D toy deterministic regression task with: heavy label+feature missingness, extra distraction column in test data

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    return [toyregres_dataset, binary_dataset, regression_dataset, multi_dataset]


def run_tabular_benchmarks(fast_benchmark, subsample_size, perf_threshold, seed_val, fit_args, dataset_indices=None, run_distill=False, crash_in_oof=False):
    print("Running fit with args:")
    print(fit_args)
    # Each train/test dataset must be located in single directory with the given names.
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    EPS = 1e-10

    # List containing dicts for each dataset to include in benchmark (try to order based on runtimes)
    datasets = get_benchmark_sets()
    if dataset_indices is not None:  # only run some datasets
        datasets = [datasets[i] for i in dataset_indices]

    # Aggregate performance summaries obtained in previous benchmark run:
    prev_perf_vals = [dataset["performance_val"] for dataset in datasets]
    previous_avg_performance = np.mean(prev_perf_vals)
    previous_median_performance = np.median(prev_perf_vals)
    previous_worst_performance = np.max(prev_perf_vals)

    # Run benchmark:
    performance_vals = [0.0] * len(datasets)  # performance obtained in this run
    directory_prefix = "./datasets/"
    with warnings.catch_warnings(record=True) as caught_warnings:
        for idx in range(len(datasets)):
            dataset = datasets[idx]
            train_data, test_data = load_data(
                directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"]
            )
            if seed_val is not None:
                seed(seed_val)
                np.random.seed(seed_val)
            print("Evaluating Benchmark Dataset %s (%d of %d)" % (dataset["name"], idx + 1, len(datasets)))
            directory = directory_prefix + dataset["name"] + "/"
            savedir = directory + "AutogluonOutput/"
            shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
            label = dataset["label"]
            y_test = test_data[label]
            test_data = test_data.drop(labels=[label], axis=1)
            if fast_benchmark:
                if subsample_size is None:
                    raise ValueError("fast_benchmark specified without subsample_size")
                if subsample_size < len(train_data):
                    # .sample instead of .head to increase diversity and test cases where data index is not monotonically increasing.
                    train_data = train_data.sample(n=subsample_size, random_state=seed_val)  # subsample for fast_benchmark
            predictor = TabularPredictor(label=label, path=savedir).fit(train_data, **fit_args)
            assert len(predictor._trainer._models_failed_to_train_errors.keys()) == 0
            results = predictor.fit_summary(verbosity=4)
            original_features = list(train_data)
            original_features.remove(label)
            assert original_features == predictor.original_features
            if predictor.problem_type != dataset["problem_type"]:
                warnings.warn(
                    "For dataset %s: Autogluon inferred problem_type = %s, but should = %s" % (dataset["name"], predictor.problem_type, dataset["problem_type"])
                )
            predictor = TabularPredictor.load(savedir)  # Test loading previously-trained predictor from file
            y_pred_empty = predictor.predict(test_data[0:0])
            assert len(y_pred_empty) == 0
            y_pred = predictor.predict(test_data)
            perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            if dataset["problem_type"] != REGRESSION:
                perf = 1.0 - perf_dict["accuracy"]  # convert accuracy to error-rate
            else:
                perf = 1.0 - perf_dict["r2"]  # unexplained variance score.
            performance_vals[idx] = perf
            print("Performance on dataset %s: %s   (previous perf=%s)" % (dataset["name"], performance_vals[idx], dataset["performance_val"]))
            if (not fast_benchmark) and (performance_vals[idx] > dataset["performance_val"] * perf_threshold):
                warnings.warn(
                    "Performance on dataset %s is %s times worse than previous performance."
                    % (dataset["name"], performance_vals[idx] / (EPS + dataset["performance_val"]))
                )
            if predictor._trainer.bagged_mode and not crash_in_oof:
                # TODO: Test index alignment with original training data (first handle duplicated rows / dropped rows edge cases)
                y_pred_oof = predictor.predict_oof()
                y_pred_proba_oof = predictor.predict_proba_oof(as_multiclass=False)
                y_pred_oof_transformed = predictor.predict_oof(transformed=True)
                y_pred_proba_oof_transformed = predictor.predict_proba_oof(as_multiclass=False, transformed=True)

                # Assert expected type output
                assert isinstance(y_pred_oof, pd.Series)
                assert isinstance(y_pred_oof_transformed, pd.Series)
                if predictor.problem_type == MULTICLASS:
                    assert isinstance(y_pred_proba_oof, pd.DataFrame)
                    assert isinstance(y_pred_proba_oof_transformed, pd.DataFrame)
                else:
                    if predictor.problem_type == BINARY:
                        assert isinstance(predictor.predict_proba_oof(), pd.DataFrame)
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
                    predictor.predict_oof()
                with pytest.raises(AssertionError):
                    predictor.predict_proba_oof()
            if run_distill:
                predictor.distill(time_limit=60, augment_args={"size_factor": 0.5})

    # Summarize:
    avg_perf = np.mean(performance_vals)
    median_perf = np.median(performance_vals)
    worst_perf = np.max(performance_vals)
    for idx in range(len(datasets)):
        print("Performance on dataset %s: %s   (previous perf=%s)" % (datasets[idx]["name"], performance_vals[idx], datasets[idx]["performance_val"]))

    print("Average performance: %s" % avg_perf)
    print("Median performance: %s" % median_perf)
    print("Worst performance: %s" % worst_perf)

    if not fast_benchmark:
        if avg_perf > previous_avg_performance * perf_threshold:
            warnings.warn("Average Performance is %s times worse than previously." % (avg_perf / (EPS + previous_avg_performance)))
        if median_perf > previous_median_performance * perf_threshold:
            warnings.warn("Median Performance is %s times worse than previously." % (median_perf / (EPS + previous_median_performance)))
        if worst_perf > previous_worst_performance * perf_threshold:
            warnings.warn("Worst Performance is %s times worse than previously." % (worst_perf / (EPS + previous_worst_performance)))

    print("Ran fit with args:")
    print(fit_args)
    # List all warnings again to make sure they are seen:
    print("\n\n WARNINGS:")
    for w in caught_warnings:
        warnings.warn(w.message)


def test_pseudolabeling():
    datasets = get_benchmark_sets()
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    directory_prefix = "./datasets/"
    hyperparam_setting = {
        "GBM": {"num_boost_round": 10},
        "XGB": {"n_estimators": 10},
    }

    fit_args = dict(
        hyperparameters=hyperparam_setting,
        time_limit=20,
    )

    fit_args_best = dict(
        presets="best_quality",
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble=dict(fold_fitting_strategy="sequential_local"),
        dynamic_stacking=False,
    )
    for idx in range(len(datasets)):
        dataset = datasets[idx]
        label = dataset["label"]
        problem_type = dataset["problem_type"]
        name = dataset["name"]
        train_data, test_data = load_data(
            directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"]
        )

        print(f"Testing dataset with name: {name}, problem type: {problem_type}")

        train_data = train_data.sample(50, random_state=1)
        test_data = test_data[test_data[label].notna()]

        if problem_type in PROBLEM_TYPES_CLASSIFICATION:
            valid_class_idxes = test_data[label].isin(train_data[label].unique())
            test_data = test_data[valid_class_idxes]

        test_data = test_data.sample(50, random_state=1)

        error_msg_og = (
            f"pseudolabel threw an exception during fit, it should have "
            f"succeeded on problem type:{problem_type} with dataset name:{name}, "
            f"with problem_type: {problem_type}. Under settings:"
        )

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
            assert False, error_msg_og + "labeled test data"

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
            assert False, error_msg_og + "labeled test data, best quality"

        # Test unlabeled pseudo data
        unlabeled_test_data = test_data.drop(columns=label)
        for flag_ensemble in [True, False]:
            error_prefix = "ensemble " if flag_ensemble else ""
            error_msg = error_prefix + error_msg_og
            for is_weighted_ensemble in [True, False]:
                error_suffix = " with pseudo label model weighted ensembling" if is_weighted_ensemble else ""

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
                    assert False, error_msg + "unlabeled test data" + error_suffix

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
                    assert False, error_msg + "unlabeled test data, best quality" + error_suffix


@pytest.mark.slow
def test_tabularHPObagstack():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 10000  # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "auto"}
    num_stack_levels = 2
    num_bag_folds = 2
    verbosity = 2  # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        nn_options = {"num_epochs": 2, "learning_rate": space.Real(0.001, 0.01)}
        gbm_options = {"num_boost_round": 20, "learning_rate": space.Real(0.01, 0.1)}
        hyperparameters = {"GBM": gbm_options, "NN_TORCH": nn_options}
        time_limit = 50

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_stack_levels": num_stack_levels,
        "verbosity": verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
        fit_args["num_bag_sets"] = 2
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)


def test_tabularHPO():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 99  # random seed
    subsample_size = None
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "auto"}
    verbosity = 2  # how much output to print
    hyperparameters = None
    time_limit = None
    fast_benchmark = True  # False
    # If True, run a faster benchmark (subsample training sets, less epochs, etc),
    # otherwise we run full benchmark with default AutoGluon settings.
    # performance_value warnings are disabled when fast_benchmark = True.

    #### If fast_benchmark = True, can control model training time here. Only used if fast_benchmark=True ####
    if fast_benchmark:
        subsample_size = 100
        time_limit = 240
        hyperparameter_tune_kwargs["num_trials"] = 3

    fit_args = {
        "verbosity": verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)


@pytest.mark.slow
def test_tabular_feature_prune():
    ############ Benchmark options you can set: ########################
    perf_threshold = 1.1  # How much worse can performance on each dataset be vs previous performance without warning
    seed_val = 99  # random seed
    subsample_size = None
    ag_args = {
        "feature_prune_kwargs": {
            "stop_threshold": 3,
            "prune_ratio": 0.05,
            "prune_threshold": None,
            "n_train_subsample": 1000,
            "n_fi_subsample": 5000,
            "min_fi_samples": 5000,
            "feature_prune_time_limit": 10,
            "raise_exception": True,
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
        gbm_options = {"num_boost_round": 20}
        hyperparameters = {"GBM": gbm_options}
        time_limit = 60

    fit_args = {
        "verbosity": verbosity,
    }
    fit_args["ag_args"] = ag_args
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    ###################################################################
    run_tabular_benchmarks(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)


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
        nn_options = {"num_epochs": 1}
        gbm_options = {"num_boost_round": 30}
        hyperparameters = {"GBM": gbm_options, "NN_TORCH": nn_options}
        time_limit = 60

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_stack_levels": num_stack_levels,
        "verbosity": verbosity,
        "ag_args_ensemble": {
            "fold_fitting_strategy": fold_fitting_strategy,
        },
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
        fit_args["num_bag_sets"] = 2
    ###################################################################
    config = dict(fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args)
    return config


def test_tabular_parallel_local_bagging():
    config = _construct_tabular_bag_test_config(PARALLEL_LOCAL_BAGGING)
    run_tabular_benchmarks(**config)


def test_tabular_sequential_local_bagging():
    config = _construct_tabular_bag_test_config(SEQUENTIAL_LOCAL_BAGGING)
    run_tabular_benchmarks(**config)


def test_sample_weight():
    dataset = {
        "url": "https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip",
        "name": "toyRegression",
        "problem_type": REGRESSION,
        "label": "y",
        "performance_val": 0.183,
    }
    directory_prefix = "./datasets/"
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = os.path.join(directory_prefix, dataset["name"])
    savedir = os.path.join(directory, "AutogluonOutput")
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    sample_weight = "sample_weights"
    weights = np.abs(
        np.random.rand(
            len(train_data),
        )
    )
    test_weights = np.abs(
        np.random.rand(
            len(test_data),
        )
    )
    train_data[sample_weight] = weights
    test_data_weighted = test_data.copy()
    test_data_weighted[sample_weight] = test_weights
    fit_args = {"time_limit": 20}
    predictor = TabularPredictor(label=dataset["label"], path=savedir, problem_type=dataset["problem_type"], sample_weight=sample_weight).fit(
        train_data, **fit_args
    )
    ldr = predictor.leaderboard(test_data)
    perf = predictor.evaluate(test_data)
    # Run again with weight_evaluation:
    # FIXME: RMSE doesn't support sample_weight, this entire call doesn't make sense
    predictor = TabularPredictor(
        label=dataset["label"], path=savedir, problem_type=dataset["problem_type"], sample_weight=sample_weight, weight_evaluation=True
    ).fit(train_data, **fit_args)
    # perf = predictor.evaluate(test_data_weighted)  # TODO: Doesn't work without implementing sample_weight in evaluate
    predictor.distill(time_limit=10)
    ldr = predictor.leaderboard(test_data_weighted)


def test_quantile():
    quantile_levels = [0.01, 0.02, 0.05, 0.98, 0.99]
    dataset = {"url": "https://autogluon.s3.amazonaws.com/datasets/toyRegression.zip", "name": "toyRegression", "problem_type": QUANTILE, "label": "y"}
    directory_prefix = "./datasets/"
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file, name=dataset["name"], url=dataset["url"])
    print(f"Evaluating Benchmark Dataset {dataset['name']}")
    directory = directory_prefix + dataset["name"] + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    fit_args = {"time_limit": 40}
    predictor = TabularPredictor(label=dataset["label"], path=savedir, problem_type=dataset["problem_type"], quantile_levels=quantile_levels).fit(
        train_data, **fit_args
    )
    ldr = predictor.leaderboard(test_data)
    perf = predictor.evaluate(test_data)


@pytest.mark.slow
def test_tabular_bagstack():
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
        nn_options = {"num_epochs": 2}
        gbm_options = [
            {"num_boost_round": 40},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ]
        hyperparameters = {"GBM": gbm_options, "NN_TORCH": nn_options}
        time_limit = 240

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_stack_levels": num_stack_levels,
        "verbosity": verbosity,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
        fit_args["num_bag_sets"] = 2
    ###################################################################
    run_tabular_benchmarks(
        fast_benchmark=fast_benchmark, subsample_size=subsample_size, perf_threshold=perf_threshold, seed_val=seed_val, fit_args=fit_args, run_distill=True
    )


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
        nn_options = {"num_epochs": 2}
        gbm_options = [
            {"num_boost_round": 40},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ]
        hyperparameters = {"GBM": gbm_options, "NN_TORCH": nn_options}
        time_limit = 240

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_stack_levels": num_stack_levels,
        "verbosity": verbosity,
        "use_bag_holdout": True,
    }
    if hyperparameter_tune_kwargs is not None:
        fit_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
    if hyperparameters is not None:
        fit_args["hyperparameters"] = hyperparameters
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
        fit_args["num_bag_sets"] = 2
    ###################################################################
    run_tabular_benchmarks(
        fast_benchmark=fast_benchmark,
        subsample_size=subsample_size,
        perf_threshold=perf_threshold,
        seed_val=seed_val,
        fit_args=fit_args,
        run_distill=True,
        crash_in_oof=True,
    )


def test_tabular_raise_on_nonfinite_float_labels():
    predictor = TabularPredictor(label="y")
    nonfinite_values = [np.nan, np.inf, -np.inf]

    for idx, nonfinite_value in enumerate(nonfinite_values):
        train_data = TabularDataset({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 3.0, 4.0]})
        train_data.loc[idx, "y"] = nonfinite_value

        with pytest.raises(ValueError) as ex_info:
            predictor.fit(train_data)
        assert str(ex_info.value).split()[-1] == str(idx)


def test_tabular_raise_on_nonfinite_class_labels():
    predictor = TabularPredictor(label="y")
    nonfinite_values = [np.nan, np.inf, -np.inf]

    for idx, nonfinite_value in enumerate(nonfinite_values):
        train_data = TabularDataset({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "c", "d", "e"]})
        train_data.loc[idx, "y"] = nonfinite_value

        with pytest.raises(ValueError) as ex_info:
            predictor.fit(train_data)
        assert str(ex_info.value).split()[-1] == str(idx)


def test_tabular_log_to_file():
    data_root = "https://autogluon.s3.amazonaws.com/datasets/Inc/"
    train_data = TabularDataset(data_root + "train.csv")
    train_data = train_data.sample(500)

    predictor = TabularPredictor(label="class", log_to_file=True).fit(train_data=train_data, hyperparameters={"DUMMY": {}})
    log = TabularPredictor.load_log(predictor_path=predictor.path)
    assert "TabularPredictor saved." in log[-1]

    log_file = os.path.join(".", "temp.log")
    predictor = TabularPredictor(label="class", log_to_file=True, log_file_path=log_file).fit(train_data=train_data, hyperparameters={"DUMMY": {}})
    log = TabularPredictor.load_log(log_file_path=log_file)
    assert "TabularPredictor saved." in log[-1]
    if not on_windows:
        os.remove(log_file)

    with pytest.raises(AssertionError):
        TabularPredictor.load_log()
