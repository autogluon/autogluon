"""Runs autogluon.tabular on multiple benchmark datasets.

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

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from autogluon.common import space
from autogluon.common.utils.simulation_utils import convert_simulation_artifacts_to_tabular_predictions_dict
from autogluon.core.constants import BINARY, MULTICLASS, PROBLEM_TYPES_CLASSIFICATION, QUANTILE
from autogluon.tabular import TabularDataset, TabularPredictor, __version__
from autogluon.tabular.testing import FitHelper

PARALLEL_LOCAL_BAGGING = "parallel_local"
SEQUENTIAL_LOCAL_BAGGING = "sequential_local"
on_windows = os.name == "nt"


def test_tabular():
    """
    Verifies that default parameter TabularPredictor works on binary, multiclass, regression and quantile tasks.
    """
    fit_args = {"time_limit": 60}
    run_tabular_benchmarks(fit_args=fit_args)


def _assert_predict_dict_identical_to_predict(predictor: TabularPredictor, data: pd.DataFrame):
    """Assert that predict_multi is identical to looping calls to predict"""
    for as_pandas in [True, False]:
        for inverse_transform in [True, False]:
            predict_dict = predictor.predict_multi(data=data, as_pandas=as_pandas, inverse_transform=inverse_transform)
            assert set(predictor.model_names()) == set(predict_dict.keys())
            for m in predictor.model_names():
                if not inverse_transform:
                    model_pred = predictor._learner.predict(
                        data, model=m, as_pandas=as_pandas, inverse_transform=inverse_transform
                    )
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


def _assert_predict_proba_dict_identical_to_predict_proba(predictor: TabularPredictor, data: pd.DataFrame):
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
                            data,
                            model=m,
                            as_pandas=as_pandas,
                            as_multiclass=as_multiclass,
                            inverse_transform=inverse_transform,
                        )
                    else:
                        model_pred_proba = predictor.predict_proba(
                            data, model=m, as_pandas=as_pandas, as_multiclass=as_multiclass
                        )
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
    directory_prefix = "./datasets/"
    dataset_name = "toy_binary_10"
    train_data, test_data, dataset_info = FitHelper.load_dataset(dataset_name)
    problem_type = dataset_info["problem_type"]
    label = dataset_info["label"]

    print(f"Evaluating Advanced Functionality on Benchmark Dataset {dataset_name}")
    directory = directory_prefix + "advanced/" + dataset_name + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(
        savedir, ignore_errors=True
    )  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    savedir_predictor_original = savedir + "predictor/"
    predictor: TabularPredictor = TabularPredictor(
        label=label, problem_type=problem_type, path=savedir_predictor_original
    ).fit(train_data)

    version_in_file = predictor._load_version_file(path=predictor.path)
    assert version_in_file == __version__

    leaderboard = predictor.leaderboard(data=test_data)

    # test metric_error leaderboard
    leaderboard_error = predictor.leaderboard(data=test_data, score_format="error")
    assert sorted(leaderboard["model"].to_list()) == sorted(leaderboard_error["model"].to_list())
    leaderboard_combined = pd.merge(
        leaderboard, leaderboard_error[["model", "metric_error_test", "metric_error_val"]], on=["model"]
    )
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
    assert sorted(list(simulation_artifact["pred_proba_dict_test"].keys())) == sorted(
        predictor.model_names(can_infer=True)
    )
    assert simulation_artifact["y_test"].equals(predictor.transform_labels(test_data[label]))
    for sim_artifact in [simulation_artifact, simulation_artifact_no_test]:
        assert sim_artifact["label"] == predictor.label
        assert sorted(list(sim_artifact["pred_proba_dict_val"].keys())) == sorted(
            predictor.model_names(can_infer=True)
        )
        assert sim_artifact["eval_metric"] == predictor.eval_metric.name
        assert sim_artifact["problem_type"] == predictor.problem_type
    simulation_artifacts = {dataset_name: {0: simulation_artifact}}

    # Test convert_simulation_artifacts_to_tabular_predictions_dict
    aggregated_pred_proba, aggregated_ground_truth = convert_simulation_artifacts_to_tabular_predictions_dict(
        simulation_artifacts=simulation_artifacts
    )
    assert set(aggregated_pred_proba[dataset_name][0]["pred_proba_dict_val"].keys()) == set(
        predictor.model_names(can_infer=True)
    )
    assert set(aggregated_pred_proba[dataset_name][0]["pred_proba_dict_test"].keys()) == set(
        predictor.model_names(can_infer=True)
    )
    ground_truth_keys_expected = set(simulation_artifact.keys())
    ground_truth_keys_expected.remove("pred_proba_dict_val")
    ground_truth_keys_expected.remove("pred_proba_dict_test")
    assert set(aggregated_ground_truth[dataset_name][0].keys()) == ground_truth_keys_expected

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
    assert set(leaderboard_extra.columns).issuperset(
        set(extra_metrics)
    )  # Assert that extra_metrics are present in output
    num_models = len(predictor.model_names())
    feature_importances = predictor.feature_importance(data=test_data)
    original_features = set(train_data.columns)
    original_features.remove(label)
    assert set(feature_importances.index) == original_features
    assert set(feature_importances.columns) == {"importance", "stddev", "p_value", "n", "p99_high", "p99_low"}
    predictor.transform_features()
    test_data_transformed = predictor.transform_features(data=test_data)
    info = predictor.info()
    for model in predictor.model_names():
        model_info = predictor.model_info(model=model)
        model_info_2 = info["model_info"][model]
        assert model_info["name"] == model_info_2["name"]
        assert model_info["name"] == model
        assert set(model_info.keys()) == set(model_info_2.keys())
        model_hyperparameters = predictor.model_hyperparameters(model=model)
        assert isinstance(model_hyperparameters, dict)

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
    assert (
        predictor.persist(models="all", max_memory=None) == []
    )  # Ensure that no additional models are persisted on repeated calls
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
    assert (
        predictor_loaded.model_names(persisted=True) == []
    )  # Assert that models were not still persisted after loading predictor

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
    predictor.delete_models(models_to_keep=[], dry_run=True)  # Test that dry-run doesn't delete models
    assert len(predictor.model_names()) == num_models * 2
    predictor.predict(data=test_data)

    # Test refit_full with train_data_extra argument
    refit_full_models = list(predictor.model_refit_map().values())
    predictor.delete_models(models_to_delete=refit_full_models)
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
    directory_prefix = "./datasets/"
    dataset_name = "toy_binary_10"
    train_data, test_data, dataset_info = FitHelper.load_dataset("toy_binary_10")
    problem_type = dataset_info["problem_type"]
    label = dataset_info["label"]

    print(f"Evaluating Advanced Functionality (Bagging) on Benchmark Dataset {dataset_name}")
    directory = directory_prefix + "advanced/" + dataset_name + "/"
    savedir = directory + "AutogluonOutput/"
    shutil.rmtree(
        savedir, ignore_errors=True
    )  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
    gbm_hyperparameters = {"ag_args_fit": {"foo": 5}}
    predictor = TabularPredictor(label=label, problem_type=problem_type, path=savedir).fit(
        train_data,
        num_bag_folds=2,
        hyperparameters={"GBM": gbm_hyperparameters},
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

    info = predictor.info()
    for model in predictor.model_names():
        model_info = predictor.model_info(model=model)
        model_info_2 = info["model_info"][model]
        assert model_info["name"] == model_info_2["name"]
        assert model_info["name"] == model
        assert set(model_info.keys()) == set(model_info_2.keys())
        model_hyperparameters = predictor.model_hyperparameters(model=model)
        assert isinstance(model_hyperparameters, dict)
    assert predictor.model_hyperparameters(model="LightGBM_BAG_L1") == gbm_hyperparameters
    lightgbm_full_params = predictor.model_hyperparameters(
        model="LightGBM_BAG_L1_FULL", include_ag_args_ensemble=False
    )
    assert lightgbm_full_params != gbm_hyperparameters
    lightgbm_full_params.pop("num_boost_round")
    assert lightgbm_full_params == gbm_hyperparameters


def verify_predictor(
    predictor: TabularPredictor,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    crash_in_oof: bool,
    run_distill: bool,
):
    label = predictor.label
    y_test = test_data[label]
    assert len(predictor._trainer._models_failed_to_train_errors.keys()) == 0
    results = predictor.fit_summary(verbosity=4)
    original_features = list(train_data)
    original_features.remove(label)
    assert original_features == predictor.original_features
    y_pred_empty = predictor.predict(test_data[0:0])
    assert len(y_pred_empty) == 0
    y_pred = predictor.predict(test_data)
    perf_dict = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    if predictor._trainer.bagged_mode and not crash_in_oof:
        # TODO: Test index alignment with original training data (first handle duplicated rows / dropped rows edge cases)
        y_pred_oof = predictor.predict_oof()
        y_pred_proba_oof = predictor.predict_proba_oof(as_multiclass=False)
        y_pred_oof_transformed = predictor.predict_oof(transformed=True)
        y_pred_proba_oof_transformed = predictor.predict_proba_oof(as_multiclass=False, transformed=True)

        # Assert expected type output
        if predictor.problem_type == QUANTILE:
            assert isinstance(y_pred_oof, pd.DataFrame)
            assert isinstance(y_pred_oof_transformed, pd.DataFrame)
        else:
            assert isinstance(y_pred_oof, pd.Series)
            assert isinstance(y_pred_oof_transformed, pd.Series)
        if predictor.problem_type in [MULTICLASS, QUANTILE]:
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
        y_pred_proba_oof_inverse_inverse = predictor.transform_labels(
            y_pred_proba_oof_inverse, proba=True, inverse=True
        )
        y_pred_oof_inverse = predictor.transform_labels(y_pred_oof)
        y_pred_oof_inverse_inverse = predictor.transform_labels(y_pred_oof_inverse, inverse=True)

        if isinstance(y_pred_proba_oof_transformed, pd.DataFrame):
            pd.testing.assert_frame_equal(y_pred_proba_oof_transformed, y_pred_proba_oof_inverse)
            pd.testing.assert_frame_equal(y_pred_proba_oof, y_pred_proba_oof_inverse_inverse)
        else:
            pd.testing.assert_series_equal(y_pred_proba_oof_transformed, y_pred_proba_oof_inverse)
            pd.testing.assert_series_equal(y_pred_proba_oof, y_pred_proba_oof_inverse_inverse)
        if isinstance(y_pred_oof_transformed, pd.DataFrame):
            pd.testing.assert_frame_equal(y_pred_oof_transformed, y_pred_oof_inverse)
            pd.testing.assert_frame_equal(y_pred_oof, y_pred_oof_inverse_inverse)
        else:
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


def run_tabular_benchmarks(
    fit_args: dict,
    subsample_size: int | None = None,
    datasets: list[str] | None = None,
    run_distill: bool = False,
    crash_in_oof: bool = False,
):
    print("Running fit with args:")
    print(fit_args)

    if datasets is None:
        datasets = [
            "toy_binary_10",
            "toy_multiclass_10",
            "toy_regression_10",
            "toy_quantile_10",
        ]
    for dataset_name in datasets:
        predictor = FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            sample_size=subsample_size,
            refit_full=False,
            expected_model_count=None,
            raise_on_model_failure=True,
            delete_directory=False,
        )
        train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)
        verify_predictor(
            predictor=predictor,
            train_data=train_data,
            test_data=test_data,
            crash_in_oof=crash_in_oof,
            run_distill=run_distill,
        )
        shutil.rmtree(predictor.path, ignore_errors=True)


def test_pseudolabeling():
    datasets = [
        "toy_binary",
        "toy_multiclass",
        "toy_regression",
    ]

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
        train_data, test_data, dataset_info = FitHelper.load_dataset(dataset)
        label = dataset_info["label"]
        problem_type = dataset_info["problem_type"]
        name = dataset

        print(f"Testing dataset with name: {name}, problem type: {problem_type}")

        if problem_type in PROBLEM_TYPES_CLASSIFICATION:
            valid_class_idxes = test_data[label].isin(train_data[label].unique())
            test_data = test_data[valid_class_idxes]

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


def test_tabular_bag_stack_hpo():
    num_bag_folds = 2
    num_bag_sets = 2
    num_stack_levels = 1
    time_limit = 50
    hyperparameters = {
        "GBM": {"num_boost_round": 20, "learning_rate": space.Real(0.01, 0.1)},
        "NN_TORCH": {"num_epochs": 1, "learning_rate": space.Real(0.001, 0.01)},
    }
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "auto"}

    datasets = ["toy_binary_10"]
    subsample_size = 100

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_bag_sets": num_bag_sets,
        "num_stack_levels": num_stack_levels,
        "time_limit": time_limit,
        "hyperparameter_tune_kwargs": hyperparameter_tune_kwargs,
        "hyperparameters": hyperparameters,
    }
    run_tabular_benchmarks(
        subsample_size=subsample_size,
        datasets=datasets,
        fit_args=fit_args,
    )


def test_tabular_hpo():
    hyperparameter_tune_kwargs = {
        "scheduler": "local",
        "searcher": "auto",
        "num_trials": 3,
    }
    subsample_size = 100
    fit_args = {
        "verbosity": 2,  # how much output to print
        "time_limit": 240,
        "hyperparameter_tune_kwargs": hyperparameter_tune_kwargs,
    }
    run_tabular_benchmarks(subsample_size=subsample_size, fit_args=fit_args)


def test_tabular_feature_prune():
    feature_prune_kwargs = {
        "stop_threshold": 3,
        "prune_ratio": 0.05,
        "prune_threshold": "none",
        "n_train_subsample": 1000,
        "n_fi_subsample": 5000,
        "min_fi_samples": 5000,
        "feature_prune_time_limit": 10,
        "raise_exception": True,
    }
    datasets = ["adult"]

    subsample_size = 1000
    gbm_options = {"num_boost_round": 20}
    hyperparameters = {"GBM": gbm_options}

    fit_args = {
        "hyperparameters": hyperparameters,
        "feature_prune_kwargs": feature_prune_kwargs,
        "time_limit": 60,
    }
    run_tabular_benchmarks(subsample_size=subsample_size, datasets=datasets, fit_args=fit_args)


def _construct_tabular_bag_test_config(fold_fitting_strategy) -> dict:
    num_bag_folds = 3
    num_bag_sets = 2
    num_stack_levels = 0

    nn_options = {"num_epochs": 1}
    gbm_options = {"num_boost_round": 30}
    hyperparameters = {"GBM": gbm_options, "NN_TORCH": nn_options}

    fit_args = {
        "num_bag_folds": num_bag_folds,
        "num_bag_sets": num_bag_sets,
        "num_stack_levels": num_stack_levels,
        "hyperparameters": hyperparameters,
        "ag_args_ensemble": {
            "fold_fitting_strategy": fold_fitting_strategy,
        },
        "time_limit": 60,
    }
    ###################################################################
    return fit_args


def test_tabular_parallel_local_bagging():
    fit_args = _construct_tabular_bag_test_config(PARALLEL_LOCAL_BAGGING)
    run_tabular_benchmarks(fit_args=fit_args)


def test_tabular_sequential_local_bagging():
    fit_args = _construct_tabular_bag_test_config(SEQUENTIAL_LOCAL_BAGGING)
    run_tabular_benchmarks(fit_args=fit_args)


def test_sample_weight():
    dataset_name = "toy_regression_10"
    train_data, test_data, dataset_info = FitHelper.load_dataset(dataset_name)
    label = dataset_info["label"]
    problem_type = dataset_info["problem_type"]

    directory_prefix = "./datasets/"
    print(f"Evaluating Benchmark Dataset {dataset_name}")
    directory = os.path.join(directory_prefix, dataset_name)
    savedir = os.path.join(directory, "AutogluonOutput")
    shutil.rmtree(
        savedir, ignore_errors=True
    )  # Delete AutoGluon output directory to ensure previous runs' information has been removed.
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
    fit_args = {"raise_on_model_failure": True}
    predictor = TabularPredictor(
        label=label, path=savedir, problem_type=problem_type, sample_weight=sample_weight
    ).fit(train_data, **fit_args)
    ldr = predictor.leaderboard(test_data)
    perf = predictor.evaluate(test_data)
    # Run again with weight_evaluation:
    # FIXME: RMSE doesn't support sample_weight, this entire call doesn't make sense
    predictor = TabularPredictor(
        label=label, path=savedir, problem_type=problem_type, sample_weight=sample_weight, weight_evaluation=True
    ).fit(train_data, **fit_args)
    # perf = predictor.evaluate(test_data_weighted)  # TODO: Doesn't work without implementing sample_weight in evaluate
    predictor.distill(time_limit=10)
    ldr = predictor.leaderboard(test_data_weighted)


def test_tabular_bag_stack():
    num_bag_folds = 2
    num_bag_sets = 1
    num_stack_levels = 1

    datasets = ["toy_binary_10"]

    nn_options = {"num_epochs": 2}
    gbm_options = [
        {"num_boost_round": 40},
        {
            "num_boost_round": 100,
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
        "num_bag_sets": num_bag_sets,
        "num_stack_levels": num_stack_levels,
        "hyperparameters": hyperparameters,
        "ag_args_ensemble": dict(fold_fitting_strategy="sequential_local"),
        "time_limit": time_limit,
    }
    run_tabular_benchmarks(fit_args=fit_args, run_distill=True, datasets=datasets)


def test_tabular_bag_stack_use_bag_holdout():
    num_bag_folds = 2
    num_bag_sets = 1
    num_stack_levels = 1

    datasets = ["toy_binary_10"]

    nn_options = {"num_epochs": 2}
    gbm_options = [
        {"num_boost_round": 40},
        {
            "num_boost_round": 100,
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
        "num_bag_sets": num_bag_sets,
        "num_stack_levels": num_stack_levels,
        "use_bag_holdout": True,
        "hyperparameters": hyperparameters,
        "time_limit": time_limit,
        "ag_args_ensemble": dict(fold_fitting_strategy="sequential_local"),
    }
    run_tabular_benchmarks(
        fit_args=fit_args,
        run_distill=True,
        crash_in_oof=True,
        datasets=datasets,
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
    dataset = "toy_binary_10"
    train_data, test_data, dataset_info = FitHelper.load_dataset(dataset)
    label = dataset_info["label"]

    predictor = TabularPredictor(label=label, log_to_file=True).fit(
        train_data=train_data, hyperparameters={"DUMMY": {}}
    )
    log = TabularPredictor.load_log(predictor_path=predictor.path)
    assert "TabularPredictor saved." in log[-1]

    log_file = os.path.join(".", "temp.log")
    predictor = TabularPredictor(label=label, log_to_file=True, log_file_path=log_file).fit(
        train_data=train_data, hyperparameters={"DUMMY": {}}
    )
    log = TabularPredictor.load_log(log_file_path=log_file)
    assert "TabularPredictor saved." in log[-1]
    if not on_windows:
        os.remove(log_file)

    with pytest.raises(AssertionError):
        TabularPredictor.load_log()
