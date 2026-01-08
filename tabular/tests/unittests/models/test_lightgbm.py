import numpy as np
import pytest

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.testing import FitHelper, ModelFitHelper


def test_lightgbm():
    model_cls = LGBModel
    model_hyperparameters = {}

    """Additionally tests that all metrics work"""
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, extra_metrics=True)


def test_lightgbm_binary_model():
    fit_args = dict()
    dataset_name = "toy_binary"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_multiclass_model():
    fit_args = dict()
    dataset_name = "toy_multiclass"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_regression_model():
    fit_args = dict()
    dataset_name = "toy_regression"
    ModelFitHelper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_quantile_model():
    fit_args = dict()
    dataset_name = "toy_quantile"
    ModelFitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        model=LGBModel(
            problem_type="quantile",
            hyperparameters={"ag.quantile_levels": [0.25, 0.5, 0.75]},
        ),
        fit_args=fit_args,
    )


def test_lightgbm_binary_with_calibrate_decision_threshold():
    """Tests that calibrate_decision_threshold works and does not make the validation score worse on the given metric"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = "toy_binary"

    predictor: TabularPredictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, delete_directory=False, refit_full=False
    )

    for metric in [None, "f1", "balanced_accuracy", "mcc", "recall", "precision"]:
        decision_threshold = predictor.calibrate_decision_threshold(metric=metric)
        if metric is None:
            metric = predictor.eval_metric.name
        assert decision_threshold >= 0
        assert decision_threshold <= 1

        X_val, y_val = predictor.load_data_internal(data="val", return_X=True, return_y=True)
        y_val = predictor.transform_labels(labels=y_val, inverse=True)

        y_pred_val = predictor.predict(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold = predictor.predict(
            data=X_val, decision_threshold=decision_threshold, transform_features=False
        )
        y_pred_multi_val_w_decision_threshold = predictor.predict_multi(
            data=X_val, decision_threshold=decision_threshold, transform_features=False
        )
        y_pred_multi_val_w_decision_threshold_cache = predictor.predict_multi(decision_threshold=decision_threshold)

        y_pred_proba_val = predictor.predict_proba(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold_from_proba = predictor.predict_from_proba(
            y_pred_proba=y_pred_proba_val, decision_threshold=decision_threshold
        )

        assert y_pred_val_w_decision_threshold.equals(y_pred_multi_val_w_decision_threshold[predictor.model_best])
        assert y_pred_val_w_decision_threshold.equals(
            y_pred_multi_val_w_decision_threshold_cache[predictor.model_best]
        )
        assert y_pred_val_w_decision_threshold.equals(y_pred_val_w_decision_threshold_from_proba)

        result = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val)
        result_calibrated = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val_w_decision_threshold)

        # Ensure validation score never becomes worse on the calibrated metric
        assert result[metric] <= result_calibrated[metric]
        if metric in ["recall"]:
            # recall should always be able to achieve a perfect validation score
            assert result_calibrated[metric] == 1.0

    assert predictor.calibrate_decision_threshold(metric="roc_auc") == 0.5


def test_lightgbm_binary_with_calibrate_decision_threshold_bagged_refit():
    """Tests that calibrate_decision_threshold works and does not make the validation score worse on the given metric"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
        num_bag_folds=2,
        calibrate_decision_threshold=True,
    )
    init_args = dict(eval_metric="f1")
    dataset_name = "toy_binary"

    train_data, test_data, dataset_info = FitHelper.load_dataset(name=dataset_name)
    label = dataset_info["label"]
    predictor: TabularPredictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, delete_directory=False, refit_full=True
    )

    expected_decision_threshold = 0.499
    assert predictor._decision_threshold is not None
    assert np.isclose(predictor.decision_threshold, expected_decision_threshold)
    assert predictor.decision_threshold == predictor._decision_threshold
    optimal_decision_threshold = predictor.calibrate_decision_threshold()
    assert optimal_decision_threshold == predictor.decision_threshold
    og_threshold = predictor.decision_threshold

    y_pred_test = predictor.predict(test_data)

    scores_predictions = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_test)
    scores = predictor.evaluate(test_data)
    scores_05 = predictor.evaluate(test_data, decision_threshold=0.5)

    for k in scores_predictions:
        assert scores[k] == scores_predictions[k]
    assert scores["f1"] > scores_05["f1"]  # Calibration should help f1
    assert (
        scores["accuracy"] == scores_05["accuracy"]
    )  # Calibration should not change accuracy (for this specific dataset)

    predictor.set_decision_threshold(0.5)
    assert predictor.decision_threshold == 0.5
    assert predictor._decision_threshold == 0.5
    scores_05_native = predictor.evaluate(test_data)

    for k in scores_05:
        assert scores_05[k] == scores_05_native[k]

    leaderboard_05 = predictor.leaderboard(test_data)
    lb_score_05 = leaderboard_05[leaderboard_05["model"] == predictor.model_best].iloc[0]["score_test"]
    assert lb_score_05 == scores_05["f1"]

    predictor.set_decision_threshold(og_threshold)
    leaderboard = predictor.leaderboard(test_data)
    lb_score = leaderboard[leaderboard["model"] == predictor.model_best].iloc[0]["score_test"]
    assert lb_score == scores["f1"]


def test_clean_column_name_replaces_expected_symbols():
    assert LGBModel._clean_column_name_for_lgb('a"b') == "a_b"
    assert LGBModel._clean_column_name_for_lgb("a,b") == "a_b"
    assert LGBModel._clean_column_name_for_lgb("a:b") == "a_b"
    assert LGBModel._clean_column_name_for_lgb("a{b}c") == "a_b_c"
    assert LGBModel._clean_column_name_for_lgb("a[b]c") == "a_b_c"


def test_rename_columns_outputs_are_unique_when_inputs_unique():
    features = ['a"b', "a,b", "x", "y"]
    m = LGBModel._rename_columns(features)
    assert set(m.keys()) == set(features)
    assert len(set(m.values())) == len(features)


def test_rename_columns_resolves_cleaning_collisions_with_suffixes():
    features = ['a"b', "a,b", "a:b"]
    m = LGBModel._rename_columns(features)
    # all clean to "a_b"
    assert m['a"b'] == "a_b"
    assert m["a,b"] == "a_b_2"
    assert m["a:b"] == "a_b_3"
    assert len(set(m.values())) == 3


def test_rename_columns_avoids_clashing_with_existing_feature_names():
    # cleaned "a" for first, then second is literally "a_2"
    # ensure collision resolution doesn't produce duplicates
    features = ["a", "a_2", 'a"2']  # 'a"2' cleans to 'a_2'
    m = LGBModel._rename_columns(features)
    assert len(set(m.values())) == len(features)
    # expected behavior given algorithm/order:
    assert m["a"] == "a"
    assert m["a_2"] == "a_2"
    assert m['a"2'] == "a_2_2"  # because "a_2" already taken


def test_rename_columns_is_order_dependent_but_still_unique():
    f1 = ['a"b', "a,b"]
    f2 = ["a,b", 'a"b']
    m1 = LGBModel._rename_columns(f1)
    m2 = LGBModel._rename_columns(f2)

    assert len(set(m1.values())) == 2
    assert len(set(m2.values())) == 2
    # but the assignment of base vs suffixed swaps with order:
    assert m1['a"b'] == "a_b"
    assert m1["a,b"] == "a_b_2"
    assert m2["a,b"] == "a_b"
    assert m2['a"b'] == "a_b_2"


def test_non_string_features_are_returned_as_is_and_can_collide():
    # Demonstrates behavior: non-strings are not cleaned, but uniqueness is still enforced.
    features = [123, "123"]
    m = LGBModel._rename_columns(features)
    assert m[123] == 123
    assert m["123"] == "123"
    assert len(set(m.values())) == 2


def test_duplicate_input_feature_names_raises_value_error():
    features = ["dup", "dup"]
    with pytest.raises(
        ValueError,
        match="features contains duplicates; cannot create 1-to-1 mapping with a dict.",
    ):
        LGBModel._rename_columns(features)


def test_bool_int_key_collision_raises_value_error():
    # True == 1 in Python, so this is a duplicate under set/dict semantics
    features = [1, True]
    with pytest.raises(
        ValueError,
        match="features contains duplicates; cannot create 1-to-1 mapping with a dict.",
    ):
        LGBModel._rename_columns(features)
