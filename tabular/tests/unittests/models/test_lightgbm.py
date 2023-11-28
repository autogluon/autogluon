from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.tabular.models.lgb.lgb_model import LGBModel


def test_lightgbm_binary(fit_helper):
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_multiclass(fit_helper):
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_regression(fit_helper):
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_binary_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "adult"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_multiclass_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "covertype_small"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_regression_model(model_fit_helper):
    fit_args = dict()
    dataset_name = "ames"
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={"GBM": {}},
    )
    dataset_name = "ames"
    init_args = dict(problem_type="quantile", quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)


def test_lightgbm_binary_with_calibrate_decision_threshold(fit_helper):
    """Tests that calibrate_decision_threshold works and does not make the validation score worse on the given metric"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = "adult"

    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False, refit_full=False)

    for metric in [None, "f1", "balanced_accuracy", "mcc", "recall", "precision"]:
        decision_threshold = predictor.calibrate_decision_threshold(metric=metric)
        if metric is None:
            metric = predictor.eval_metric.name
        assert decision_threshold >= 0
        assert decision_threshold <= 1

        X_val, y_val = predictor.load_data_internal(data="val", return_X=True, return_y=True)
        y_val = predictor.transform_labels(labels=y_val, inverse=True)

        y_pred_val = predictor.predict(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold = predictor.predict(data=X_val, decision_threshold=decision_threshold, transform_features=False)
        y_pred_multi_val_w_decision_threshold = predictor.predict_multi(data=X_val, decision_threshold=decision_threshold, transform_features=False)
        y_pred_multi_val_w_decision_threshold_cache = predictor.predict_multi(decision_threshold=decision_threshold)

        y_pred_proba_val = predictor.predict_proba(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold_from_proba = predictor.predict_from_proba(y_pred_proba=y_pred_proba_val, decision_threshold=decision_threshold)

        assert y_pred_val_w_decision_threshold.equals(y_pred_multi_val_w_decision_threshold[predictor.model_best])
        assert y_pred_val_w_decision_threshold.equals(y_pred_multi_val_w_decision_threshold_cache[predictor.model_best])
        assert y_pred_val_w_decision_threshold.equals(y_pred_val_w_decision_threshold_from_proba)

        result = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val)
        result_calibrated = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val_w_decision_threshold)

        # Ensure validation score never becomes worse on the calibrated metric
        assert result[metric] <= result_calibrated[metric]
        if metric in ["recall"]:
            # recall should always be able to achieve a perfect validation score
            assert result_calibrated[metric] == 1.0

    assert predictor.calibrate_decision_threshold(metric="roc_auc") == 0.5


def test_lightgbm_binary_with_calibrate_decision_threshold_bagged_refit(fit_helper, dataset_loader_helper):
    """Tests that calibrate_decision_threshold works and does not make the validation score worse on the given metric"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
        num_bag_folds=2,
        calibrate_decision_threshold=True,
    )
    init_args = dict(eval_metric="f1")
    dataset_name = "adult"

    directory_prefix = "./datasets/"
    train_data, test_data, dataset_info = dataset_loader_helper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
    label = dataset_info["label"]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, delete_directory=False, refit_full=True)

    assert predictor._decision_threshold is not None
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
    assert scores["accuracy"] < scores_05["accuracy"]  # Calibration should harm accuracy

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
