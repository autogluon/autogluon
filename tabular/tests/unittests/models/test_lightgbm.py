
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS


def test_lightgbm_binary(fit_helper):
    """Additionally tests that all binary metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'adult'
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_multiclass(fit_helper):
    """Additionally tests that all multiclass metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[MULTICLASS])

    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_regression(fit_helper):
    """Additionally tests that all regression metrics work"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    extra_metrics = list(METRICS[REGRESSION])

    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, extra_metrics=extra_metrics)


def test_lightgbm_binary_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'adult'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_multiclass_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'covertype_small'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_regression_model(model_fit_helper):
    fit_args = dict()
    dataset_name = 'ames'
    model_fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, model=LGBModel(), fit_args=fit_args)


def test_lightgbm_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={'GBM': {}},
    )
    dataset_name = 'ames'
    init_args = dict(problem_type='quantile', quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)


def test_lightgbm_binary_with_calibrate_decision_threshold(fit_helper):
    """Tests that calibrate_decision_threshold works and does not make the validation score worse on the given metric"""
    fit_args = dict(
        hyperparameters={LGBModel: {}},
    )
    dataset_name = 'adult'

    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False, refit_full=False)

    for metric in [None, 'f1', 'balanced_accuracy', 'mcc', 'recall', 'precision']:
        decision_threshold = predictor.calibrate_decision_threshold(metric=metric)
        if metric is None:
            metric = predictor.eval_metric.name
        assert decision_threshold >= 0
        assert decision_threshold <= 1

        X_val, y_val = predictor.load_data_internal(data='val', return_X=True, return_y=True)
        y_val = predictor.transform_labels(labels=y_val, inverse=True)

        y_pred_val = predictor.predict(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold = predictor.predict(data=X_val, decision_threshold=decision_threshold, transform_features=False)
        y_pred_multi_val_w_decision_threshold = predictor.predict_multi(data=X_val, decision_threshold=decision_threshold, transform_features=False)
        y_pred_multi_val_w_decision_threshold_cache = predictor.predict_multi(decision_threshold=decision_threshold)

        y_pred_proba_val = predictor.predict_proba(data=X_val, transform_features=False)
        y_pred_val_w_decision_threshold_from_proba = predictor.get_pred_from_proba(y_pred_proba=y_pred_proba_val, decision_threshold=decision_threshold)

        assert y_pred_val_w_decision_threshold.equals(y_pred_multi_val_w_decision_threshold[predictor.get_model_best()])
        assert y_pred_val_w_decision_threshold.equals(y_pred_multi_val_w_decision_threshold_cache[predictor.get_model_best()])
        assert y_pred_val_w_decision_threshold.equals(y_pred_val_w_decision_threshold_from_proba)

        result = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val)
        result_calibrated = predictor.evaluate_predictions(y_true=y_val, y_pred=y_pred_val_w_decision_threshold)

        # Ensure validation score never becomes worse on the calibrated metric
        assert result[metric] <= result_calibrated[metric]
        if metric in ['recall']:
            # recall should always be able to achieve a perfect validation score
            assert result_calibrated[metric] == 1.0

    for decision_threshold in [0.0, 0.01, 0.02, 0.03, 0.1, 0.2, 0.4999, 0.5, 0.5001, 0.8, 0.9, 0.97, 0.98, 0.99, 1.0]:
        # TODO: Verify that predict_proba + get_pred_from_proba w/ threshold is equivalent to predict w/ threshold
        pass

    assert predictor.calibrate_decision_threshold(metric='roc_auc') == 0.5
