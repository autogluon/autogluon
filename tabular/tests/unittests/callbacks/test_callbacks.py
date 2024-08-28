from autogluon.core.callbacks import EarlyStoppingCallback, EarlyStoppingEnsembleCallback, ExampleCallback
from autogluon.core.models import DummyModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel


def test_early_stopping_callback(fit_helper):
    callback = EarlyStoppingCallback()

    fit_args = dict(
        hyperparameters={
            DummyModel: {},
            LGBModel: {},
        },
        infer_limit=100,
        infer_limit_batch_size=1000,
        callbacks=[callback],
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=3, refit_full=False)

    assert callback.model_best == "LightGBM"
    assert callback.infer_limit is not None


def test_early_stopping_callback_v2(fit_helper):
    """
    Tests EarlyStoppingCallback early stops prior to fitting LightGBM.
    Tests `patience_per_level=True`
    """
    callback = EarlyStoppingCallback(patience=2, patience_per_level=True)

    fit_args = dict(
        hyperparameters={
            DummyModel: [{}, {}, {}, {}],
            LGBModel: {},
        },
        callbacks=[callback],
        num_bag_folds=2,
        num_stack_levels=1,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=6, refit_full=False)

    assert callback.model_best == "DummyModel_BAG_L1"
    assert callback.score_best == 0.76
    assert callback.infer_limit is None


def test_early_stopping_callback_v3(fit_helper):
    """
    Test EarlyStoppingCallback early stops prior to fitting LightGBM.
    Tests `patience_per_level=False`
    Tests passing multiple callbacks.
    """
    callback = EarlyStoppingCallback(patience=2, patience_per_level=False)

    fit_args = dict(
        hyperparameters={
            DummyModel: [{}, {}, {}, {}],
            LGBModel: {},
        },
        callbacks=[callback, ExampleCallback()],
        num_bag_folds=2,
        num_stack_levels=1,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=3, refit_full=False)

    assert callback.model_best == "DummyModel_BAG_L1"
    assert callback.score_best == 0.76
    assert callback.infer_limit is None


def test_early_stopping_ensemble_callback(fit_helper):
    callback = EarlyStoppingEnsembleCallback()

    fit_args = dict(
        hyperparameters={
            DummyModel: {},
            LGBModel: {},
        },
        infer_limit=100,
        infer_limit_batch_size=1000,
        callbacks=[callback],
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=4, refit_full=False)

    assert callback.model_best == "LightGBM"
    assert callback.infer_limit is not None
    assert callback.infer_limit_batch_size == 1000


def test_early_stopping_ensemble_callback_v2(fit_helper):
    """
    Tests EarlyStoppingEnsembleCallback early stops prior to fitting LightGBM.
    Tests `patience_per_level=True`
    """
    callback = EarlyStoppingEnsembleCallback(patience=2, patience_per_level=True)

    fit_args = dict(
        hyperparameters={
            DummyModel: [{}, {}, {}, {}],
            LGBModel: {},
        },
        callbacks=[callback],
        num_bag_folds=2,
        num_stack_levels=1,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=9, refit_full=False)

    assert callback.model_best == "DummyModel_BAG_L1"
    assert callback.score_best == 0.76
    assert callback.infer_limit is None
    assert callback.infer_limit_batch_size is None


def test_early_stopping_ensemble_callback_v3(fit_helper):
    """
    Test EarlyStoppingEnsembleCallback early stops prior to fitting LightGBM.
    Tests `patience_per_level=False`
    Tests passing multiple callbacks.
    """
    callback = EarlyStoppingEnsembleCallback(patience=2, patience_per_level=False)

    fit_args = dict(
        hyperparameters={
            DummyModel: [{}, {}, {}, {}],
            LGBModel: {},
        },
        callbacks=[callback, ExampleCallback()],
        num_bag_folds=2,
        num_stack_levels=1,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=5, refit_full=False)

    assert callback.model_best == "DummyModel_BAG_L1"
    assert callback.score_best == 0.76
    assert callback.infer_limit is None
    assert callback.infer_limit_batch_size is None
