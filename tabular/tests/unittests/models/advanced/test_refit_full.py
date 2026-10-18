import shutil

from autogluon.tabular import TabularPredictor
from autogluon.tabular.testing import FitHelper


def test_refit_full_train_data_extra():
    """
    Verifies that `refit_full(train_data_extra)` works.
    """
    dataset = "adult"  # Need a dataset with categorical features + NaNs in categories
    train_data, test_data, dataset_info = FitHelper.load_dataset(dataset)

    len_train = len(train_data)
    len_test = len(test_data)
    len_combined = len_train + len_test

    predictor = TabularPredictor(label=dataset_info["label"], problem_type=dataset_info["problem_type"])

    predictor.fit(
        train_data=train_data,
        hyperparameters={"NN_TORCH": {"num_epochs": 1}},
        raise_on_model_failure=True,
        fit_weighted_ensemble=False,
    )

    assert len(predictor.model_names()) == 1
    model_name = predictor.model_names()[0]
    refit_model_map = predictor.refit_full(train_data_extra=test_data)
    refit_model_name = refit_model_map[model_name]

    assert len(predictor.model_names()) == 2

    refit_model_info = predictor.model_info(refit_model_name)

    # Ensure refit uses all of train_data and all of train_data_extra
    assert refit_model_info["num_samples"] == len_combined

    predictor.predict(test_data, model=refit_model_name)

    shutil.rmtree(predictor.path, ignore_errors=True)


def test_refit_full_train_data_extra_bag():
    """
    Verifies that `refit_full(train_data_extra)` works when bagging
    """
    dataset = "adult"  # Need a dataset with categorical features + NaNs in categories
    train_data, test_data, dataset_info = FitHelper.load_dataset(dataset)

    len_train = len(train_data)
    len_test = len(test_data)
    len_combined = len_train + len_test

    predictor = TabularPredictor(label=dataset_info["label"], problem_type=dataset_info["problem_type"])

    predictor.fit(
        train_data=train_data,
        hyperparameters={"NN_TORCH": {"num_epochs": 1}},
        raise_on_model_failure=True,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    assert len(predictor.model_names()) == 1
    model_name = predictor.model_names()[0]
    refit_model_map = predictor.refit_full(train_data_extra=test_data)
    refit_model_name = refit_model_map[model_name]

    assert len(predictor.model_names()) == 2

    refit_model_info = predictor.model_info(refit_model_name)

    # Ensure refit uses all of train_data and all of train_data_extra
    assert refit_model_info["num_samples"] == len_train
    assert refit_model_info["children_info"]["S1F1"]["num_samples"] == len_combined

    predictor.predict(test_data, model=refit_model_name)

    shutil.rmtree(predictor.path, ignore_errors=True)


def _fit_bagged_dummy_with_refit_full() -> TabularPredictor:
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        num_bag_sets=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    return FitHelper.fit_and_validate_dataset(
        dataset_name="toy_binary",
        fit_args=fit_args,
        expected_model_count=2,
        refit_full=True,
        delete_directory=False,
    )


def _simulate_refit_without_predict_time(predictor: TabularPredictor) -> None:
    """Dummy refit_full clones the parent and copies predict_time; real refits leave it None."""
    for full_model in predictor.model_refit_map().values():
        predictor._trainer._update_model_attr(full_model, predict_time=None)
    predictor._trainer.save()


def test_get_model_best_when_only_refit_full_models_remain():
    """get_model_best() must work when only `_FULL` models remain (issue #5552)."""
    predictor = _fit_bagged_dummy_with_refit_full()
    _, test_data, _ = FitHelper.load_dataset(name="toy_binary")
    _simulate_refit_without_predict_time(predictor)

    full_models = list(predictor.model_refit_map().values())
    assert full_models
    for full_model in full_models:
        assert predictor._trainer.get_model_attribute(full_model, "val_score") is None
        assert predictor._trainer.get_model_attribute_full(full_model, "predict_time") is None

    predictor.delete_models(models_to_keep=full_models, dry_run=False)
    assert set(predictor.model_names()) == set(full_models)
    assert predictor._trainer.get_model_best() in full_models
    predictor.predict(test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_clone_for_deployment_non_best_refit_full_model():
    """clone_for_deployment of a non-best `_FULL` model (issue #5552)."""
    predictor = _fit_bagged_dummy_with_refit_full()
    _, test_data, _ = FitHelper.load_dataset(name="toy_binary")
    _simulate_refit_without_predict_time(predictor)

    refit_map = predictor.model_refit_map()
    assert len(refit_map) >= 2
    non_best_full = next(full for full in refit_map.values() if full != predictor.model_best)
    assert predictor._trainer.get_model_attribute(non_best_full, "val_score") is None
    assert predictor._trainer.get_model_attribute_full(non_best_full, "predict_time") is None

    clone_path = predictor.path + "_clone_for_deployment_full"
    clone = predictor.clone_for_deployment(path=clone_path, model=non_best_full, return_clone=True)
    assert clone.model_best == non_best_full
    clone.predict(test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)
    shutil.rmtree(clone_path, ignore_errors=True)
