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
