def test_calibrate_binary(fit_helper):
    """Tests that calibrate=True doesn't crash in binary"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "toy_binary"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_binary_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in binary w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_binary"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_multiclass(fit_helper):
    """Tests that calibrate=True doesn't crash in multiclass"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "toy_multiclass"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_multiclass_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in multiclass w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_multiclass"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_quantile(fit_helper):
    """Tests that calibrate=True doesn't crash in quantile"""
    fit_args = dict(
        hyperparameters={"RF": {}},
        calibrate=True,
    )
    dataset_name = "toy_quantile"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_calibrate_quantile_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in quantile w/ bagging"""
    fit_args = dict(
        hyperparameters={"RF": {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "toy_quantile"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
