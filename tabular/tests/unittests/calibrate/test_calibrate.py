def test_calibrate_binary(fit_helper):
    """Tests that calibrate=True doesn't crash in binary"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100)


def test_calibrate_binary_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in binary w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100)


def test_calibrate_multiclass(fit_helper):
    """Tests that calibrate=True doesn't crash in multiclass"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
    )
    dataset_name = "covertype_small"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100)


def test_calibrate_multiclass_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in multiclass w/ bagging"""
    fit_args = dict(
        hyperparameters={"GBM": {}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "covertype_small"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100)


def test_calibrate_quantile(fit_helper):
    """Tests that calibrate=True doesn't crash in quantile"""
    fit_args = dict(
        hyperparameters={"RF": {}},
        calibrate=True,
    )
    dataset_name = "ames"
    init_args = dict(problem_type="quantile", quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args, sample_size=100)


def test_calibrate_quantile_bag(fit_helper):
    """Tests that calibrate=True doesn't crash in quantile w/ bagging"""
    fit_args = dict(
        hyperparameters={"RF": {}},
        calibrate=True,
        num_bag_folds=3,
    )
    dataset_name = "ames"
    init_args = dict(problem_type="quantile", quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args, sample_size=100)
