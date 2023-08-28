def test_stack_info_leak_protection_binary(fit_helper):
    """Tests that stack_info_leak_protection=True doesn't crash in binary"""
    fit_args = dict(
        hyperparameters={"GBM": {}, "RF": {}},
        ag_args_fit=dict(stack_info_leak_protection=True),
        num_stack_levels=1,
        num_bag_folds=3,
    )
    dataset_name = "adult"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100, expected_model_count=7, refit_full=False)


def test_stack_info_leak_protection_multiclass(fit_helper):
    """Tests that stack_info_leak_protection=True doesn't crash in multiclass"""
    fit_args = dict(
        hyperparameters={"GBM": {}, "RF": {}},
        ag_args_fit=dict(stack_info_leak_protection=True),
        num_stack_levels=1,
        num_bag_folds=3,
    )
    dataset_name = "covertype_small"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100, expected_model_count=7, refit_full=False)


def test_stack_info_leak_protection_regression(fit_helper):
    """Tests that stack_info_leak_protection=True doesn't crash in regression"""
    fit_args = dict(
        hyperparameters={"GBM": {}, "RF": {}},
        ag_args_fit=dict(stack_info_leak_protection=True),
        num_stack_levels=1,
        num_bag_folds=3,
    )
    dataset_name = "ames"

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100, expected_model_count=7, refit_full=False)


# TODO: support this?
# def test_stack_info_leak_protection_quantile(fit_helper):
#    pass
