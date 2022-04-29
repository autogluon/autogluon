def test_tabular_nn_widedeep_binary(fit_helper):
    fit_helper.fit_and_validate_dataset(dataset_name='adult', fit_args=dict(hyperparameters={'WD_TAB_MLP': {}}))


def test_tabular_nn_widedeep_multiclass(fit_helper):
    fit_helper.fit_and_validate_dataset(dataset_name='covertype_small', fit_args=dict(hyperparameters={'WD_TAB_MLP': {}}))


def test_tabular_nn_widedeep_regression(fit_helper):
    fit_helper.fit_and_validate_dataset(dataset_name='ames', fit_args=dict(hyperparameters={'WD_TAB_MLP': {}}))
