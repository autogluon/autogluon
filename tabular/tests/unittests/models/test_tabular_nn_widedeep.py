def test_tabular_nn_widedeep_binary(fit_helper):
    mlp_model_args = {'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True}
    fit_args = dict(
        hyperparameters={'WIDEDEEPNN': {'type': 'tabmlp',  'ag_args': {'name_suffix': 'TabMLP'}, 'model_args': mlp_model_args}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_widedeep_multiclass(fit_helper):
    mlp_model_args = {'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True}
    fit_args = dict(
        hyperparameters={'WIDEDEEPNN': {'type': 'tabmlp',  'ag_args': {'name_suffix': 'TabMLP'}, 'model_args': mlp_model_args}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_widedeep_regression(fit_helper):
    mlp_model_args = {'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True}
    fit_args = dict(
        # hyperparameters={'FASTAI': {}},
        hyperparameters={'WIDEDEEPNN': {'type': 'tabresnet',  'ag_args': {'name_suffix': 'TabMLP'}, 'model_args': mlp_model_args}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


