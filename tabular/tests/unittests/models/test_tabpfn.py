from autogluon.tabular.models.tabpfn.tabpfn_model import TabPFNModel


def test_tabpfn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={TabPFNModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabpfn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={TabPFNModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
