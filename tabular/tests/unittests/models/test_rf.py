
from autogluon.tabular.models.rf.rf_model import RFModel


# TODO: Consider adding post-test dataset cleanup (not for each test, since they reuse the datasets)
def test_rf_binary(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_rf_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_rf_regression(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
