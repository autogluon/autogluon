
from autogluon.tabular.models import LGBModel, TabularNeuralNetTorchModel


def test_cascade_binary(fit_helper):
    fit_args = dict(
        hyperparameters={
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'adult'
    cascade = ['LightGBM', 'WeightedEnsemble_L2']
    fit_helper.fit_and_validate_dataset_with_cascade(dataset_name=dataset_name, fit_args=fit_args, cascade=cascade, model_count=2)


def test_cascade_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'covertype_small'
    cascade = ['LightGBM', 'WeightedEnsemble_L2']
    fit_helper.fit_and_validate_dataset_with_cascade(dataset_name=dataset_name, fit_args=fit_args, cascade=cascade, model_count=2)
