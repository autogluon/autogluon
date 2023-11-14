import pytest

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel


def test_tabular_nn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_regression(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        time_limit=20,  # TabularNN trains for a long time on ames
    )
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


# Testing with good_quality to ensure tabularNN work well with ParallelLocalFoldFittingStrategy
def test_tabular_nn_binary_good_quality(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        presets="good_quality",
        num_bag_folds=2,
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=4, refit_full=False)


def test_tabular_nn_binary_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = "adult"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer)


def test_tabular_nn_multiclass_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = "covertype_small"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer)


def test_tabular_nn_regression_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        time_limit=20,  # TabularNN trains for a long time on ames
    )
    dataset_name = "ames"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer)
