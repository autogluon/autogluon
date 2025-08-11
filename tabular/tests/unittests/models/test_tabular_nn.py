import copy
import shutil

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"num_epochs": 3}


def test_tabular_nn():
    model_cls = TabularNeuralNetTorchModel
    model_hyperparameters = copy.deepcopy(toy_model_params)
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


def test_tabular_nn_binary_compile_onnx():
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: toy_model_params},
    )
    dataset_name = "toy_binary"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(
        predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer
    )


def test_tabular_nn_binary_compile_onnx_as_ag_arg():
    model_params = {"ag.compile": {"compiler": "onnx"}}
    model_params.update(toy_model_params)
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: model_params},
    )
    dataset_name = "toy_binary"
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, refit_full=True, delete_directory=False
    )
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(
        predictor._learner.trainer.load_model("NeuralNetTorch").processor, TabularNeuralNetTorchOnnxTransformer
    )
    assert isinstance(
        predictor._learner.trainer.load_model("NeuralNetTorch_FULL").processor, TabularNeuralNetTorchOnnxTransformer
    )
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_tabular_nn_multiclass_compile_onnx():
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: toy_model_params},
    )
    dataset_name = "toy_multiclass"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(
        predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer
    )


def test_tabular_nn_regression_compile_onnx():
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: toy_model_params},
    )
    dataset_name = "toy_regression"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(
        predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer
    )
