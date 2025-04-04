import shutil

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


# Testing with bagging to ensure tabularNN work well with ParallelLocalFoldFittingStrategy
def test_tabular_nn_binary_bagging(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        num_bag_folds=2,
        num_bag_sets=1,
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, expected_model_count=2, refit_full=False)


def test_tabular_nn_binary_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = "adult"
    compiler_configs = {TabularNeuralNetTorchModel: {"compiler": "onnx"}}
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(predictor._learner.trainer.models["NeuralNetTorch"].processor, TabularNeuralNetTorchOnnxTransformer)


def test_tabular_nn_binary_compile_onnx_as_ag_arg(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {"ag.compile": {"compiler": "onnx"}}},
    )
    dataset_name = "adult"
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, refit_full=True, delete_directory=False)
    from autogluon.tabular.models.tabular_nn.compilers.onnx import TabularNeuralNetTorchOnnxTransformer

    assert isinstance(predictor._learner.trainer.load_model("NeuralNetTorch").processor, TabularNeuralNetTorchOnnxTransformer)
    assert isinstance(predictor._learner.trainer.load_model("NeuralNetTorch_FULL").processor, TabularNeuralNetTorchOnnxTransformer)
    shutil.rmtree(predictor.path, ignore_errors=True)


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
