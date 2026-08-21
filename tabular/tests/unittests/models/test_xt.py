from autogluon.tabular.models.xt.xt_model import XTModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 10}


def test_xt():
    model_cls = XTModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


def test_xt_binary_compile_onnx():
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    dataset_name = "toy_binary"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )


def test_xt_multiclass_compile_onnx():
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    dataset_name = "toy_multiclass"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )


def test_xt_regression_compile_onnx():
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    dataset_name = "toy_regression"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
    )
