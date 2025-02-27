from autogluon.tabular.models.xt.xt_model import XTModel

toy_model_params = {"n_estimators": 10}


def test_xt(fit_helper):
    model_cls = XTModel
    model_hyperparameters = toy_model_params

    fit_helper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, bag="first", refit_full="first")


def test_xt_binary_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    dataset_name = "toy_binary"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)


def test_xt_multiclass_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    dataset_name = "toy_multiclass"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)


def test_xt_regression_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: toy_model_params},
    )
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    dataset_name = "toy_regression"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs)
