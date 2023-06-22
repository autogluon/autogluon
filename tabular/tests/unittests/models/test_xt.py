from autogluon.tabular.models.xt.xt_model import XTModel


def test_xt_binary(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_xt_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_xt_regression(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_xt_binary_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    dataset_name = "adult"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile_models=True, compiler_configs=compiler_configs)


def test_xt_multiclass_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    dataset_name = "covertype_small"
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile_models=True, compiler_configs=compiler_configs)


def test_xt_regression_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={XTModel: {}},
    )
    compiler_configs = {XTModel: {"compiler": "onnx"}}
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, compile_models=True, compiler_configs=compiler_configs)
