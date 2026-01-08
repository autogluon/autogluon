import copy

from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 10}


def test_rf():
    model_cls = RFModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


def test_rf_compile_onnx():
    model_cls = RFModel
    model_hyperparameters = toy_model_params
    compiler_configs = {RFModel: {"compiler": "onnx"}}

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        compile=True,
        compiler_configs=compiler_configs,
        # RandomForest ONNX compilation not supported for quantile
        problem_types=["binary", "multiclass", "regression"],
        bag=False,  # ONNX compilation fails in bagging with refit_full
    )


def test_rf_binary_compile_onnx_as_ag_arg():
    model_params = copy.deepcopy(toy_model_params)
    model_params["ag.compile"] = {"compiler": "onnx"}

    fit_args = dict(
        hyperparameters={RFModel: model_params},
    )
    dataset_name = "toy_binary"
    FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_rf_binary_compile_onnx_no_config_bagging():
    # FIXME: The below code will crash because:
    #  1. We train with a bag, specifically RandomForest that has efficient OOB and thus only trains 1 fold model
    #  2. We compile RandomForest bag to onnx
    #  3. We call refit_full, performing efficient cloning when fitting refit_full for RandomForest, avoiding fitting again
    #  4. The save of the fold 1 model of RandomForest_BAG_L1 into new location in RandomForest_BAG_L1_FULL does not carry over the model.onnx file
    #  5. Crashes when trying to load the model.onnx file because it doesn't exist.
    # FIXME: This bug only appears if the user calls refit_full on compiled models.
    # Solution: Either,
    #  1. force copy the entire directory of the fold 1 model when cloning instead of calling model.save() -> reuse logic in predictor.clone()
    #  2. model._compiler stores context of files it has created / depends on, and then detects if saving to a new location,
    #     then copies the files to that location or computes them again.
    run_test = False
    if run_test:
        fit_args = dict(
            hyperparameters={RFModel: toy_model_params},
            num_bag_folds=2,
        )
        dataset_name = "toy_binary"
        compiler_configs = "auto"
        FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name, fit_args=fit_args, compile=True, compiler_configs=compiler_configs
        )
