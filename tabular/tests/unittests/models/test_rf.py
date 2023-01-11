
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


def test_rf_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={'RF': {}},
    )
    dataset_name = 'ames'
    init_args = dict(problem_type='quantile', quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)


def test_rf_binary_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'adult'
    compiler_configs = {RFModel: {'compiler': 'onnx'}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)


def test_rf_multiclass_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'covertype_small'
    compiler_configs = {RFModel: {'compiler': 'onnx'}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)


def test_rf_regression_compile_onnx(fit_helper):
    fit_args = dict(
        hyperparameters={RFModel: {}},
    )
    dataset_name = 'ames'
    compiler_configs = {RFModel: {'compiler': 'onnx'}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)


def test_rf_binary_compile_onnx_no_config_bagging(fit_helper):
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
            hyperparameters={RFModel: {}},
            num_bag_folds=2,
        )
        dataset_name = 'adult'
        compiler_configs = "auto"
        fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                            compile_models=True, compiler_configs=compiler_configs)


