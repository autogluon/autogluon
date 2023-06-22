from autogluon.tabular.models.catboost.catboost_model import CatBoostModel


def test_catboost_binary(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
    )
    dataset_name = "adult"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_catboost_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
    )
    dataset_name = "covertype_small"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_catboost_regression(fit_helper):
    fit_args = dict(
        hyperparameters={CatBoostModel: {}},
        time_limit=10,  # CatBoost trains for a very long time on ames (many iterations)
    )
    dataset_name = "ames"
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_catboost_quantile(fit_helper):
    fit_args = dict(
        hyperparameters={"CAT": {}},
        time_limit=10,  # CatBoost trains for a very long time on ames (many iterations)
    )
    dataset_name = "ames"
    init_args = dict(problem_type="quantile", quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, init_args=init_args)
