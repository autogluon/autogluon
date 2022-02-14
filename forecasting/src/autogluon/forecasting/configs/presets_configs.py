"""Preset configurations for autogluon.forecasting Predictors"""
# TODO: change default HPO settings when other HPO strategies are available
forecasting_presets_configs = dict(
    best_quality={
        "hyperparameters": "default_hpo",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 20,
        },
        "refit_full": True,
    },
    high_quality={
        "hyperparameters": "default_hpo",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 10,
        },
        "refit_full": True,
    },
    good_quality={
        "hyperparameters": "default_hpo",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 2,
        },
        "refit_full": True,
    },
    medium_quality={"hyperparameters": "default", "refit_full": True},
    low_quality={"hyperparameters": "toy"},
    low_quality_hpo={
        "hyperparameters": "toy_hpo",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 2,
        },
        "refit_full": True,
    },
)
