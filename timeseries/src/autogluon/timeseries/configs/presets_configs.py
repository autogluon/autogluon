"""Preset configurations for autogluon.timeseries Predictors"""

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PRESETS_CONFIGS = dict(
    best_quality={
        "hyperparameters": "best_quality",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "auto",
            "num_trials": 10,
        },
    },
    high_quality={
        "hyperparameters": "high_quality",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "auto",
            "num_trials": 5,
        },
    },
    medium_quality={"hyperparameters": "medium_quality"},
    fast_training={"hyperparameters": "local_only"},
)
