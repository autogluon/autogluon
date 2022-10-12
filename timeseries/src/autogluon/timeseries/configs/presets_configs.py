"""Preset configurations for autogluon.timeseries Predictors"""

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PRESETS_CONFIGS = dict(
    best_quality={
        "hyperparameters": "default_hpo",
        "hyperparameter_tune_kwargs": {
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 20,
        },
    },
    medium_quality={"hyperparameters": "default"},
    fast_training={"hyperparameters": "local_only"},
    # TODO: Remove deprecated presets below
    high_quality={"hyperparameters": "default"},
    good_quality={"hyperparameters": "default"},
    low_quality={"hyperparameters": "local_only"},
)
