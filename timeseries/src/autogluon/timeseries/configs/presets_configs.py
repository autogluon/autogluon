"""Preset configurations for autogluon.timeseries Predictors"""

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PRESETS_CONFIGS = dict(
    best_quality={"hyperparameters": "default", "num_val_windows": 2},
    high_quality={"hyperparameters": "default"},
    medium_quality={"hyperparameters": "light"},
    fast_training={"hyperparameters": "very_light"},
)
