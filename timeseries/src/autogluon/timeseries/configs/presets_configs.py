"""Preset configurations for autogluon.timeseries Predictors"""

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PRESETS_CONFIGS = dict(
    best_quality={"hyperparameters": "best_quality"},
    high_quality={"hyperparameters": "high_quality"},
    medium_quality={"hyperparameters": "medium_quality"},
    fast_training={"hyperparameters": "fast_training"},
)
