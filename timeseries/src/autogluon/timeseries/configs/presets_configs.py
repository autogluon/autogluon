"""Preset configurations for autogluon.timeseries Predictors"""

from autogluon.timeseries.models.presets import get_default_hps

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PRESETS_CONFIGS = dict(
    best_quality={"hyperparameters": "default", "num_val_windows": 2},
    high_quality={"hyperparameters": "default"},
    medium_quality={"hyperparameters": "light"},
    fast_training={"hyperparameters": "very_light"},
    chronos_tiny={
        "hyperparameters": {"Chronos": {"model_path": "tiny"}},
        "skip_model_selection": True,
    },
    chronos_mini={
        "hyperparameters": {"Chronos": {"model_path": "mini"}},
        "skip_model_selection": True,
    },
    chronos_small={
        "hyperparameters": {"Chronos": {"model_path": "small"}},
        "skip_model_selection": True,
    },
    chronos_base={
        "hyperparameters": {"Chronos": {"model_path": "base"}},
        "skip_model_selection": True,
    },
    chronos_large={
        "hyperparameters": {"Chronos": {"model_path": "large", "batch_size": 8}},
        "skip_model_selection": True,
    },
    chronos_ensemble={
        "hyperparameters": {
            "Chronos": {"model_path": "small"},
            **get_default_hps("default"),
        }
    },
    chronos_large_ensemble={
        "hyperparameters": {
            "Chronos": {"model_path": "large", "batch_size": 8},
            **get_default_hps("default"),
        }
    },
)

TIMESERIES_PRESETS_ALIASES = dict(
    chronos="chronos_small",
    best="best_quality",
    high="high_quality",
    medium="medium_quality",
    bq="best_quality",
    hq="high_quality",
    mq="medium_quality",
)

# update with aliases
TIMESERIES_PRESETS_CONFIGS = {
    **TIMESERIES_PRESETS_CONFIGS,
    **{k: TIMESERIES_PRESETS_CONFIGS[v].copy() for k, v in TIMESERIES_PRESETS_ALIASES.items()},
}
