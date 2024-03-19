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
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-tiny", "skip_validation": True}},
        "disable_length_check": True,
    },
    chronos_mini={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-mini", "skip_validation": True}},
        "disable_length_check": True,
    },
    chronos_small={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-small", "skip_validation": True}},
        "disable_length_check": True,
    },
    chronos_base={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-base", "skip_validation": True}},
        "disable_length_check": True,
    },
    chronos_large={
        "hyperparameters": {
            "Chronos": {"model_path": "amazon/chronos-t5-large", "skip_validation": True, "batch_size": 8}
        },
        "disable_length_check": True,
    },
    chronos_ensemble={
        "hyperparameters": {
            **{"Chronos": {"model_path": "amazon/chronos-t5-small"}},
            **get_default_hps("default"),
        }
    },
    chronos_large_ensemble={
        "hyperparameters": {
            **{"Chronos": {"model_path": "amazon/chronos-t5-large", "batch_size": 8}},
            **get_default_hps("default"),
        }
    },
)

timeseries_presets_aliases = dict(
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
    **{k: TIMESERIES_PRESETS_CONFIGS[v].copy() for k, v in timeseries_presets_aliases.items()},
}
