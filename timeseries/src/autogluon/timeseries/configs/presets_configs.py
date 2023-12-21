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
    chronos_mini_ensemble={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-mini"}} | get_default_hps("default"),
    },
    chronos_small_ensemble={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-small"}} | get_default_hps("default"),
    },
    chronos_base_ensemble={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-base"}} | get_default_hps("default"),
    },
    chronos_large_ensemble={
        "hyperparameters": {"Chronos": {"model_path": "amazon/chronos-t5-large", "batch_size": 8}}
        | get_default_hps("default"),
    },
)
