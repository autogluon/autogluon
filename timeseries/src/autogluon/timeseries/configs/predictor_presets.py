"""Preset configurations for autogluon.timeseries Predictors"""

from . import HYPERPARAMETER_PRESETS

# TODO: change default HPO settings when other HPO strategies (e.g., Ray tune) are available
# TODO: add refit_full arguments once refitting is available

TIMESERIES_PREDICTOR_PRESETS = dict(
    best_quality={"hyperparameters": "default", "num_val_windows": 2},
    high_quality={"hyperparameters": "default"},
    medium_quality={"hyperparameters": "light"},
    fast_training={"hyperparameters": "very_light"},
    # Chronos-Bolt models
    bolt_tiny={
        "hyperparameters": {"Chronos": {"model_path": "bolt_tiny"}},
        "skip_model_selection": True,
    },
    bolt_mini={
        "hyperparameters": {"Chronos": {"model_path": "bolt_mini"}},
        "skip_model_selection": True,
    },
    bolt_small={
        "hyperparameters": {"Chronos": {"model_path": "bolt_small"}},
        "skip_model_selection": True,
    },
    bolt_base={
        "hyperparameters": {"Chronos": {"model_path": "bolt_base"}},
        "skip_model_selection": True,
    },
    # Original Chronos models
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
            **HYPERPARAMETER_PRESETS["light_inference"],
        }
    },
    chronos_large_ensemble={
        "hyperparameters": {
            "Chronos": {"model_path": "large", "batch_size": 8},
            **HYPERPARAMETER_PRESETS["light_inference"],
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
TIMESERIES_PREDICTOR_PRESETS = {
    **TIMESERIES_PREDICTOR_PRESETS,
    **{k: TIMESERIES_PREDICTOR_PRESETS[v].copy() for k, v in TIMESERIES_PRESETS_ALIASES.items()},
}
