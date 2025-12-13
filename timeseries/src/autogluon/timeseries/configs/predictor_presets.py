"""Preset configurations for autogluon.timeseries Predictors"""

from typing import Any

from . import get_hyperparameter_presets

TIMESERIES_PRESETS_ALIASES = dict(
    chronos="chronos_small",
    best="best_quality",
    high="high_quality",
    medium="medium_quality",
    bq="best_quality",
    hq="high_quality",
    mq="medium_quality",
)


def get_predictor_presets() -> dict[str, Any]:
    hp_presets = get_hyperparameter_presets()

    predictor_presets = dict(
        best_quality={"hyperparameters": "default", "num_val_windows": 2},
        high_quality={"hyperparameters": "default"},
        medium_quality={"hyperparameters": "light"},
        fast_training={"hyperparameters": "very_light"},
        # Chronos-2 models
        chronos2={
            "hyperparameters": {"Chronos2": {"model_path": "autogluon/chronos-2"}},
            "skip_model_selection": True,
        },
        chronos2_small={
            "hyperparameters": {"Chronos2": {"model_path": "autogluon/chronos-2-small"}},
            "skip_model_selection": True,
        },
        chronos2_ensemble={
            "hyperparameters": {
                "Chronos2": [
                    {"model_path": "autogluon/chronos-2", "ag_args": {"name_suffix": "ZeroShot"}},
                    {
                        "model_path": "autogluon/chronos-2-small",
                        "fine_tune": True,
                        "eval_during_fine_tune": True,
                        "ag_args": {"name_suffix": "SmallFineTuned"},
                    },
                ]
            },
        },
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
                **hp_presets["light_inference"],
            }
        },
        chronos_large_ensemble={
            "hyperparameters": {
                "Chronos": {"model_path": "large", "batch_size": 8},
                **hp_presets["light_inference"],
            }
        },
    )

    # update with aliases
    predictor_presets = {
        **predictor_presets,
        **{k: predictor_presets[v].copy() for k, v in TIMESERIES_PRESETS_ALIASES.items()},
    }

    return predictor_presets
