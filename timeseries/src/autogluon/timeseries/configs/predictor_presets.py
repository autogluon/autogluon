"""Preset configurations for autogluon.timeseries Predictors"""

from typing import Any

TIMESERIES_PRESETS_ALIASES = dict(
    best="best_quality",
    high="high_quality",
    medium="medium_quality",
    bq="best_quality",
    hq="high_quality",
    mq="medium_quality",
)


def get_predictor_presets() -> dict[str, Any]:
    predictor_presets = dict(
        # TODO: Change names before merging
        new1={
            "hyperparameters": "default",
            "num_val_windows": "auto",
            "refit_every_n_windows": "auto",
            "ensemble_hyperparameters": [
                # Layer 1
                {"Weighted": {}, "Median": {}, "LinearStacker": {"weights_per": "mq"}, "Tabular": {}},
                # Layer 2
                {"Weighted": {}},
            ],
        },
        new2={
            "hyperparameters": "default",
            "num_val_windows": "auto",
            "refit_every_n_windows": "auto",
            "ensemble_hyperparameters": [
                # Layer 1
                {"Weighted": {}, "Median": {}, "LinearStacker": {"weights_per": "mq"}},
                # Layer 2
                {"Weighted": {}},
            ],
        },
        new3={
            "hyperparameters": "default",
            "num_val_windows": (4, 1),
            "refit_every_n_windows": "auto",
            "ensemble_hyperparameters": [
                # Layer 1
                {"Weighted": {}, "Median": {}, "LinearStacker": {"weights_per": "mq"}, "Tabular": {}},
                # Layer 2
                {"Weighted": {}},
            ],
        },
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
    )

    # update with aliases
    predictor_presets = {
        **predictor_presets,
        **{k: predictor_presets[v].copy() for k, v in TIMESERIES_PRESETS_ALIASES.items()},
    }

    return predictor_presets
