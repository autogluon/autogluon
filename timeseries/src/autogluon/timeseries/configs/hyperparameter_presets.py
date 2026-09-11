from typing import Any


def get_hyperparameter_presets() -> dict[str, dict[str, dict[str, Any] | list[dict[str, Any]]]]:
    default = {
        "SeasonalNaive": {},
        "AutoETS": {},
        "DynamicOptimizedTheta": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
        "TemporalFusionTransformer": {},
        "DeepAR": {},
        "Chronos2": [
            {},
            {
                "ag_args": {"name_suffix": "SmallFineTuned"},
                "model_path": "autogluon/chronos-2-small",
                "fine_tune": True,
                "eval_during_fine_tune": True,
            },
        ],
        "Toto2": {"model_path": "Toto-2.0-22m"},
    }

    # Import lazily so regular AutoGluon presets do not acquire a backend import.
    from autogluon.timeseries.models.neuralforecast import NEURALFORECAST_MODELS

    default_all_nf = {
        **default,
        **{f"NF{model_name}": {} for model_name in NEURALFORECAST_MODELS},
    }

    return {
        "light": {
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "Chronos2": {"model_path": "autogluon/chronos-2-small"},
            "Toto2": {"model_path": "Toto-2.0-4m"},
        },
        "default": default,
        "default_all_nf": default_all_nf,
        "experimental": {
            "Chronos2": [
                {},
                {
                    "ag_args": {"name_suffix": "SmallFineTuned"},
                    "model_path": "autogluon/chronos-2-small",
                    "fine_tune": True,
                    "eval_during_fine_tune": True,
                },
            ],
            "Toto2": {"model_path": "Toto-2.0-313m"},
        },
    }
