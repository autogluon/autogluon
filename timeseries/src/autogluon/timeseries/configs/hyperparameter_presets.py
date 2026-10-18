from typing import Any


def get_hyperparameter_presets() -> dict[str, dict[str, dict[str, Any] | list[dict[str, Any]]]]:
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
        "default": {
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
        },
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
