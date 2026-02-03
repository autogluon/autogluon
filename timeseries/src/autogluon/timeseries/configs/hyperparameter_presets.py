from typing import Any


def get_hyperparameter_presets() -> dict[str, dict[str, dict[str, Any] | list[dict[str, Any]]]]:
    return {
        "very_light": {
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {"max_num_samples": 100_000},
            "DirectTabular": {"max_num_samples": 100_000},
        },
        "light": {
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "Chronos2": {"model_path": "autogluon/chronos-2-small"},
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
            "Chronos": {
                "ag_args": {"name_suffix": "WithRegressor"},
                "model_path": "bolt_small",
                "target_scaler": "standard",
                "covariate_regressor": {"model_name": "CAT", "model_hyperparameters": {"iterations": 1000}},
            },
        },
    }
