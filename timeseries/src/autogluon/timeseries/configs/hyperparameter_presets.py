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
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "Chronos": {"model_path": "bolt_small"},
        },
        "light_inference": {
            "SeasonalNaive": {},
            "DirectTabular": {},
            "RecursiveTabular": {},
            "TemporalFusionTransformer": {},
            "PatchTST": {},
        },
        "default": {
            "SeasonalNaive": {},
            "AutoETS": {},
            "DynamicOptimizedTheta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "Chronos2": [
                {},
                {
                    "ag_args": {"name_suffix": "SmallFineTuned"},
                    "model_path": "autogluon/chronos-2-small",
                    "fine_tune": True,
                    "eval_during_fine_tune": True,
                },
            ],
        },
    }
