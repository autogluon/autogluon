from typing import Any, Union


def get_hyperparameter_presets() -> dict[str, dict[str, Union[dict[str, Any], list[dict[str, Any]]]]]:
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
            "NPTS": {},
            "DynamicOptimizedTheta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "PatchTST": {},
            "DeepAR": {},
            "Chronos": [
                {
                    "ag_args": {"name_suffix": "ZeroShot"},
                    "model_path": "bolt_base",
                },
                {
                    "ag_args": {"name_suffix": "FineTuned"},
                    "model_path": "bolt_small",
                    "fine_tune": True,
                    "target_scaler": "standard",
                    "covariate_regressor": {"model_name": "CAT", "model_hyperparameters": {"iterations": 1_000}},
                },
            ],
            "TiDE": {
                "encoder_hidden_dim": 256,
                "decoder_hidden_dim": 256,
                "temporal_hidden_dim": 64,
                "num_batches_per_epoch": 100,
                "lr": 1e-4,
            },
        },
    }
