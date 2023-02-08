class QuickFitDefaults:
    DEFAULT_RF_CONFIG = {
        "RF": [
            {
                "criterion": "entropy",
                "max_depth": 15,
                "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]},
            },
            {
                "criterion": "squared_error",
                "max_depth": 15,
                "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]},
            },
        ],
    }
    DEFAULT_LGBM_CONFIG = {
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        ]
    }
