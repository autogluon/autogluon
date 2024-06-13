def is_advanced_hyperparameter_format(hyperparameters: dict) -> bool:
    """
    Returns True if hyperparameters are stack-level formatted.

    Simple formatting example:
    {
        "GBM": [...],
        "CAT": [...],
    }

    Stack-level formatting example:
    {
        1: {"GBM": [...], "CAT": [...]},
        2: {"XGB": [...], "FASTAI": [...]},
    }
    """
    assert isinstance(hyperparameters, dict), f"`hyperparameters` must be a dict, but found: {type(hyperparameters)}"
    advanced_hyperparameter_format = None
    for key in hyperparameters:
        if isinstance(key, int) or (isinstance(key, str) and key == "default"):
            if advanced_hyperparameter_format is None:
                advanced_hyperparameter_format = True
            if not advanced_hyperparameter_format:
                raise ValueError(f"Invalid `hyperparameters` format:\n{hyperparameters}")
        else:
            if advanced_hyperparameter_format is None:
                advanced_hyperparameter_format = False
            if advanced_hyperparameter_format:
                raise ValueError(f"Invalid `hyperparameters` format:\n{hyperparameters}")

    if advanced_hyperparameter_format is None:
        advanced_hyperparameter_format = False
    return advanced_hyperparameter_format


def get_hyperparameter_str_deprecation_msg() -> str:
    extra_msg = (
        f"Attempted to specify 'GBMLarge' model preset in hyperparameters. "
        f"Support for hyperparameter shorthands via strings is deprecated and will raise an exception starting in `autogluon==1.3`"
        f"\nTo avoid this, you can still train the desired 'GBMLarge' model "
        f"by editing your hyperparameters dictionary, replacing 'GBMLarge' with the following:\n"
        """{
    "learning_rate": 0.03,
    "num_leaves": 128,
    "feature_fraction": 0.9,
    "min_data_in_leaf": 3,
    "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
}"""
    )
    return extra_msg


def get_deprecated_lightgbm_large_hyperparameters() -> dict:
    return {
        "learning_rate": 0.03,
        "num_leaves": 128,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 3,
        "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
    }
