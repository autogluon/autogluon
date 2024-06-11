

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
