import os
from .constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
)


def list_model_presets():
    """
    List all available model types.
    Image/text backbones can be customized for one model type.

    Returns
    -------
    A list of model types.
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_config_dir = os.path.join(cur_path, "configs", "model")
    model_config_files = [f for f in os.listdir(model_config_dir) if f.endswith((".yaml", ".yml"))]
    model_presets = [f.split(".")[0] for f in model_config_files]
    return model_presets


def get_preset(model_preset: str):
    """
    Get the preset of one predictor in AutoMM.
    Currently, we only use model presets to differentiate different predictors.
    In future, we can simultaneously consider model, data, optimization,
    and environment to construct more diverse presets.

    Parameters
    ----------
    model_preset
        A model preset supported by AutoMM.

    Returns
    -------
    AutoMM predictor's presets of MODEL, DATA, OPTIMIZATION, and ENVIRONMENT.
    """
    model_preset = model_preset.lower()
    preset = {
        MODEL: model_preset,
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_config_dir = os.path.join(cur_path, "configs", "model")
    model_config_path = os.path.join(model_config_dir, f"{model_preset}.yaml")
    if not os.path.isfile(model_config_path):
        raise ValueError(
            f"Model preset '{model_preset}' is not supported yet. Consider one of these: {list_model_presets()}"
        )

    return preset
