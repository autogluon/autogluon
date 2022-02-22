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


def preset_to_config(model_type: str):
    """
    Get the default config of one predictor in AutoMM.
    Currently, we only use model type to differentiate different predictors.
    In future, we can simultaneously consider model type, data type, optimization type,
    and environment type to construct more diverse presets.

    Parameters
    ----------
    model_type
        Type of model supported by AutoMM.

    Returns
    -------
    A predictor's preset config strings of MODEL, DATA, OPTIMIZATION, and ENVIRONMENT.
    """
    model_type = model_type.lower()
    config = {
        MODEL: "",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    if model_type == "text":
        config[MODEL] = "hf_text"
    elif model_type == "image":
        config[MODEL] = "timm_image"
    else:
        config[MODEL] = model_type

        cur_path = os.path.dirname(os.path.abspath(__file__))
        model_config_dir = os.path.join(cur_path, "configs", "model")
        model_config_path = os.path.join(model_config_dir, f"{model_type}.yaml")
        if not os.path.isfile(model_config_path):
            raise ValueError(
                f"Model type '{model_type}' is not supported yet. "
                f"Consider one of these: {list_model_presets()}"
            )

    return config
