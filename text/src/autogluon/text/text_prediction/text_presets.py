from ..automm.presets import get_preset


def list_presets():
    """
    Returns
    -------
    A list of naive presets available in AutoGluon-Text.
    """
    simple_presets = [
        "default", "lower_quality_fast_train",
        "medium_quality_faster_train", "best_quality",
    ]
    return simple_presets


def preset_to_config(preset: str):
    """
    Convert a preset string to AutoMM's config.

    Parameters
    ----------
    preset
        A simple preset string, which should be covered by list_presets().

    Returns
    -------
    config
        A config dictionary of AutoMM.
    overrides
        A dictionary with customized backbones.
    """
    model_type = "fusion_mlp_text_tabular"
    config = get_preset(model_type)
    preset = preset.lower()
    if preset == "default" or preset == "medium_quality_faster_train":
        overrides = {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        }
    elif preset == "lower_quality_fast_train":
        overrides = {
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        }
    elif preset == "best_quality":
        overrides = {
            "model.hf_text.checkpoint_name": "google/electra-large-discriminator",
        }
    else:
        raise ValueError(
            f"Provided preset '{preset}' is not supported. "
            f"Consider one of these: {list_presets()}"
        )

    return config, overrides
