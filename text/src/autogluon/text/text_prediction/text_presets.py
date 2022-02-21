from ..automm.presets import preset_to_config


def list_text_presets():
    """
    Returns
    -------
    A list of simple presets available in AutoGluon-Text.
    """
    simple_presets = [
        "default", "medium_quality_faster_train",
        "high_quality", "best_quality", "multilingual",
    ]
    return simple_presets


def text_preset_to_config(preset: str):
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
    config = preset_to_config(model_type)
    preset = preset.lower()
    if preset == "default" or preset == "high_quality":
        overrides = {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        }
    elif preset == "medium_quality_faster_train":
        overrides = {
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "optimization.learning_rate": 4e-4,
        }
    elif preset == "best_quality":
        overrides = {
            "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
            "env.per_gpu_batch_size": 2,
        }
    elif preset == "multilingual":
        overrides = {
            "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
            "env.precision": "bf16",
            "env.per_gpu_batch_size": 2,
        }
    else:
        raise ValueError(
            f"Provided preset '{preset}' is not supported. "
            f"Consider one of these: {list_text_presets()}"
        )

    return config, overrides
