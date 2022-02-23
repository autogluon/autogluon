from ..automm.presets import preset_to_config


def list_text_presets(verbose=False):
    """
    Returns
    -------
    If verbose==True, return all the preset strings and their corresponding config customizations.
    If verbose==False, return a list of simple presets strings.
    """
    simple_presets = {
        "default": {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        },
        "medium_quality_faster_train": {
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "optimization.learning_rate": 4e-4,
        },
        "high_quality": {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        },
        "best_quality": {
            "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
            "env.per_gpu_batch_size": 2,
        },
        "multilingual": {
            "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
            "env.precision": "bf16",
            "env.per_gpu_batch_size": 2,
        },
    }

    if verbose:
        return simple_presets
    else:
        return list(simple_presets.keys())


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
    available_presets = list_text_presets(verbose=True)

    if preset in available_presets:
        overrides = available_presets[preset]
    else:
        raise ValueError(
            f"Provided preset '{preset}' is not supported. "
            f"Consider one of these: {list_text_presets()}"
        )

    return config, overrides
