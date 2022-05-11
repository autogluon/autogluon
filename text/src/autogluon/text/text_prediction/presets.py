from ..automm.presets import get_preset


def list_text_presets(verbose=False):
    """
    Returns
    -------
    If verbose==True, return all the preset strings and their corresponding config customizations.
    If verbose==False, return a list of presets strings.
    """
    simple_presets = {
        "default": {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
            "optimization.lr_decay": 0.9,
        },
        "medium_quality_faster_train": {
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "optimization.learning_rate": 4e-4,
            "optimization.lr_decay": 0.9,
        },
        # TODO(?) Revise to use registry
        "high_quality": {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
            "optimization.lr_decay": 0.9,
        },
        "best_quality": {
            "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
            "optimization.lr_decay": 0.9,
            "env.per_gpu_batch_size": 2,
        },
        "multilingual": {
            "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
            "optimization.top_k": 1,
            "optimization.lr_decay": 0.9,
            "env.precision": 'bf16',
            "env.per_gpu_batch_size": 4,
        },
    }

    if verbose:
        return simple_presets
    else:
        return list(simple_presets.keys())


def get_text_preset(preset: str):
    """
    Convert text predictor's preset to AutoMM's preset.

    Parameters
    ----------
    preset
        A simple preset string, which should be covered by list_presets().

    Returns
    -------
    automm_preset
        AutoMM's preset.
    overrides
        A dictionary of overriding configs.
    """
    automm_preset = get_preset("fusion_mlp_image_text_tabular")
    overrides = {"model.names": ["hf_text", "numerical_mlp", "categorical_mlp", "fusion_mlp"]}
    preset = preset.lower()
    available_presets = list_text_presets(verbose=True)

    if preset in available_presets:
        overrides.update(available_presets[preset])
    else:
        raise ValueError(
            f"Provided preset '{preset}' is not supported. "
            f"Consider one of these: {list_text_presets()}"
        )

    return automm_preset, overrides
