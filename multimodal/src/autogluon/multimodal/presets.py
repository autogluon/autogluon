import os
from .constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
)
from .registry import Registry

automm_presets = Registry("automm_presets")


@automm_presets.register()
def default():
    return {
        "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    }


@automm_presets.register()
def medium_quality_faster_train():
    return {
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
        "optimization.learning_rate": 4e-4,
    }


@automm_presets.register()
def high_quality():
    return {
        "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    }


@automm_presets.register()
def best_quality():
    return {
        "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
        "env.per_gpu_batch_size": 1,
    }


@automm_presets.register()
def multilingual():
    return {
        "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
        "optimization.top_k": 1,
        "env.precision": "bf16",
        "env.per_gpu_batch_size": 4,
    }


def list_automm_presets(verbose: bool = False):
    """
    List all available presets.

    Returns
    -------
    A list of presets.
    """
    preset_keys = automm_presets.list_keys()
    if not verbose:
        return preset_keys

    preset_details = {}
    for k in preset_keys:
        preset_details[k] = automm_presets.create(k)

    return preset_details


def get_basic_automm_config():
    """
    Get the basic config of AutoMM.

    Returns
    -------
    A dict config with keys: MODEL, DATA, OPTIMIZATION, ENVIRONMENT, and their default values.
    """
    return {
        MODEL: "fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }


def automm_preset_to_config(preset: str):
    """
    Map a AutoMM preset string to its config including a basic config and an overriding dict.

    Parameters
    ----------
    preset
        Name of a preset.

    Returns
    -------
    basic_config
        The basic config of AutoMM.
    overrides
        The hyperparameter overrides of this preset.
    """
    basic_config = get_basic_automm_config()
    preset = preset.lower()
    if preset in automm_presets.list_keys():
        overrides = automm_presets.create(preset)
    else:
        raise ValueError(
            f"Provided preset '{preset}' is not supported. " f"Consider one of these: {automm_presets.list_keys()}"
        )

    return basic_config, overrides
