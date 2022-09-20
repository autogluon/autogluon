from typing import Optional

from .constants import DATA, DISTILLER, ENVIRONMENT, MODEL, OPTIMIZATION
from .registry import Registry

automm_presets = Registry("automm_presets")


@automm_presets.register()
def default():
    return {
        "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    }


@automm_presets.register()
def medium_quality_faster_train():
    return {
        "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
        "optimization.learning_rate": 4e-4,
    }


@automm_presets.register()
def high_quality():
    return {
        "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    }


@automm_presets.register()
def best_quality():
    return {
        "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
        "env.per_gpu_batch_size": 1,
    }


@automm_presets.register()
def multilingual():
    return {
        "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
        "optimization.top_k": 1,
        "env.precision": "bf16",
        "env.per_gpu_batch_size": 4,
    }


@automm_presets.register()
def few_shot_text_classification():
    return {
        "model.names": ["t_few"],
        "model.t_few.checkpoint_name": "bigscience/T0_3B",
        "model.t_few.gradient_checkpointing": False,  # Currently instable.
        "optimization.learning_rate": 3e-3,
        "optimization.lr_decay": 1.0,
        "optimization.efficient_finetune": "ia3",
        "optimization.max_steps": 1000,  # Find better solution to train for long
        "optimization.check_val_every_n_epoch": 10,  # Might need adjustment
        "optimization.val_check_interval": 1.0,
        "optimization.top_k_average_method": "best",
        "optimization.warmup_steps": 0.06,
        "optimization.lora.module_filter": [".*SelfAttention|.*EncDecAttention|.*DenseReluDense"],
        "optimization.lora.filter": ["k|v|wi_1.*"],
        "optimization.top_k": 1,
        "optimization.max_epochs": -1,
        "env.batch_size": 8,
        "env.per_gpu_batch_size": 1,
        "env.precision": "bf16",
        "data.templates.turn_on": True,
    }


@automm_presets.register()
def zero_shot_classification():
    return {
        "model.names": ["hf_text"],
        "model.hf_text.checkpoint_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "env.eval_batch_size_ratio": 1,
    }


@automm_presets.register()
def zero_shot_image_classification():
    return {
        "model.names": ["clip"],
        "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
        "model.clip.max_text_len": 0,
        "env.eval_batch_size_ratio": 1,
    }


@automm_presets.register()
def object_detection():
    return {
        "model.names": ["mmdet_image"],
        "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
        "env.eval_batch_size_ratio": 1,
        "env.precision": 32,
    }


@automm_presets.register()
def ocr_text_detection():
    return {
        "model.names": ["mmocr_text_detection"],
        "model.mmocr_text_detection.checkpoint_name": "TextSnake",
        "env.eval_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }


@automm_presets.register()
def ocr_text_recognition():
    return {
        "model.names": ["mmocr_text_recognition"],
        "model.mmocr_text_recognition.checkpoint_name": "ABINet",
        "env.eval_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }


@automm_presets.register()
def feature_extraction():
    return {
        "model.names": ["hf_text"],
        "model.hf_text.checkpoint_name": "sentence-transformers/msmarco-MiniLM-L-12-v3",
        "model.hf_text.pooling_mode": "mean",
        "env.eval_batch_size_ratio": 1,
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


def get_basic_automm_config(is_distill: Optional[bool] = False):
    """
    Get the basic config of AutoMM.

    Parameters
    ----------
    is_distill
        Whether in the distillation mode.

    Returns
    -------
    A dict config with keys: MODEL, DATA, OPTIMIZATION, ENVIRONMENT, and their default values.
    """
    config = {
        MODEL: "fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    if is_distill:
        config[DISTILLER] = "default"

    return config


def get_automm_presets(presets: str):
    """
    Map a AutoMM preset string to its config including a basic config and an overriding dict.

    Parameters
    ----------
    presets
        Name of a preset.

    Returns
    -------
    basic_config
        The basic config of AutoMM.
    overrides
        The hyperparameter overrides of this preset.
    """
    presets = presets.lower()
    if presets in automm_presets.list_keys():
        overrides = automm_presets.create(presets)
    else:
        raise ValueError(
            f"Provided preset '{presets}' is not supported. " f"Consider one of these: {automm_presets.list_keys()}"
        )

    return overrides
