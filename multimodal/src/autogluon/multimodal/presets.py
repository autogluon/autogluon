from typing import List, Optional

from .constants import (
    BEST_QUALITY,
    BINARY,
    DATA,
    DEFAULT,
    ENVIRONMENT,
    HIGH_QUALITY,
    MEDIUM_QUALITY,
    MODEL,
    MULTICLASS,
    OPTIMIZATION,
    REGRESSION,
)
from .registry import Registry

automm_presets = Registry("automm_presets")
matcher_presets = Registry("matcher_presets")


@automm_presets.register()
def default(presets: str = DEFAULT):
    """
    Register the presets for problem types: binary, multiclass, classification, and regression.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": [
            "categorical_mlp",
            "numerical_mlp",
            "timm_image",
            "hf_text",
            "fusion_mlp",
        ],
        "env.num_workers": 2,
    }
    hyperparameter_tune_kwargs = None

    if presets in [HIGH_QUALITY, DEFAULT]:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
                "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
            }
        )
    elif presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                "model.timm_image.checkpoint_name": "mobilenetv3_large_100",
                "optimization.learning_rate": 4e-4,
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
                "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
                "env.per_gpu_batch_size": 1,
            }
        )
    elif presets == "multilingual":
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
                "optimization.top_k": 1,
                "env.precision": "bf16",
                "env.per_gpu_batch_size": 4,
            }
        )
        hyperparameters.pop("env.num_workers", None)
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def few_shot_text_classification(presets: str = DEFAULT):
    """
    Register the presets for few_shot_text_classification.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["t_few"],
        "model.t_few.checkpoint_name": "google/flan-t5-xl",  # 3B model. google/flan-t5-xxl for 11B model.
        "model.t_few.gradient_checkpointing": True,
        "optimization.learning_rate": 1e-3,
        "optimization.lr_decay": 1.0,
        "optimization.efficient_finetune": "ia3_lora",
        "optimization.max_steps": 600,  # Find better solution to train for long
        "optimization.check_val_every_n_epoch": 10,  # Might need adjustment
        "optimization.val_check_interval": 1.0,
        "optimization.top_k_average_method": "best",
        "optimization.warmup_steps": 0.06,
        "optimization.lora.module_filter": [".*SelfAttention|.*EncDecAttention|.*DenseReluDense"],
        "optimization.lora.filter": ["q|k|v|wi_1.*"],
        "optimization.top_k": 1,
        "optimization.max_epochs": -1,
        "env.batch_size": 8,
        "env.per_gpu_batch_size": 8,
        "env.precision": "bf16",
        "data.templates.turn_on": True,
        "env.eval_batch_size_ratio": 2,
    }
    hyperparameter_tune_kwargs = None

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def zero_shot_image_classification(presets: str = DEFAULT):
    """
    Register the presets for zero_shot_image_classification.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["clip"],
        "model.clip.max_text_len": 0,
        "env.num_workers": 2,
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, BEST_QUALITY]:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
                "env.eval_batch_size_ratio": 1,
            }
        )
    elif presets == HIGH_QUALITY:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14",
                "env.eval_batch_size_ratio": 1,
            }
        )
    elif presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def object_detection(presets: str = DEFAULT):
    """
    Register the presets for object_detection.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["mmdet_image"],
        "env.eval_batch_size_ratio": 1,
        "env.precision": 32,
        "env.strategy": "ddp",
        "env.auto_select_gpus": False,  # Have to turn off for detection!
        "optimization.lr_decay": 0.95,
        "optimization.lr_mult": 100,
        "optimization.lr_choice": "two_stages",
        "optimization.top_k": 1,
        "optimization.top_k_average_method": "best",
        "optimization.warmup_steps": 0.0,
        "optimization.patience": 10,
        "env.num_workers": 2,
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, MEDIUM_QUALITY]:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
                "optimization.learning_rate": 1e-4,
                "optimization.max_epochs": 10,
                "optimization.val_metric": "direct_loss",
            }
        )
    elif presets == HIGH_QUALITY:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "yolov3_d53_mstrain-416_273e_coco",
                "optimization.learning_rate": 1e-5,
                "optimization.max_epochs": 20,
                "optimization.val_metric": "map",
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco",
                "optimization.learning_rate": 1e-5,
                "optimization.max_epochs": 30,
                "optimization.val_metric": "map",
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def ocr_text_detection(presets: str = DEFAULT):
    """
    Register the presets for ocr_text_detection.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["mmocr_text_detection"],
        "model.mmocr_text_detection.checkpoint_name": "TextSnake",
        "env.eval_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }
    hyperparameter_tune_kwargs = None

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def ocr_text_recognition(presets: str = DEFAULT):
    """
    Register the presets for ocr_text_recognition.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["mmocr_text_recognition"],
        "model.mmocr_text_recognition.checkpoint_name": "ABINet",
        "env.eval_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }
    hyperparameter_tune_kwargs = None

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def feature_extraction(presets: str = DEFAULT):  # TODO: rename the problem type as text_feature_extraction?
    """
    Register the presets for feature_extraction.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["hf_text"],
        "model.hf_text.checkpoint_name": "sentence-transformers/msmarco-MiniLM-L-12-v3",
        "model.hf_text.pooling_mode": "mean",
        "env.eval_batch_size_ratio": 1,
    }
    hyperparameter_tune_kwargs = None

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
@matcher_presets.register()
def image_similarity(presets: str = DEFAULT):
    """
    Register the presets for image_similarity.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["timm_image"],
        "env.num_workers": 2,
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, HIGH_QUALITY]:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
            }
        )
    elif presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
@matcher_presets.register()
def text_similarity(presets: str = DEFAULT):
    """
    Register the presets for text_similarity.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["hf_text"],
        "model.hf_text.pooling_mode": "mean",
        "data.categorical.convert_to_text": True,
        "data.numerical.convert_to_text": True,
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, HIGH_QUALITY]:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L12-v2",
            }
        )
    elif presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "sentence-transformers/all-mpnet-base-v2",
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
@matcher_presets.register()
def image_text_similarity(presets: str = DEFAULT):
    """
    Register the presets for image_text_similarity.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": ["clip"],
        "matcher.loss.type": "multi_negatives_softmax_loss",
        "optimization.learning_rate": 1e-5,
        "env.num_workers": 2,
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, MEDIUM_QUALITY]:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
                "env.per_gpu_batch_size": 128,
            }
        )
    elif presets == HIGH_QUALITY:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14",
                "env.per_gpu_batch_size": 16,
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
                "env.per_gpu_batch_size": 8,
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def ner(presets: str = DEFAULT):
    """
    Register the presets for ner.

    Parameters
    ----------
    presets
        The preset name.

    Returns
    -------
    hyperparameters
        The hyperparameters for a given preset.
    hyperparameter_tune_kwargs
        The hyperparameter tuning kwargs.
    """
    hyperparameters = {
        "model.names": [
            "categorical_mlp",
            "numerical_mlp",
            "timm_image",
            "ner_text",
            "fusion_ner",
        ],
    }
    hyperparameter_tune_kwargs = None

    if presets in [DEFAULT, HIGH_QUALITY]:
        hyperparameters.update(
            {
                "model.ner_text.checkpoint_name": "microsoft/deberta-v3-base",
            }
        )
    elif presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.ner_text.checkpoint_name": "google/electra-small-discriminator",
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.ner_text.checkpoint_name": "microsoft/deberta-v3-large",
                "env.per_gpu_batch_size": 4,
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


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


def get_basic_automm_config(extra: Optional[List[str]] = None):
    """
    Get the basic config of AutoMM.

    Parameters
    ----------
    extra
        A list of extra config keys.

    Returns
    -------
    A dict config with keys: MODEL, DATA, OPTIMIZATION, ENVIRONMENT, and their default values.
    """
    config = {
        MODEL: "fusion_mlp_image_text_tabular",
        DATA: DEFAULT,
        OPTIMIZATION: "adamw",
        ENVIRONMENT: DEFAULT,
    }
    if extra:
        for k in extra:
            config[k] = DEFAULT

    return config


def get_automm_presets(problem_type: str, presets: str):
    """
    Get the default hyperparameters and hyperparameter_tune_kwargs given problem type and presets.

    Parameters
    ----------
    problem_type
        Problem type.
    presets
        Name of a preset.

    Returns
    -------
    hyperparameters
        The hyperparameter overrides of this preset.
    hyperparameter_tune_kwargs
        Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
    """
    presets = presets.lower()
    if problem_type in [
        BINARY,
        MULTICLASS,
        REGRESSION,
        None,
    ]:
        problem_type = DEFAULT

    if problem_type in automm_presets.list_keys():
        hyperparameters, hyperparameter_tune_kwargs = automm_presets.create(problem_type, presets)
    else:
        raise ValueError(
            f"Problem type '{problem_type}' doesn't have any presets yet. "
            f"Consider one of these: {automm_presets.list_keys()}"
        )

    return hyperparameters, hyperparameter_tune_kwargs
