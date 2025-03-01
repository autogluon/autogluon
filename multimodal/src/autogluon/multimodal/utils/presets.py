from typing import List, Optional

from autogluon.common.utils.try_import import try_import_ray

from ..constants import (
    BEST_QUALITY,
    BINARY,
    DATA,
    DEFAULT,
    ENV,
    HIGH_QUALITY,
    MEDIUM_QUALITY,
    MODEL,
    MULTICLASS,
    OPTIM,
    REGRESSION,
)
from .registry import Registry

automm_presets = Registry("automm_presets")
matcher_presets = Registry("matcher_presets")


def get_default_hpo_setup():
    try_import_ray()
    from ray import tune

    default_hyperparameter_tune_kwargs = {
        "searcher": "bayes",
        "scheduler": "ASHA",
        "num_trials": 512,
    }

    default_tunable_hyperparameters = {
        "optim.lr": tune.loguniform(1e-5, 1e-2),
        "optim.optim_type": tune.choice(["adamw", "sgd"]),
        "optim.max_epochs": tune.choice(list(range(5, 31))),
        "env.batch_size": tune.choice([16, 32, 64, 128, 256]),
    }

    return default_tunable_hyperparameters, default_hyperparameter_tune_kwargs


def parse_presets_str(presets: str):
    use_hpo = False
    if presets.endswith("_hpo"):
        presets = presets[:-4]
        use_hpo = True

    return presets, use_hpo


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
            "ft_transformer",
            "timm_image",
            "hf_text",
            "document_transformer",
            "fusion_mlp",
        ],
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

    if presets in [HIGH_QUALITY, DEFAULT]:
        if use_hpo:
            from ray import tune

            hyperparameters.update(
                {
                    "env.per_gpu_batch_size": 2,  # Cover some corner cases of HPO on multimodal data.
                    "model.hf_text.checkpoint_name": tune.choice(
                        [
                            "google/electra-base-discriminator",
                            "google/flan-t5-base",
                            "microsoft/deberta-v3-small",
                            "roberta-base",
                            "albert-xlarge-v2",
                        ]
                    ),
                    "model.timm_image.checkpoint_name": tune.choice(
                        [
                            "swin_base_patch4_window7_224",
                            "convnext_base_in22ft1k",
                            "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
                            "caformer_b36.sail_in22k_ft_in1k",
                        ]
                    ),
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv3-base",
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
                    "model.timm_image.checkpoint_name": "caformer_b36.sail_in22k_ft_in1k",
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv3-base",
                }
            )
    elif presets == MEDIUM_QUALITY:
        if use_hpo:
            from ray import tune

            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": tune.choice(
                        [
                            "google/electra-small-discriminator",
                            "google/flan-t5-small",
                            "microsoft/deberta-v3-xsmall",
                            "albert-base-v2",
                            "microsoft/MiniLM-L12-H384-uncased",
                        ]
                    ),
                    "model.timm_image.checkpoint_name": tune.choice(
                        ["mobilenetv3_large_100", "gluon_resnet18_v1b", "maxvit_rmlp_pico_rw_256.sw_in1k"]
                    ),
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv2-base-uncased",
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                    "model.timm_image.checkpoint_name": "mobilenetv3_large_100",
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv2-base-uncased",
                    "optim.lr": 4e-4,
                }
            )
    elif presets == BEST_QUALITY:
        hyperparameters.update({"env.per_gpu_batch_size": 1})
        if use_hpo:
            from ray import tune

            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": tune.choice(
                        [
                            "microsoft/deberta-v3-base",
                            "google/flan-t5-large",
                            "google/electra-large-discriminator",
                            "roberta-large",
                        ]
                    ),
                    "model.timm_image.checkpoint_name": tune.choice(
                        [
                            "swin_large_patch4_window7_224",
                            "eva_large_patch14_336.in22k_ft_in22k_in1k",
                            "vit_large_patch14_clip_336.openai_ft_in12k_in1k",
                        ]
                    ),
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv3-large",
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
                    "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
                    "model.document_transformer.checkpoint_name": "microsoft/layoutlmv3-large",
                }
            )
    elif presets == "multilingual":
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
                "optim.top_k": 1,
                "env.precision": "bf16-mixed",
                "env.per_gpu_batch_size": 4,
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def few_shot_classification(presets: str = DEFAULT):
    """
    Register the presets for few_shot_classification.

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
    hyperparameters, hyperparameter_tune_kwargs = default(presets=presets)
    hyperparameters.update(
        {
            "model.hf_text.checkpoint_name": "sentence-transformers/all-mpnet-base-v2",
            "model.hf_text.pooling_mode": "mean",
            "model.names": ["hf_text", "clip"],
            "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
            "model.clip.image_size": 336,
            "env.inference_batch_size_ratio": 1,
        }
    )
    hyperparameter_tune_kwargs = {}

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
    }
    hyperparameter_tune_kwargs = {}

    if presets in [DEFAULT, BEST_QUALITY]:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
                "model.clip.image_size": 336,
                "env.inference_batch_size_ratio": 1,
            }
        )
    elif presets == HIGH_QUALITY:
        hyperparameters.update(
            {
                "model.clip.checkpoint_name": "openai/clip-vit-large-patch14",
                "env.inference_batch_size_ratio": 1,
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
        "model.mmdet_image.frozen_layers": [],
        "optim.patience": 20,
        "optim.val_check_interval": 1.0,
        "optim.check_val_every_n_epoch": 1,
        "env.batch_size": 32,
        "env.per_gpu_batch_size": 1,
        "env.num_workers": 2,
        "optim.lr": 1e-5,
        "optim.weight_decay": 1e-4,
        "optim.lr_mult": 10,
        "optim.lr_choice": "two_stages",
        "optim.lr_schedule": "multi_step",
        "optim.gradient_clip_val": 0.1,
        "optim.max_epochs": 60,
        "optim.warmup_steps": 0.0,
        "optim.top_k": 1,
        "optim.top_k_average_method": "best",
        "env.inference_batch_size_ratio": 1,
        "env.strategy": "ddp",
        "env.auto_select_gpus": True,  # Turn on for detection to return devices in a list, TODO: fix the extra GPU usage bug
        "env.num_gpus": -1,
        "optim.lr_decay": 0.9,
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

    if presets == MEDIUM_QUALITY:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "yolox_l",
                "env.per_gpu_batch_size": 2,  # Works on 8G GPU
                "optim.lr": 5e-5,
                "optim.patience": 5,
                "optim.max_epochs": 50,
                "optim.val_check_interval": 1.0,
                "optim.check_val_every_n_epoch": 3,
                "optim.lr_mult": 100,
                "optim.weight_decay": 1e-3,
                "optim.lr_schedule": "cosine_decay",
                "optim.gradient_clip_val": 1,
            }
        )
    elif presets in [DEFAULT, HIGH_QUALITY]:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "dino-4scale_r50_8xb2-12e_coco",
            }
        )
    elif presets == BEST_QUALITY:
        hyperparameters.update(
            {
                "model.mmdet_image.checkpoint_name": "dino-5scale_swin-l_8xb2-36e_coco",
            }
        )
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, hyperparameter_tune_kwargs


@automm_presets.register()
def semantic_segmentation(presets: str = DEFAULT):
    """
    Register the presets for semantic_segmentation.

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
        "model.names": ["sam"],
        "model.sam.checkpoint_name": "facebook/sam-vit-huge",
        "env.batch_size": 4,
        "env.per_gpu_batch_size": 1,
        "env.inference_batch_size_ratio": 1,
        "env.strategy": "ddp_find_unused_parameters_true",
        "env.auto_select_gpus": False,
        "env.num_gpus": -1,
        "env.num_workers": 4,
        "env.precision": "16-mixed",
        "optim.lr": 1e-4,
        "optim.loss_func": "structure_loss",
        "optim.lr_decay": 0,
        "optim.lr_mult": 1,
        "optim.lr_choice": "single_stage",
        "optim.lr_schedule": "polynomial_decay",
        "optim.max_epochs": 30,
        "optim.top_k": 3,
        "optim.top_k_average_method": "best",
        "optim.warmup_steps": 0.0,
        "optim.weight_decay": 0.0001,
        "optim.patience": 10,
        "optim.val_check_interval": 1.0,
        "optim.check_val_every_n_epoch": 1,
        "optim.peft": "lora",
        "optim.lora.module_filter": [".*vision_encoder.*attn"],
        "optim.lora.filter": ["q", "v"],
        "optim.extra_trainable_params": [".*mask_decoder"],
        "optim.lora.r": 3,
        "optim.lora.alpha": 32,
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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
        "env.inference_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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
        "env.inference_batch_size_ratio": 1,
        "env.num_gpus": 1,
        "env.precision": 32,
    }
    hyperparameter_tune_kwargs = {}
    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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
        "env.inference_batch_size_ratio": 1,
    }
    hyperparameter_tune_kwargs = {}

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
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

    if presets in [DEFAULT, HIGH_QUALITY]:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": "caformer_b36.sail_in22k_ft_in1k",
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
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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
        "optim.lr": 1e-5,
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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
                "model.clip.image_size": 336,
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
            "ft_transformer",
            "timm_image",
            "ner_text",
            "fusion_ner",
        ],
    }
    hyperparameter_tune_kwargs = {}

    presets, use_hpo = parse_presets_str(presets)
    if use_hpo:
        default_tunable_hyperparameters, default_hyperparameter_tune_kwargs = get_default_hpo_setup()
        hyperparameters.update(default_tunable_hyperparameters)
        hyperparameter_tune_kwargs.update(default_hyperparameter_tune_kwargs)

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


@automm_presets.register()
def ensemble(presets: str = DEFAULT):
    hyperparameters = {
        "lf_mlp": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "lf_transformer": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_transformer"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "lf_clip": {
            "model.names": ["ft_transformer", "clip_image", "clip_text", "fusion_mlp"],
            "model.clip_image.data_types": ["image"],
            "model.clip_text.data_types": ["text"],
            "model.clip_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.clip_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.clip_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "early_fusion": {
            "model.names": ["meta_transformer"],
            "model.meta_transformer.checkpoint_path": "null",
            "model.meta_transformer.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.meta_transformer.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.meta_transformer.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "convert_categorical_to_text": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": True,
            "data.categorical.convert_to_text_template": "latex",
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "convert_numeric_to_text": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": True,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "cross_modal_align_pos_only": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "positive_only",
            "optim.cross_modal_align_weight": 1,
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "input_aug": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"],
            "model.hf_text.text_trivial_aug_maxscale": 0.1,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "feature_aug_lemda": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": True,
            "optim.automatic_optimization": False,
        },
        "modality_dropout": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0.2,
            "model.timm_image.use_learnable_image": False,
            "optim.lemda.turn_on": False,
        },
        "learnable_image": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0,
            "model.timm_image.use_learnable_image": True,
            "optim.lemda.turn_on": False,
        },
        "modality_dropout_and_learnable_image": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.hf_text.text_trivial_aug_maxscale": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
            "optim.cross_modal_align": "null",
            "data.modality_dropout": 0.2,
            "model.timm_image.use_learnable_image": True,
            "optim.lemda.turn_on": False,
        },
    }

    if presets in [DEFAULT, HIGH_QUALITY]:
        for v in hyperparameters.values():
            if "timm_image" in v["model.names"]:
                v["model.timm_image.checkpoint_name"] = "caformer_b36.sail_in22k_ft_in1k"
            if "hf_text" in v["model.names"]:
                v["model.hf_text.checkpoint_name"] = "google/electra-base-discriminator"
            if "meta_transformer" in v["model.names"]:
                v["model.meta_transformer.model_version"] = "base"
            if "clip_image" in v["model.names"]:
                v["model.clip_image.checkpoint_name"] = "openai/clip-vit-base-patch32"
            if "clip_text" in v["model.names"]:
                v["model.clip_text.checkpoint_name"] = "openai/clip-vit-base-patch32"

    elif presets == MEDIUM_QUALITY:
        for v in hyperparameters.values():
            if "timm_image" in v["model.names"]:
                v["model.timm_image.checkpoint_name"] = "mobilenetv3_large_100"
            if "hf_text" in v["model.names"]:
                v["model.hf_text.checkpoint_name"] = "google/electra-small-discriminator"
            if "meta_transformer" in v["model.names"]:
                v["model.meta_transformer.model_version"] = "base"
            if "clip_image" in v["model.names"]:
                v["model.clip_image.checkpoint_name"] = "openai/clip-vit-base-patch32"
            if "clip_text" in v["model.names"]:
                v["model.clip_text.checkpoint_name"] = "openai/clip-vit-base-patch32"
    elif presets == BEST_QUALITY:
        for v in hyperparameters.values():
            if "timm_image" in v["model.names"]:
                v["model.timm_image.checkpoint_name"] = "swin_large_patch4_window7_224"
            if "hf_text" in v["model.names"]:
                v["model.hf_text.checkpoint_name"] = "microsoft/deberta-v3-base"
            if "meta_transformer" in v["model.names"]:
                v["model.meta_transformer.model_version"] = "large"
            if "clip_image" in v["model.names"]:
                v["model.clip_image.checkpoint_name"] = "openai/clip-vit-large-patch14"
            if "clip_text" in v["model.names"]:
                v["model.clip_text.checkpoint_name"] = "openai/clip-vit-large-patch14"
    else:
        raise ValueError(f"Unknown preset type: {presets}")

    return hyperparameters, None


def list_presets(verbose: bool = False):
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


def get_basic_config(extra: Optional[List[str]] = None):
    """
    Get the basic config of AutoMM.

    Parameters
    ----------
    extra
        A list of extra config keys.

    Returns
    -------
    A dict config with keys: MODEL, DATA, OPTIM, ENV, and their default values.
    """
    config = {
        MODEL: DEFAULT,
        DATA: DEFAULT,
        OPTIM: DEFAULT,
        ENV: DEFAULT,
    }
    if extra:
        for k in extra:
            config[k] = DEFAULT

    return config


def get_presets(problem_type: str, presets: str):
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
    if not presets:
        presets = DEFAULT
    if presets == "hpo":
        presets = f"{DEFAULT}_{presets}"
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


def get_ensemble_presets(presets):
    if not presets:
        presets = DEFAULT
    return automm_presets.create("ensemble", presets)
