import copy
import os
import pickle
import shutil
import tempfile

import numpy.testing as npt
import pytest
from omegaconf import OmegaConf
from torch import nn
from unittest_datasets import AEDataset, HatefulMeMesDataset, PetFinderDataset
from utils import get_home_dir

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    BEST,
    BINARY,
    BIT_FIT,
    DATA,
    DISTILLER,
    ENVIRONMENT,
    GREEDY_SOUP,
    IA3,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MODEL,
    MULTICLASS,
    NORM_FIT,
    OPTIMIZATION,
    UNIFORM_SOUP,
)
from autogluon.multimodal.utils import modify_duplicate_model_names

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def verify_predictor_save_load(predictor, df, verify_embedding=True, cls=MultiModalPredictor):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df, as_pandas=False)
        loaded_predictor = cls.load(root)
        predictions2 = loaded_predictor.predict(df, as_pandas=False)
        predictions2_df = loaded_predictor.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2, predictions2_df.to_numpy())
        if predictor.problem_type in [BINARY, MULTICLASS]:
            predictions_prob = predictor.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            embeddings = predictor.extract_embedding(df)
            assert embeddings.shape[0] == len(df)


@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,efficient_finetune,loss_function",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            LORA,
            "auto",
        ),
        (
            "petfinder",
            ["t_few"],
            "t5-small",
            None,
            BEST,
            IA3,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image", "t_few", "clip", "fusion_mlp"],
            "t5-small",
            "swin_tiny_patch4_window7_224",
            BEST,
            IA3,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image", "hf_text", "clip", "fusion_mlp"],
            "monsoon-nlp/hindi-bert",
            "swin_tiny_patch4_window7_224",
            UNIFORM_SOUP,
            LORA_NORM,
            "auto",
        ),
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "fusion_mlp"],
            None,
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            None,
            "auto",
        ),
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "hf_text", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            None,
            UNIFORM_SOUP,
            None,
            "auto",
        ),
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
            None,
            None,
            BEST,
            BIT_FIT,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image"],
            None,
            "swin_tiny_patch4_window7_224",
            UNIFORM_SOUP,
            NORM_FIT,
            "auto",
        ),
        (
            "ae",
            ["hf_text"],
            "prajjwal1/bert-tiny",
            None,
            BEST,
            LORA_BIAS,
            "bcewithlogitsloss",
        ),
        (
            "hateful_memes",
            ["clip"],
            None,
            None,
            BEST,
            NORM_FIT,
            "auto",
        ),
    ],
)
def test_predictor(
    dataset_name,
    model_names,
    text_backbone,
    image_backbone,
    top_k_average_method,
    efficient_finetune,
    loss_function,
):
    dataset = ALL_DATASETS[dataset_name]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": top_k_average_method,
        "optimization.efficient_finetune": efficient_finetune,
        "optimization.loss_function": loss_function,
    }
    if text_backbone is not None:
        if "t_few" in model_names:
            hyperparameters.update(
                {
                    "model.t_few.checkpoint_name": "t5-small",
                    "model.t_few.gradient_checkpointing": False,
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": text_backbone,
                }
            )
    if image_backbone is not None:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": image_backbone,
            }
        )
    save_path = os.path.join(get_home_dir(), "outputs", dataset_name)
    if text_backbone is not None:
        save_path = os.path.join(save_path, text_backbone)
    if image_backbone is not None:
        save_path = os.path.join(save_path, image_backbone)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
    )
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = MultiModalPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            hyperparameters=hyperparameters,
            time_limit=30,
        )


def test_standalone():  # test standalong feature in MultiModalPredictor.save()
    from unittest import mock

    import torch

    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )

    dataset = PetFinderDataset()

    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp", "t_few"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "model.t_few.checkpoint_name": "t5-small",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "standalone", "false")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    save_path_standalone = os.path.join(get_home_dir(), "standalone", "true")

    predictor.save(
        path=save_path_standalone,
        standalone=True,
    )

    del predictor
    torch.cuda.empty_cache()

    loaded_online_predictor = MultiModalPredictor.load(path=save_path)
    online_predictions = loaded_online_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_online_predictor

    # Check if the predictor can be loaded from an offline enivronment.
    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path_standalone)

    offline_predictions = loaded_offline_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_offline_predictor

    # check if save with standalone=True coincide with standalone=False
    npt.assert_equal(online_predictions, offline_predictions)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image_0", "timm_image_1", "fusion_mlp"],
            "model.timm_image_0.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.timm_image_1.checkpoint_name": "swin_small_patch4_window7_224",
        },
        {
            "model.names": "[timm_image_0, timm_image_1, fusion_mlp]",
            "model.timm_image_0.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.timm_image_1.checkpoint_name": "swin_small_patch4_window7_224",
        },
        {
            "model.names": ["hf_text_abc", "hf_text_def", "hf_text_xyz", "fusion_mlp_123"],
            "model.hf_text_def.checkpoint_name": "monsoon-nlp/hindi-bert",
            "model.hf_text_xyz.checkpoint_name": "prajjwal1/bert-tiny",
            "model.hf_text_abc.checkpoint_name": "roberta-base",
        },
        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "prajjwal1/bert-tiny",
            "data.numerical.convert_to_text": False,
        },
    ],
)
def test_customizing_model_names(
    hyperparameters,
):
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters.update(
        {
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
        }
    )
    hyperparameters_gt = copy.deepcopy(hyperparameters)
    if isinstance(hyperparameters_gt["model.names"], str):
        hyperparameters_gt["model.names"] = OmegaConf.from_dotlist([f'names={hyperparameters["model.names"]}']).names

    save_path = os.path.join(get_home_dir(), "outputs", "petfinder")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )

    assert sorted(predictor._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._config.model, per_name)

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=20,
    )
    assert sorted(predictor._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._config.model, per_name)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = MultiModalPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            hyperparameters=hyperparameters,
            time_limit=10,
        )
        assert sorted(predictor._config.model.names) == sorted(hyperparameters_gt["model.names"])
        for per_name in hyperparameters_gt["model.names"]:
            assert hasattr(predictor._config.model, per_name)


def test_model_configs():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    model_config = {
        "model": {
            "names": [
                "hf_text",
                "timm_image",
                "clip",
                "categorical_transformer",
                "numerical_transformer",
                "fusion_transformer",
            ],
            "categorical_transformer": {
                "out_features": 192,
                "d_token": 192,
                "num_trans_blocks": 0,
                "num_attn_heads": 4,
                "residual_dropout": 0.0,
                "attention_dropout": 0.2,
                "ffn_dropout": 0.1,
                "normalization": "layer_norm",
                "ffn_activation": "reglu",
                "head_activation": "relu",
                "data_types": ["categorical"],
            },
            "numerical_transformer": {
                "out_features": 192,
                "d_token": 192,
                "num_trans_blocks": 0,
                "num_attn_heads": 4,
                "residual_dropout": 0.0,
                "attention_dropout": 0.2,
                "ffn_dropout": 0.1,
                "normalization": "layer_norm",
                "ffn_activation": "reglu",
                "head_activation": "relu",
                "data_types": ["numerical"],
                "embedding_arch": ["linear", "relu"],
                "merge": "concat",
            },
            "t_few": {
                "checkpoint_name": "t5-small",
                "gradient_checkpointing": False,
                "data_types": ["text"],
                "tokenizer_name": "hf_auto",
                "length_norm": 1.0,
                "unlikely_loss": 1.0,
                "mc_loss": 1.0,
                "max_text_len": 512,
                "insert_sep": True,
                "text_segment_num": 2,
                "stochastic_chunk": False,
                "text_aug_detect_length": 10,
            },
            "hf_text": {
                "checkpoint_name": "google/electra-small-discriminator",
                "data_types": ["text"],
                "tokenizer_name": "hf_auto",
                "max_text_len": 512,
                "insert_sep": True,
                "text_segment_num": 2,
                "stochastic_chunk": False,
                "text_aug_detect_length": 10,
                "text_trivial_aug_maxscale": 0.05,
                "test_train_augment_types": ["synonym_replacement(0.1)"],
            },
            "timm_image": {
                "checkpoint_name": "swin_tiny_patch4_window7_224",
                "mix_choice": "all_logits",
                "data_types": ["image"],
                "train_transform_types": ["resize_shorter_side", "center_crop"],
                "val_transform_types": ["resize_shorter_side", "center_crop"],
                "image_norm": "imagenet",
                "image_size": 224,
                "max_img_num_per_col": 2,
            },
            "clip": {
                "checkpoint_name": "openai/clip-vit-base-patch32",
                "data_types": ["image", "text"],
                "train_transform_types": ["resize_shorter_side", "center_crop"],
                "val_transform_types": ["resize_shorter_side", "center_crop"],
                "image_norm": "clip",
                "image_size": 224,
                "max_img_num_per_col": 2,
                "tokenizer_name": "clip",
                "max_text_len": 77,
                "insert_sep": False,
                "text_segment_num": 1,
                "stochastic_chunk": False,
                "text_aug_detect_length": 10,
                "text_trivial_aug_maxscale": 0.05,
                "test_train_augment_types": ["synonym_replacement(0.1)"],
            },
            "fusion_transformer": {
                "hidden_size": 192,
                "n_blocks": 2,
                "attention_n_heads": 4,
                "adapt_in_features": "max",
                "attention_dropout": 0.2,
                "residual_dropout": 0.0,
                "ffn_dropout": 0.1,
                "ffn_d_hidden": 192,
                "normalization": "layer_norm",
                "ffn_activation": "geglu",
                "head_activation": "relu",
                "data_types": None,
            },
        }
    }

    hyperparameters = {
        "optimization.max_epochs": 1,
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
    }

    config = {
        MODEL: model_config,
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)


def test_modifying_duplicate_model_names():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        config=config,
        time_limit=1,
    )
    student_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    student_predictor.fit(
        train_data=dataset.train_df,
        config=config,
        time_limit=0,
    )

    teacher_predictor = modify_duplicate_model_names(
        predictor=teacher_predictor,
        postfix="teacher",
        blacklist=student_predictor._config.model.names,
    )

    # verify teacher and student have no duplicate model names
    assert all([n not in teacher_predictor._config.model.names for n in student_predictor._config.model.names]), (
        f"teacher model names {teacher_predictor._config.model.names} and"
        f" student model names {student_predictor._config.model.names} have duplicates."
    )

    # verify each model name prefix is valid
    assert teacher_predictor._model.prefix in teacher_predictor._config.model.names
    if isinstance(teacher_predictor._model.model, nn.ModuleList):
        for per_model in teacher_predictor._model.model:
            assert per_model.prefix in teacher_predictor._config.model.names

    # verify each data processor's prefix is valid
    for per_modality_processors in teacher_predictor._data_processors.values():
        for per_processor in per_modality_processors:
            assert per_processor.prefix in teacher_predictor._config.model.names
