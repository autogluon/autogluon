import os
import shutil
import pytest
import numpy.testing as npt
import tempfile
import copy
import pickle

from omegaconf import OmegaConf
from ray import tune
from torch import nn

from autogluon.core.hpo.constants import SEARCHER_PRESETS, SCHEDULER_PRESETS
from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.utils import modify_duplicate_model_names
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    DISTILLER,
    BINARY,
    MULTICLASS,
    UNIFORM_SOUP,
    GREEDY_SOUP,
    BEST,
    NORM_FIT,
    BIT_FIT,
    LORA,
    LORA_BIAS,
    LORA_NORM,
)
from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)
from utils import get_home_dir

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def verify_predictor_save_load(predictor, df,
                               verify_embedding=True):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df, as_pandas=False)
        loaded_predictor = AutoMMPredictor.load(root)
        predictions2 = loaded_predictor.predict(df, as_pandas=False)
        predictions2_df = loaded_predictor.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2,
                         predictions2_df.to_numpy())
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
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,efficient_finetune",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            LORA
        ),

        (
            "hateful_memes",
            ["timm_image", "hf_text", "clip", "fusion_mlp"],
            "monsoon-nlp/hindi-bert",
            "swin_tiny_patch4_window7_224",
            UNIFORM_SOUP,
            LORA_NORM
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "fusion_mlp"],
            None,
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            None
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "hf_text", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            None,
            UNIFORM_SOUP,
            None
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
            None,
            None,
            BEST,
            BIT_FIT
        ),

        (
            "hateful_memes",
            ["timm_image"],
            None,
            "swin_tiny_patch4_window7_224",
            UNIFORM_SOUP,
            NORM_FIT
        ),

        (
            "ae",
            ["hf_text"],
            "prajjwal1/bert-tiny",
            None,
            BEST,
            LORA_BIAS
        ),

        (
            "hateful_memes",
            ["clip"],
            None,
            None,
            BEST,
            NORM_FIT
        ),

    ]
)
def test_predictor(
        dataset_name,
        model_names,
        text_backbone,
        image_backbone,
        top_k_average_method,
        efficient_finetune
):
    dataset = ALL_DATASETS[dataset_name]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
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
    }
    if text_backbone is not None:
        hyperparameters.update({
            "model.hf_text.checkpoint_name": text_backbone,
        })
    if image_backbone is not None:
        hyperparameters.update({
            "model.timm_image.checkpoint_name": image_backbone,
        })
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
        predictor = AutoMMPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            hyperparameters=hyperparameters,
            time_limit=30,
        )


def test_standalone(): # test standalong feature in AutoMMPredictor.save()
    from unittest import mock
    import torch

    requests_gag = mock.patch(
        'requests.Session.request',
        mock.Mock(side_effect=RuntimeError(
            'Please use the `responses` library to mock HTTP in your tests.'
        ))
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
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    predictor = AutoMMPredictor(
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

    loaded_online_predictor = AutoMMPredictor.load(path=save_path)
    online_predictions = loaded_online_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_online_predictor

    # Check if the predictor can be loaded from an offline enivronment.
    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname) # block reading files in `.cache`
            loaded_offline_predictor = AutoMMPredictor.load(path=save_path_standalone)


    offline_predictions = loaded_offline_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_offline_predictor

    # check if save with standalone=True coincide with standalone=False
    npt.assert_equal(online_predictions,offline_predictions)


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
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "categorical_mlp_abc", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "prajjwal1/bert-tiny",
            "data.categorical.convert_to_text": False,
        },
    ]
)
def test_customizing_model_names(
        hyperparameters,
):
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
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
        time_limit=10,
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
        time_limit=10,
    )
    assert sorted(predictor._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._config.model, per_name)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = AutoMMPredictor.load(root)
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

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    model_config = { 
        'model': {
                'names': ['hf_text', 'timm_image', 'clip', 'categorical_transformer', 'numerical_transformer', 'fusion_transformer'], 
                'categorical_transformer': {
                    'out_features': 192, 
                    'd_token': 192, 
                    'num_trans_blocks': 0, 
                    'num_attn_heads': 4, 
                    'residual_dropout': 0.0, 
                    'attention_dropout': 0.2, 
                    'ffn_dropout': 0.1, 
                    'normalization': 'layer_norm', 
                    'ffn_activation': 'reglu', 
                    'head_activation': 'relu', 
                    'data_types': ['categorical']
                }, 
                'numerical_transformer': {
                    'out_features': 192, 
                    'd_token': 192, 
                    'num_trans_blocks': 0, 
                    'num_attn_heads': 4, 
                    'residual_dropout': 0.0, 
                    'attention_dropout': 0.2, 
                    'ffn_dropout': 0.1, 
                    'normalization': 'layer_norm', 
                    'ffn_activation': 'reglu', 
                    'head_activation': 'relu', 
                    'data_types': ['numerical'], 
                    'embedding_arch': ['linear','relu'],
                    'merge': 'concat'
                }, 
                'hf_text': {
                    'checkpoint_name': 'google/electra-base-discriminator', 
                    'data_types': ['text'], 
                    'tokenizer_name': 'hf_auto', 
                    'max_text_len': 512, 
                    'insert_sep': True, 
                    'text_segment_num': 2, 
                    'stochastic_chunk': False,
                    'text_aug_detect_length': 10,
                    'text_trivial_aug_maxscale': 0.05,
                    'test_train_augment_types' : ["synonym_replacement(0.1)"],
                }, 
                'timm_image': {
                    'checkpoint_name': 'swin_base_patch4_window7_224', 
                    'mix_choice': 'all_logits', 
                    'data_types': ['image'], 
                    'train_transform_types': ['resize_shorter_side', 'center_crop'], 
                    'val_transform_types': ['resize_shorter_side', 'center_crop'], 
                    'image_norm': 'imagenet', 
                    'image_size': 224,
                    'max_img_num_per_col': 2,
                },
                'clip': {
                    'checkpoint_name': 'openai/clip-vit-base-patch32', 
                    'data_types': ['image', 'text'], 
                    'train_transform_types': ['resize_shorter_side', 'center_crop'], 
                    'val_transform_types': ['resize_shorter_side', 'center_crop'], 
                    'image_norm': 'clip', 
                    'image_size': 224, 
                    'max_img_num_per_col': 2, 
                    'tokenizer_name': 'clip', 
                    'max_text_len': 77, 
                    'insert_sep': False, 
                    'text_segment_num': 1, 
                    'stochastic_chunk': False,
                    'text_aug_detect_length': 10,
                    'text_trivial_aug_maxscale': 0.05,
                    'test_train_augment_types' : ["synonym_replacement(0.1)"],
                }, 
                'fusion_transformer': {
                    'hidden_size': 192, 
                    'n_blocks': 2, 
                    'attention_n_heads': 4, 
                    'adapt_in_features': 'max', 
                    'attention_dropout': 0.2, 
                    'residual_dropout': 0.0, 
                    'ffn_dropout': 0.1, 
                    'ffn_d_hidden': 192, 
                    'normalization': 'layer_norm', 
                    'ffn_activation': 'geglu', 
                    'head_activation': 'relu', 
                    'data_types': None
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

    teacher_predictor = AutoMMPredictor(
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
    teacher_predictor.fit(
        train_data=dataset.train_df,
        config=config,
        time_limit=1,
    )
    student_predictor = AutoMMPredictor(
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
    assert all([n not in teacher_predictor._config.model.names for n in student_predictor._config.model.names]), \
        f"teacher model names {teacher_predictor._config.model.names} and" \
        f" student model names {student_predictor._config.model.names} have duplicates."

    # verify each model name prefix is valid
    assert teacher_predictor._model.prefix in teacher_predictor._config.model.names
    if isinstance(teacher_predictor._model.model, nn.ModuleList):
        for per_model in teacher_predictor._model.model:
            assert per_model.prefix in teacher_predictor._config.model.names

    # verify each data processor's prefix is valid
    for per_modality_processors in teacher_predictor._data_processors.values():
        for per_processor in per_modality_processors:
            assert per_processor.prefix in teacher_predictor._config.model.names

def test_mixup():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
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
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)

def test_textagumentor_deepcopy():
    dataset = ALL_DATASETS["ae"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
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
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "model.hf_text.text_trivial_aug_maxscale": 0.05,
        "model.hf_text.text_train_augment_types": ["synonym_replacement(0.05)",
                                                   "random_swap(0.05)"],
        "optimization.top_k_average_method": "uniform_soup",
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

    # Deep copy data preprocessor
    df_preprocessor_copy = copy.deepcopy(predictor._df_preprocessor)
    predictor._df_preprocessor = df_preprocessor_copy

    # Test for copied preprocessor 
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
    )

    # Copy data preprocessor via pickle + load
    df_preprocessor_copy_via_pickle = pickle.loads(pickle.dumps(predictor._df_preprocessor))
    predictor._df_preprocessor = df_preprocessor_copy_via_pickle

    # Test for copied preprocessor
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
    )

@pytest.mark.parametrize('searcher', list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize('scheduler', list(SCHEDULER_PRESETS.keys()))
def test_hpo(searcher, scheduler):
    dataset = PetFinderDataset()

    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.0001, 0.01),
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }
    
    hyperparameter_tune_kwargs = {
        'searcher': searcher,
        'scheduler': scheduler,
        'num_trials': 2,
    }

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )
    
    save_path = os.path.join(get_home_dir(), 'hpo', f'_{searcher}', f'_{scheduler}')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=60,
        save_path=save_path,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
    
    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)
    
    # test for continuous training
    predictor = predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=60,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
    
    
@pytest.mark.parametrize('searcher', list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize('scheduler', list(SCHEDULER_PRESETS.keys()))
def test_hpo_distillation(searcher, scheduler):
    dataset = PetFinderDataset()

    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
        DISTILLER: "default",
    }

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }
    
    hyperparameter_tune_kwargs = {
        'searcher': searcher,
        'scheduler': scheduler,
        'num_trials': 2,
    }

    teacher_predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )
    
    teacher_save_path = os.path.join(get_home_dir(), 'hpo_distillation_teacher', f'_{searcher}', f'_{scheduler}')
    if os.path.exists(teacher_save_path):
        shutil.rmtree(teacher_save_path)

    teacher_predictor = teacher_predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=60,
        save_path=teacher_save_path,
    )
    
    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.0001, 0.01),
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }
    # test for distillation
    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )
    
    student_save_path = os.path.join(get_home_dir(), 'hpo_distillation_student', f'_{searcher}', f'_{scheduler}')
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        teacher_predictor=teacher_save_path,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=60,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        save_path=student_save_path,
    )

def test_trivialaugment():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
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
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
        "model.hf_text.text_trivial_aug_maxscale":0.1,
        "model.hf_text.text_aug_detect_length":10,
        "model.timm_image.train_transform_types": ["resize_shorter_side", "center_crop", "trivial_augment"],
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)