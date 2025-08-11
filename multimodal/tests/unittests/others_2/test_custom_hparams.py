import copy
import os
import shutil
import tempfile

import pytest
from omegaconf import OmegaConf
from torchvision import transforms

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    DATA,
    IMAGE_PATH,
    MODEL,
    MULTICLASS,
    NUMERICAL,
    REGRESSION,
    TEXT,
)
from autogluon.multimodal.utils import (
    filter_hyperparameters,
    get_default_config,
    shopee_dataset,
    split_hyperparameters,
    update_ensemble_hyperparameters,
)

from ..utils import PetFinderDataset, get_home_dir, verify_predictor_save_load


def test_hyperparameters_in_terminal_format():
    download_dir = "./"
    train_df, tune_df = shopee_dataset(download_dir=download_dir)
    predictor = MultiModalPredictor(label="label")
    hyperparameters = [
        "model.names=[timm_image]",
        "model.timm_image.checkpoint_name=ghostnet_100",
        "env.num_workers=0",
        "env.num_workers_inference=0",
        "optim.top_k_average_method=best",
        "optim.val_check_interval=1.0",
    ]
    predictor.fit(
        train_data=train_df,
        tuning_data=tune_df,
        hyperparameters=hyperparameters,
        time_limit=5,
    )


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
        },
        {
            "model.names": ["hf_text"],
            "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            "data.numerical.convert_to_text": False,
        },
    ],
)
def test_hyperparameters_consistency(hyperparameters):
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    # pass hyperparameters to init()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
        hyperparameters=hyperparameters,
    )
    predictor.fit(dataset.train_df, time_limit=0)

    # pass hyperparameters to fit()
    predictor_2 = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    predictor_2.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=0,
    )
    assert predictor._learner._config == predictor_2._learner._config


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image_0", "timm_image_1", "fusion_mlp"],
            "model.timm_image_0.checkpoint_name": "mobilenetv3_small_100",
            "model.timm_image_1.checkpoint_name": "mobilenetv3_small_100",
        },
        {
            "model.names": ["hf_text_abc", "hf_text_def", "hf_text_xyz", "fusion_mlp_123"],
            "model.hf_text_def.checkpoint_name": "monsoon-nlp/hindi-bert",
            "model.hf_text_xyz.checkpoint_name": "distilroberta-base",
            "model.hf_text_abc.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "distilroberta-base",
            "data.numerical.convert_to_text": False,
        },
    ],
)
def test_customize_model_names(
    hyperparameters,
):
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters.update(
        {
            "env.num_workers": 0,
            "env.num_workers_inference": 0,
        }
    )
    hyperparameters_gt = copy.deepcopy(hyperparameters)
    if isinstance(hyperparameters_gt["model.names"], str):
        hyperparameters_gt["model.names"] = OmegaConf.from_dotlist([f"names={hyperparameters['model.names']}"]).names

    save_path = os.path.join(get_home_dir(), "outputs", "petfinder")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=0,
        save_path=save_path,
    )

    assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._learner._config.model, per_name)

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=0,
    )
    assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._learner._config.model, per_name)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = MultiModalPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            hyperparameters=hyperparameters,
            time_limit=0,
        )
        assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
        for per_name in hyperparameters_gt["model.names"]:
            assert hasattr(predictor._learner._config.model, per_name)


@pytest.mark.parametrize(
    "hyperparameters,column_types,model_in_config,fit_called,result",
    [
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH},
            None,
            False,
            {
                "model.names": ["timm_image", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT},
            None,
            False,
            {
                "model.names": ["hf_text", "fusion_mlp"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT, "c": NUMERICAL},
            None,
            False,
            {
                "model.names": ["numerical_mlp", "hf_text", "fusion_mlp"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH},
            None,
            False,
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["hf_text"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            {
                "model.names": ["hf_text"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["hf_text"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            AssertionError,
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "timm_image",
            False,
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "numerical_mlp",
            False,
            {
                "model.names": ["numerical_mlp"],
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "fusion_mlp"],
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "hf_text",
            False,
            AssertionError,
        ),
        (
            {
                "model.names": ["timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT},
            None,
            True,
            {},
        ),
    ],
)
def test_filter_hyperparameters(hyperparameters, column_types, model_in_config, fit_called, result):
    if model_in_config:
        config = get_default_config()
        config.model.names = [model_in_config]
        model_keys = list(config.model.keys())
        for key in model_keys:
            if key != "names" and key != model_in_config:
                delattr(config.model, key)
    else:
        config = None

    if result == AssertionError:
        with pytest.raises(AssertionError):
            filtered_hyperparameters = filter_hyperparameters(
                hyperparameters=hyperparameters,
                column_types=column_types,
                config=config,
                fit_called=fit_called,
            )
    else:
        filtered_hyperparameters = filter_hyperparameters(
            hyperparameters=hyperparameters,
            column_types=column_types,
            config=config,
            fit_called=fit_called,
        )

        assert filtered_hyperparameters == result


@pytest.mark.parametrize(
    "train_transforms,val_transforms,empty_advanced_hyperparameters",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
            True,
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
            False,
        ),
    ],
)
def test_split_hyperparameters(train_transforms, val_transforms, empty_advanced_hyperparameters):
    hyperparameters = {
        "model.timm_image.train_transforms": train_transforms,
        "model.timm_image.val_transforms": val_transforms,
    }
    hyperparameters, advanced_hyperparameters = split_hyperparameters(hyperparameters)
    if empty_advanced_hyperparameters:
        assert not advanced_hyperparameters
    else:
        assert advanced_hyperparameters


@pytest.mark.parametrize(
    "provided_hyperparameters",
    [
        {
            "learner_names": ["early_fusion", "lf_mlp"],
            "early_fusion": {
                "model.meta_transformer.checkpoint_path": "local_meta_transformer_path",
            },
        },
        {
            "early_fusion": {
                "model.meta_transformer.checkpoint_path": "local_meta_transformer_path",
            }
        },
    ],
)
def test_ensemble_hyperparameters(provided_hyperparameters):
    hyperparameters = update_ensemble_hyperparameters(
        presets=None,
        provided_hyperparameters=provided_hyperparameters,
    )
    provided_hyperparameters.pop("learner_names", None)
    for k, v in provided_hyperparameters.items():
        for kk, vv in provided_hyperparameters[k].items():
            assert hyperparameters[k][kk] == provided_hyperparameters[k][kk]


@pytest.mark.parametrize(
    "hyperparameters,key_mappings",
    [
        (
            {
                "optimization.learning_rate": 1e-3,
                "optimization.efficient_finetune": "bit_fit",
                "optimization.loss_function": "cross_entropy",
            },
            {
                "optimization.learning_rate": "optim.lr",
                "optimization.efficient_finetune": "optim.peft",
                "optimization.loss_function": "optim.loss_func",
            },
        ),
        (
            {
                "env.num_workers_evaluation": 100,
                "env.eval_batch_size_ratio": 5,
            },
            {
                "env.num_workers_evaluation": "env.num_workers_inference",
                "env.eval_batch_size_ratio": "env.inference_batch_size_ratio",
            },
        ),
        (
            {
                "data.label.numerical_label_preprocessing": "minmaxscaler",
            },
            {
                "data.label.numerical_label_preprocessing": "data.label.numerical_preprocessing",
            },
        ),
        (
            {
                "model.names": ["timm_image", "numerical_mlp", "categorical_mlp", "fusion_mlp"],
                "model.timm_image.max_img_num_per_col": 3,
                "model.categorical_mlp.drop_rate": 0.5,
                "model.numerical_mlp.drop_rate": 0.5,
                "model.numerical_mlp.d_token": 16,
                "model.fusion_mlp.weight": 0.1,
                "model.fusion_mlp.drop_rate": 0.5,
            },
            {
                "model.names": "model.names",
                "model.timm_image.max_img_num_per_col": "model.timm_image.max_image_num_per_column",
                "model.categorical_mlp.drop_rate": "model.categorical_mlp.dropout",
                "model.numerical_mlp.drop_rate": "model.numerical_mlp.dropout",
                "model.numerical_mlp.d_token": "model.numerical_mlp.token_dim",
                "model.fusion_mlp.weight": "model.fusion_mlp.aux_loss_weight",
                "model.fusion_mlp.drop_rate": "model.fusion_mlp.dropout",
            },
        ),
        (
            {
                "model.names": ["ft_transformer", "clip_image", "clip_text", "fusion_transformer"],
                "model.clip_image.data_types": ["image"],
                "model.clip_text.data_types": ["text"],
                "model.clip_image.max_img_num_per_col": 3,
                "model.fusion_transformer.n_blocks": 6,
                "model.fusion_transformer.attention_n_heads": 16,
                "model.fusion_transformer.ffn_d_hidden": 256,
                "model.ft_transformer.attention_n_heads": 16,
            },
            {
                "model.names": "model.names",
                "model.clip_image.data_types": "model.clip_image.data_types",
                "model.clip_text.data_types": "model.clip_text.data_types",
                "model.clip_image.max_img_num_per_col": "model.clip_image.max_image_num_per_column",
                "model.fusion_transformer.n_blocks": "model.fusion_transformer.num_blocks",
                "model.fusion_transformer.attention_n_heads": "model.fusion_transformer.attention_num_heads",
                "model.fusion_transformer.ffn_d_hidden": "model.fusion_transformer.ffn_hidden_size",
                "model.ft_transformer.attention_n_heads": "model.ft_transformer.attention_num_heads",
            },
        ),
    ],
)
def test_hyperparameter_backward_compatibility(hyperparameters, key_mappings):
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    predictor.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=0,
    )
    for k, v in hyperparameters.items():
        if k == "model.names":
            assert sorted(OmegaConf.select(predictor._learner._config, key_mappings[k])) == sorted(v)
        else:
            assert OmegaConf.select(predictor._learner._config, key_mappings[k]) == v
