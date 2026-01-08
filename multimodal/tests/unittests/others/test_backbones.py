import os
import shutil

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import IA3_LORA
from autogluon.multimodal.models import HFAutoModelForTextPrediction, TimmAutoModelForImagePrediction
from autogluon.multimodal.utils import download
from datasets import load_dataset

from ..utils import PetFinderDataset, get_home_dir, verify_no_redundant_model_configs, verify_predictor_save_load


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "distilroberta-base",
        "huawei-noah/TinyBERT_General_4L_312D",
        "google/electra-base-discriminator",
        "microsoft/deberta-v3-base",
        "bert-base-uncased",
        "xlm-roberta-base",
        "microsoft/deberta-base",
        "roberta-base",
        "distilbert-base-uncased",
        "bert-base-chinese",
        "gpt2",
    ],
)
def test_hf_text_init(checkpoint_name):
    model = HFAutoModelForTextPrediction(prefix="hf_text", checkpoint_name=checkpoint_name, num_classes=5)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_base_patch4_window7_224",
        "vit_small_patch16_384",
        "resnet18",
        "legacy_seresnet18",
        "regnety_002",
    ],
)
def test_timm_image_init(checkpoint_name):
    model = TimmAutoModelForImagePrediction(prefix="timm_image", checkpoint_name=checkpoint_name, num_classes=5)


@pytest.mark.parametrize("checkpoint_name", ["facebook/bart-base"])
@pytest.mark.parametrize("peft", [None, IA3_LORA])
def test_backbone_bart(checkpoint_name, peft):
    train_data = load_dataset("glue", "mrpc")["train"].to_pandas().drop("idx", axis=1).sample(500)
    test_data = load_dataset("glue", "mrpc")["validation"].to_pandas().drop("idx", axis=1).sample(20)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(
        train_data,
        hyperparameters={
            "model.hf_text.checkpoint_name": checkpoint_name,
            "optim.max_epochs": 1,
            "optim.peft": peft,
            "optim.top_k": 1,
            "optim.top_k_average_method": "best",
            "env.batch_size": 2,
        },
        time_limit=20,
    )
    predictor.predict(test_data)


@pytest.mark.skip(
    reason="Skip this test because the meta-transformer checkpoint needs to be put into the s3 bucket first."
)
def test_meta_transformer():
    model_name = "Meta-Transformer_base_patch16_encoder"
    model_version = "base"
    model_path = f"./{model_name}"
    if not os.path.isfile(model_path):
        download(
            url=f"s3://automl-mm-bench/meta-transformer/{model_name}",
            path=model_path,
        )
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["meta_transformer"],
        "model.meta_transformer.checkpoint_path": model_path,
        "model.meta_transformer.model_version": model_version,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
    }
    save_path = os.path.join(get_home_dir(), "outputs", "meta-transformer")

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )
    verify_no_redundant_model_configs(predictor)
    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)
