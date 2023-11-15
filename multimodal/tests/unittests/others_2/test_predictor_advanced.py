import os
import shutil

import numpy.testing as npt
import pytest
from datasets import load_dataset

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BIT_FIT, IA3, IA3_BIAS, IA3_LORA, LORA_BIAS, LORA_NORM, NORM_FIT
from autogluon.multimodal.models.timm_image import TimmAutoModelForImagePrediction
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils.unittest_datasets import AmazonReviewSentimentCrossLingualDataset


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "backbone,efficient_finetuning,pooling_mode,precision,expected_ratio,standalone",
    [
        ("t5-small", LORA_NORM, "mean", "bf16-mixed", 0.00557, True),
        ("google/flan-t5-small", IA3_LORA, "mean", "bf16-mixed", 0.006865, True),
        ("google/flan-t5-small", IA3, "cls", "bf16-mixed", 0.0004201, False),
        ("microsoft/deberta-v3-small", LORA_BIAS, "mean", "16-mixed", 0.001422, True),
        ("microsoft/deberta-v3-small", IA3_BIAS, "cls", "16-mixed", 0.00044566, False),
    ],
)
def test_predictor_gradient_checkpointing(
    backbone, efficient_finetuning, pooling_mode, precision, expected_ratio, standalone
):
    dataset = AmazonReviewSentimentCrossLingualDataset()
    train_data = dataset.train_df.sample(200)
    test_data = dataset.test_df.sample(50)
    save_path = f"gradient_checkpointing_{backbone}_{efficient_finetuning}_{pooling_mode}_{precision}"
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(label=dataset.label_columns[0], path=save_path)
    predictor.fit(
        train_data,
        standalone=standalone,
        hyperparameters={
            "model.names": ["hf_text"],
            "model.hf_text.checkpoint_name": backbone,
            "model.hf_text.pooling_mode": pooling_mode,
            "model.hf_text.gradient_checkpointing": True,
            "optimization.efficient_finetune": efficient_finetuning,
            "optimization.lr_decay": 1.0,
            "optimization.learning_rate": 1e-03,
            "optimization.max_epochs": 1,
            "env.precision": precision,
            "env.per_gpu_batch_size": 1,
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
            "env.num_gpus": -1,
        },
        time_limit=30,
    )
    predictions = predictor.predict(test_data, as_pandas=False)
    tunable_ratio = predictor.trainable_parameters / predictor.total_parameters
    npt.assert_allclose(tunable_ratio, expected_ratio, 2e-05, 2e-05)
    save_path = save_path + "_new"
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    predictor.save(save_path, standalone=standalone)
    new_predictor = MultiModalPredictor.load(save_path)
    new_predictions = new_predictor.predict(test_data, as_pandas=False)
    npt.assert_allclose(new_predictions, predictions)


def test_predictor_skip_final_val():
    download_dir = "./"
    save_path = "petfinder_checkpoint"
    train_df, tune_df = shopee_dataset(download_dir=download_dir)
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(label="label", path=save_path)
    hyperparameters = {
        "model.names": ["timm_image"],
        "model.timm_image.checkpoint_name": "ghostnet_100",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": "best",
        "optimization.val_check_interval": 1.0,
        "optimization.skip_final_val": True,
    }
    predictor.fit(
        train_data=train_df,
        tuning_data=tune_df,
        hyperparameters=hyperparameters,
        time_limit=5,
    )
    predictor_new = MultiModalPredictor.load(path=save_path)
    assert isinstance(predictor_new._learner._model, TimmAutoModelForImagePrediction)


def test_hyperparameters_in_terminal_format():
    download_dir = "./"
    train_df, tune_df = shopee_dataset(download_dir=download_dir)
    predictor = MultiModalPredictor(label="label")
    hyperparameters = [
        "model.names=[timm_image]",
        "model.timm_image.checkpoint_name=ghostnet_100",
        "env.num_workers=0",
        "env.num_workers_evaluation=0",
        "optimization.top_k_average_method=best",
        "optimization.val_check_interval=1.0",
    ]
    predictor.fit(
        train_data=train_df,
        tuning_data=tune_df,
        hyperparameters=hyperparameters,
        time_limit=2,
    )


@pytest.mark.parametrize("eval_metric", ["spearmanr", "pearsonr"])
def test_predictor_with_spearman_pearson_eval(eval_metric):
    train_df = load_dataset("SetFit/stsb", split="train").to_pandas()
    predictor = MultiModalPredictor(label="label", eval_metric=eval_metric)
    predictor.fit(train_df, presets="medium_quality", time_limit=5)
    assert predictor.eval_metric == eval_metric


@pytest.mark.parametrize("checkpoint_name", ["facebook/bart-base"])
@pytest.mark.parametrize("efficient_finetune", [None, IA3_LORA])
def test_predictor_with_bart(checkpoint_name, efficient_finetune):
    train_data = load_dataset("glue", "mrpc")["train"].to_pandas().drop("idx", axis=1).sample(500)
    test_data = load_dataset("glue", "mrpc")["validation"].to_pandas().drop("idx", axis=1).sample(20)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(
        train_data,
        hyperparameters={
            "model.hf_text.checkpoint_name": checkpoint_name,
            "optimization.max_epochs": 1,
            "optimization.efficient_finetune": efficient_finetune,
            "optimization.top_k": 1,
            "optimization.top_k_average_method": "best",
            "env.batch_size": 2,
        },
        time_limit=20,
    )
    predictor.predict(test_data)
