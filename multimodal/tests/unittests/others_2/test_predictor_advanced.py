import os
import shutil
import warnings

import numpy.testing as npt
import pytest
from torch import Tensor

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BIT_FIT, IA3, IA3_BIAS, IA3_LORA, LORA_BIAS, LORA_NORM, NORM_FIT

from ..utils.unittest_datasets import AmazonReviewSentimentCrossLingualDataset, PetFinderDataset


def _is_lazy_weight_tensor(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False


def total_parameters(model) -> int:
    return sum(p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in model.parameters())


def trainable_parameters(model) -> int:
    return sum(p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "backbone,efficient_finetuning,pooling_mode,precision,expected_ratio,standalone",
    [
        ("t5-small", LORA_NORM, "mean", "bf16", 0.00557, True),
        ("google/flan-t5-small", IA3_LORA, "mean", "bf16", 0.006865, True),
        ("google/flan-t5-small", IA3, "mean", "bf16", 0.0004201, False),
        ("microsoft/deberta-v3-small", LORA_BIAS, "mean", "16", 0.001422, True),
        ("microsoft/deberta-v3-small", IA3_BIAS, "mean", "16", 0.00044566, False),
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
            "env.per_gpu_batch_size": 2,
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
            "env.num_gpus": 1,
        },
        time_limit=30,
    )
    predictions = predictor.predict(test_data, as_pandas=False)
    tunable_ratio = trainable_parameters(predictor._model) / total_parameters(predictor._model)
    npt.assert_allclose(tunable_ratio, expected_ratio, 2e-05, 2e-05)
    save_path = save_path + "_new"
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    predictor.save(save_path, standalone=standalone)
    new_predictor = MultiModalPredictor.load(save_path)
    new_predictions = new_predictor.predict(test_data, as_pandas=False)
    npt.assert_allclose(new_predictions, predictions)


def test_predictor_skip_final_val():
    dataset = PetFinderDataset()
    save_path = "petfinder_checkpoint"

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric, path=save_path
    )
    hyperparameters = {
        "model.names": ["timm_image", "hf_text", "fusion_mlp"],
        "model.timm_image.checkpoint_name": "ghostnet_100",
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "optimization.skip_final_val": True,
    }
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=2,
        seed=42,
    )
    model_file = f"{save_path}/model.ckpt"
    assert os.path.isfile(model_file)
