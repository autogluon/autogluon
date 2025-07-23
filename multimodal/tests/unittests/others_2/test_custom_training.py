import os
import shutil

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BIT_FIT, IA3, IA3_BIAS, IA3_LORA, LORA, LORA_BIAS, LORA_NORM, NORM_FIT
from autogluon.multimodal.models.timm_image import TimmAutoModelForImagePrediction
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import AmazonReviewSentimentCrossLingualDataset, PetFinderDataset


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "backbone,peft,pooling_mode,precision,expected_ratio,standalone",
    [
        ("t5-small", LORA_NORM, "mean", "bf16-mixed", 0.00557, True),
        ("google/flan-t5-small", IA3_LORA, "mean", "bf16-mixed", 0.006865, True),
        ("google/flan-t5-small", IA3, "cls", "bf16-mixed", 0.0004201, False),
        ("microsoft/deberta-v3-small", LORA_BIAS, "mean", "16-mixed", 0.001422, True),
        ("microsoft/deberta-v3-small", LORA, "cls", "16-mixed", 0.0010533, False),
    ],
)
def test_gradient_checkpointing(backbone, peft, pooling_mode, precision, expected_ratio, standalone):
    dataset = AmazonReviewSentimentCrossLingualDataset()
    train_data = dataset.train_df.sample(200)
    test_data = dataset.test_df.sample(50)
    save_path = f"gradient_checkpointing_{backbone}_{peft}_{pooling_mode}_{precision}"
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
            "optim.peft": peft,
            "optim.lr_decay": 1.0,
            "optim.lr": 1e-03,
            "optim.max_epochs": 1,
            "env.precision": precision,
            "env.per_gpu_batch_size": 1,
            "env.num_workers": 0,
            "env.num_workers_inference": 0,
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


def test_skip_final_val():
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
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": "best",
        "optim.val_check_interval": 1.0,
        "optim.skip_final_val": True,
    }
    predictor.fit(
        train_data=train_df,
        tuning_data=tune_df,
        hyperparameters=hyperparameters,
        time_limit=5,
    )
    predictor_new = MultiModalPredictor.load(path=save_path)
    assert isinstance(predictor_new._learner._model, TimmAutoModelForImagePrediction)


def test_fit_with_data_path():
    download_dir = "./"
    train_csv_file = "shopee_train_data.csv"
    train_data, _ = shopee_dataset(download_dir=download_dir)
    train_data.to_csv(train_csv_file)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(train_data=train_csv_file, time_limit=0)
    predictor.fit(train_data=train_csv_file, tuning_data=train_csv_file, time_limit=0)


def test_train_with_cpu_only():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["ft_transformer"],
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
        "env.accelerator": "cpu",
    }
    predictor.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    predictor.evaluate(dataset.test_df)
    predictor.predict(dataset.test_df)
