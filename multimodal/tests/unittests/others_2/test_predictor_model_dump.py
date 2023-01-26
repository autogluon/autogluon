import os

import timm
import transformers

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils.unittest_datasets import AEDataset, PetFinderDataset

ALL_DATASETS = {
    "petfinder": PetFinderDataset(),
    "ae": AEDataset(),
}


def test_dump_timm_image():
    download_dir = "./"
    model_dump_path = "./timm_image_test"
    base_model_name = "mobilenetv3_large_100"
    train_data, _ = shopee_dataset(download_dir=download_dir)
    predictor_1 = MultiModalPredictor(
        label="label",
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["timm_image_1"],
        "model.timm_image_1.checkpoint_name": base_model_name,
    }
    predictor_1.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )
    predictor_1.dump_model(path=model_dump_path)
    model = timm.create_model(
        model_name=base_model_name, checkpoint_path=f"{model_dump_path}/timm_image_1/pytorch_model.bin", num_classes=0
    )
    assert isinstance(model, timm.models.mobilenetv3.MobileNetV3)
    predictor_2 = MultiModalPredictor(
        label="label",
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.timm_image.checkpoint_name": f"{model_dump_path}/timm_image_1",
    }
    predictor_2.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )


def test_dump_hf_text():
    model_dump_path = "./hf_text_test"
    base_model_name = "nlpaueb/legal-bert-small-uncased"
    dataset = ALL_DATASETS["ae"]
    predictor_1 = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.hf_text.checkpoint_name": base_model_name,
    }
    predictor_1.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )
    predictor_1.dump_model(path=model_dump_path)

    model = transformers.AutoModel.from_pretrained(f"{model_dump_path}/hf_text")
    assert isinstance(model, transformers.models.bert.modeling_bert.BertModel)
    predictor_2 = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.hf_text.checkpoint_name": f"{model_dump_path}/hf_text",
    }
    predictor_2.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )


def test_fusion_model_dump():
    model_dump_path = "./test_fusion_models"
    dataset = ALL_DATASETS["petfinder"]
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["timm_image", "hf_text", "fusion_mlp"],
        "model.timm_image.checkpoint_name": "ghostnet_100",
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
    }
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )
    predictor.dump_model(path=model_dump_path)
    hf_text_dir = f"{model_dump_path}/hf_text"
    timm_image_dir = f"{model_dump_path}/timm_image"
    assert os.path.exists(hf_text_dir) and (len(os.listdir(hf_text_dir)) > 2) == True
    assert os.path.exists(timm_image_dir) and (len(os.listdir(timm_image_dir)) == 2) == True
