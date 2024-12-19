import os

import pytest
import timm
import transformers

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import AEDataset, PetFinderDataset


def test_dump_timm_image():
    download_dir = "./"
    model_dump_path = "./timm_image_test"
    base_model_name = "mobilenetv3_large_100"
    train_data, _ = shopee_dataset(download_dir=download_dir)
    predictor_1 = MultiModalPredictor(
        label="label",
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["timm_image_1"],
        "model.timm_image_1.checkpoint_name": base_model_name,
    }
    predictor_1.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )
    predictor_1.dump_model(save_path=model_dump_path)
    model = timm.create_model(
        model_name=base_model_name, checkpoint_path=f"{model_dump_path}/timm_image_1/pytorch_model.bin", num_classes=0
    )
    assert isinstance(model, timm.models.mobilenetv3.MobileNetV3)
    predictor_2 = MultiModalPredictor(
        label="label",
    )
    hyperparameters = {
        "optim.max_epochs": 1,
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
    base_model_name = "prajjwal1/bert-tiny"
    dataset = AEDataset()
    predictor_1 = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.hf_text.checkpoint_name": base_model_name,
        "env.num_workers": 0,  # https://github.com/pytorch/pytorch/issues/33296
    }
    predictor_1.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )
    predictor_1.dump_model(save_path=model_dump_path)

    model = transformers.AutoModel.from_pretrained(f"{model_dump_path}/hf_text")
    assert isinstance(model, transformers.models.bert.modeling_bert.BertModel)
    predictor_2 = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.hf_text.checkpoint_name": f"{model_dump_path}/hf_text",
        "env.num_workers": 0,  # https://github.com/pytorch/pytorch/issues/33296
    }
    predictor_2.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
        seed=42,
    )


def test_dump_fusion_model():
    model_dump_path = "./test_fusion_models"
    dataset = PetFinderDataset()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    hyperparameters = {
        "optim.max_epochs": 1,
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
    predictor.dump_model(save_path=model_dump_path)
    hf_text_dir = f"{model_dump_path}/hf_text"
    timm_image_dir = f"{model_dump_path}/timm_image"
    assert os.path.exists(hf_text_dir) and (len(os.listdir(hf_text_dir)) > 2) == True
    assert os.path.exists(timm_image_dir) and (len(os.listdir(timm_image_dir)) == 2) == True


# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
def test_mmdet_object_detection_save_and_load():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco",
            "env.num_gpus": -1,
        },
        problem_type="object_detection",
    )

    pred = predictor.predict(test_path)

    model_save_dir = predictor.dump_model()
    detection_model_save_subdir = os.path.join(model_save_dir, predictor._learner._model.prefix)

    new_predictor = MultiModalPredictor(
        hyperparameters={"model.mmdet_image.checkpoint_name": detection_model_save_subdir, "env.num_gpus": -1},
        problem_type="object_detection",
    )
    new_pred = new_predictor.predict(test_path)

    assert abs(pred["bboxes"][0][0]["score"] - new_pred["bboxes"][0][0]["score"]) < 1e-4
