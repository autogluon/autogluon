import json
import os

import numpy as np
import pytest
import requests
from PIL import Image

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_coco_or_voc


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/demo/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mmdet_image_name = "demo.jpg"
    image.save(mmdet_image_name)

    return mmdet_image_name


def download_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")

    return data_dir


# TODO: Pytest does not support DDP
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolox_s"])
def test_mmdet_object_detection_fit_basics(checkpoint_name):
    mmdet_image_name = download_sample_images()
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={"model.mmdet_image.checkpoint_name": checkpoint_name, "env.num_gpus": -1},
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(train_path, hyperparameters={"optim.lr": 2e-4, "env.per_gpu_batch_size": 2}, time_limit=40)

    # Evaluate on COCO format data
    predictor.evaluate(test_path)

    # Inference on COCO format data
    pred = predictor.predict(test_path)  # test batch inference

    # Inference on dictionary format data
    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference

    test_df = from_coco_or_voc(test_path)

    # Inference on dataframe format data
    pred = predictor.predict(test_df.iloc[:100])

    # Inference on COCO format data without annotations
    test_path_with_images_only = os.path.join(data_dir, "Annotations", "test_cocoformat_nolabel.json")
    with open(test_path, "r") as f:
        test_data = json.load(f)
    test_data_images_only = {}
    test_data_images_only["images"] = test_data["images"][:100]
    with open(test_path_with_images_only, "w+") as f_wo_ann:
        json.dump(test_data_images_only, f_wo_ann)
    pred = predictor.predict(test_path_with_images_only)


# TODO: Pytest does not support DDP
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolov3_mobilenetv2_8xb24-320-300e_coco"])
def test_mmdet_object_detection_inference_basics(checkpoint_name):
    mmdet_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": -1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )

    # Inference on list format data
    pred = predictor.predict([mmdet_image_name] * 20)  # test batch inference
    assert len(pred) == 20  # test data size is 20

    # Inference on single image path
    pred = predictor.predict(mmdet_image_name)  # test single entry data inference
    assert len(pred) == 1  # test data size is 1

    # Inference on dict format data
    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference
    assert len(pred) == 10  # test data size is 10

    data_dir = download_sample_dataset()

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    test_path_with_images_only = os.path.join(data_dir, "Annotations", "test_cocoformat_nolabel.json")
    with open(test_path, "r") as f:
        test_data = json.load(f)
    test_data_images_only = {}
    test_data_images_only["images"] = test_data["images"][:100]
    with open(test_path_with_images_only, "w+") as f_wo_ann:
        json.dump(test_data_images_only, f_wo_ann)

    # Inference on COCO format data
    test_df = from_coco_or_voc(test_path)

    # Inference on dataframe format data
    pred = predictor.predict(test_df.iloc[:100])

    # Inference on data without annotations
    pred = predictor.predict(test_path_with_images_only)

    # Save inference in COCO on data without annotations
    pred = predictor.predict(test_path_with_images_only, save_results=True)

    # Save inference in Pandas Dataframe (.csv) on data without annotations
    pred = predictor.predict(test_path_with_images_only, save_results=True, as_coco=False)


# TODO: FIX DDP multi runs!
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolov3_mobilenetv2_8xb24-320-300e_coco"])
def test_mmdet_object_detection_inference_xywh_output(checkpoint_name):
    mmdet_image_name = download_sample_images()

    xywh_predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "model.mmdet_image.output_bbox_format": "xywh",
            "env.num_gpus": -1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )
    xywh_preds = xywh_predictor.predict([mmdet_image_name] * 10, as_pandas=True)  # test batch inference
    assert len(xywh_preds) == 10  # test data size is 10

    xyxy_predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": -1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )
    xyxy_preds = xyxy_predictor.predict([mmdet_image_name] * 10, as_pandas=True)  # test batch inference
    assert len(xyxy_preds) == 10  # test data size is 10

    xywh_bbox = xywh_preds.iloc[0]["bboxes"][0]
    xyxy_bbox = xyxy_preds.iloc[0]["bboxes"][0]
    x, y, w, h = xywh_bbox["bbox"]
    x1, y1, x2, y2 = xyxy_bbox["bbox"]
    assert xywh_bbox["class"] == xyxy_bbox["class"]
    assert abs(xywh_bbox["score"] - xyxy_bbox["score"]) < 1e-4
    assert abs(x - x1) < 1e-4
    assert abs(y - y1) < 1e-4
    assert abs(x2 - x1 + 1 - w) < 1e-4
    assert abs(y2 - y1 + 1 - h) < 1e-4


# TODO: FIX DDP multi runs!
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolov3_mobilenetv2_8xb24-320-300e_coco"])
def test_mmdet_object_detection_save_and_load(checkpoint_name):
    data_dir = download_sample_dataset()

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={"model.mmdet_image.checkpoint_name": checkpoint_name, "env.num_gpus": -1},
        problem_type="object_detection",
    )

    preds = predictor.predict(test_path)

    model_save_subdir = predictor._learner._model.save()

    new_predictor = MultiModalPredictor(
        hyperparameters={"model.mmdet_image.checkpoint_name": model_save_subdir, "env.num_gpus": -1},
        problem_type="object_detection",
    )
    new_preds = new_predictor.predict(test_path)

    for batch_idx in range(len(preds)):
        for in_batch_idx in range(len(preds[batch_idx])):
            # Convert tensors to numpy arrays for element-wise comparison
            pred_scores = preds[batch_idx][in_batch_idx]["scores"].detach().cpu().numpy()
            new_pred_scores = new_preds[batch_idx][in_batch_idx]["scores"].detach().cpu().numpy()
            # Check if all differences are within tolerance
            assert (np.abs(pred_scores - new_pred_scores) < 1e-4).all(), (
                f"{preds[batch_idx][in_batch_idx]}\n{new_preds[batch_idx][in_batch_idx]}"
            )


# TODO: FIX DDP multi runs!
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolov3_mobilenetv2_8xb24-320-300e_coco"])
def test_mmdet_object_detection_fit_eval_predict_df(checkpoint_name):
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    train_df = from_coco_or_voc(train_path)
    predictor = MultiModalPredictor(
        hyperparameters={"model.mmdet_image.checkpoint_name": checkpoint_name, "env.num_gpus": -1},
        problem_type="object_detection",
        sample_data_path=train_df,
    )

    predictor.fit(train_df, hyperparameters={"optim.lr": 2e-4, "env.per_gpu_batch_size": 2}, time_limit=30)

    test_df = from_coco_or_voc(test_path)
    preds = predictor.predict(data=test_df)
    results = predictor.evaluate(data=test_df)


# TODO: Pytest does not support DDP
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
@pytest.mark.parametrize("checkpoint_name", ["yolov3_mobilenetv2_8xb24-320-300e_coco"])
def test_mmdet_object_detection_fit_with_freeze_backbone(checkpoint_name):
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    train_df = from_coco_or_voc(train_path)
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "model.mmdet_image.frozen_layers": ["backbone"],
            "env.num_gpus": -1,
        },
        problem_type="object_detection",
        sample_data_path=train_df,
    )

    predictor.fit(train_df, hyperparameters={"optim.lr": 2e-4, "env.per_gpu_batch_size": 2}, time_limit=30)


# TODO: FIX DDP multi runs!
# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
def test_detector_hyperparameters_consistency():
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    train_df = from_coco_or_voc(train_path)

    hyperparameters = {
        "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco",
        "env.num_gpus": -1,
    }

    # pass hyperparameters to init()
    predictor = MultiModalPredictor(
        problem_type="object_detection", sample_data_path=train_df, hyperparameters=hyperparameters
    )
    predictor.fit(train_df, time_limit=10)

    # pass hyperparameters to fit()
    predictor_2 = MultiModalPredictor(problem_type="object_detection", sample_data_path=train_df)
    predictor_2.fit(train_df, hyperparameters=hyperparameters, time_limit=10)
    assert predictor._learner._config == predictor_2._learner._config


# TODO: Issue #4126 Skipping object detection tests due to incompatibility of mmdet with Torch 2.2
@pytest.mark.torch_mmdet
def test_detector_coco_root_setup():
    data_dir = download_sample_dataset()
    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    train_df = from_coco_or_voc(train_path)

    hyperparameters = {
        "model.mmdet_image.coco_root": "../",
        "env.num_gpus": 1,  # no need to test multigpu
    }

    # pass hyperparameters to init()
    predictor = MultiModalPredictor(
        problem_type="object_detection",
        sample_data_path=train_df,
        hyperparameters=hyperparameters,
    )
    predictor.fit(train_df, time_limit=10)
