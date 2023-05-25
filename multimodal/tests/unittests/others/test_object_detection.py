import os

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


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolox_s",
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_fit_then_evaluate_coco(checkpoint_name):
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=40,
    )

    # Evaluate
    predictor.evaluate(test_path)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_inference_list_str_dict(checkpoint_name):
    mmdet_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )

    pred = predictor.predict([mmdet_image_name] * 10)  # test batch inference
    assert len(pred) == 10  # test data size is 10

    pred = predictor.predict(mmdet_image_name)  # test batch inference
    assert len(pred) == 1  # test data size is 1

    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference
    assert len(pred) == 10  # test data size is 10


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_inference_xywh_output(checkpoint_name):
    mmdet_image_name = download_sample_images()

    xywh_predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "model.mmdet_image.output_bbox_format": "xywh",
            "env.num_gpus": 1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )
    xywh_preds = xywh_predictor.predict([mmdet_image_name] * 10)  # test batch inference
    assert len(xywh_preds) == 10  # test data size is 10

    xyxy_predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )
    xyxy_preds = xyxy_predictor.predict([mmdet_image_name] * 10)  # test batch inference
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


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_inference_df(checkpoint_name):

    data_dir = download_sample_dataset()

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )

    test_df = from_coco_or_voc(test_path)

    pred = predictor.predict(test_df.iloc[:100])


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_inference_coco(checkpoint_name):
    data_dir = download_sample_dataset()

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )

    pred = predictor.predict(test_path)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_save_and_load(checkpoint_name):
    data_dir = download_sample_dataset()

    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )

    pred = predictor.predict(test_path)

    model_save_subdir = predictor._model.save()

    new_predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": model_save_subdir,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )
    new_pred = new_predictor.predict(test_path)

    assert abs(pred["bboxes"][0][0]["score"] - new_pred["bboxes"][0][0]["score"]) < 1e-4


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_fit_then_inference_dict(checkpoint_name):
    data_dir = download_sample_dataset()
    mmdet_image_name = download_sample_images()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=30,
    )
    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_fit_then_inference_df(checkpoint_name):

    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=30,
    )

    df = from_coco_or_voc(test_path)
    pred = predictor.predict(df)  # test batch inference


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_fit_then_inference_coco(checkpoint_name):
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=30,
    )

    pred = predictor.predict(test_path)  # test batch inference


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_8xb24-320-300e_coco",
    ],
)
def test_mmdet_object_detection_fit_eval_predict_df(checkpoint_name):
    data_dir = download_sample_dataset()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    train_df = from_coco_or_voc(train_path)
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_df,
    )

    predictor.fit(
        train_df,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=30,
    )

    test_df = from_coco_or_voc(test_path)
    preds = predictor.predict(data=test_df)
    results = predictor.evaluate(data=test_df)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "yolov3_mobilenetv2_320_300e_coco",
    ],
)
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
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
        sample_data_path=train_df,
    )

    predictor.fit(
        train_df,
        hyperparameters={
            "optimization.learning_rate": 2e-4,
            "env.per_gpu_batch_size": 2,
        },
        time_limit=30,
    )
