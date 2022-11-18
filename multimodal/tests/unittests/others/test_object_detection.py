import os
import pytest
import requests
from PIL import Image

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


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
        "faster_rcnn_r50_fpn_2x_coco",
        "yolov3_mobilenetv2_320_300e_coco",
        "mask_rcnn_r50_fpn_2x_coco",
        "detr_r50_8x2_150e_coco",
    ],
)
def test_mmdet_object_detection_inference_dict(checkpoint_name):
    mmdet_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,  # currently mmdet only support single gpu inference
        },
        problem_type="object_detection",
    )

    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference
    assert len(pred) == 10  # test data size is 100
    assert len(pred[0]) == 80  # COCO has 80 classes
    assert pred[0][0].ndim == 2  # two dimensions, (# of proposals, 5)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "faster_rcnn_r50_fpn_2x_coco",
        "yolov3_mobilenetv2_320_300e_coco",
        "detr_r50_8x2_150e_coco",
    ],
)
def test_mmdet_object_detection_inference_dict(checkpoint_name):
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

    # Evaluate
    predictor.evaluate(test_path)
