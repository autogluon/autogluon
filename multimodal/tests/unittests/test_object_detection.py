import numpy as np
import pytest
import requests
from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/demo/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mmdet_image_name = "demo.jpg"
    image.save(mmdet_image_name)

    return mmdet_image_name


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "faster_rcnn_r50_fpn_2x_coco",
        "yolov3_mobilenetv2_320_300e_coco",
        "cascade_rcnn_x101_64x4d_fpn_20e_coco",
        "detr_r50_8x2_150e_coco",
    ],
)
def test_mmdet_object_detection_inference(checkpoint_name):
    mmdet_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
        },
        pipeline="object_detection",
    )

    pred = predictor.predict({"image": [mmdet_image_name]})
    assert len(pred[0][0]) == 80  # COCO has 80 classes
    assert pred[0][0][0].ndim == 2  # two dimensions, (# of proposals, 5)
