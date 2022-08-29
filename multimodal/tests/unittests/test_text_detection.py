import numpy as np
import pytest
import requests
from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_text_det.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mmocr_image_name = "demo.jpg"
    image.save(mmocr_image_name)

    return mmocr_image_name


@pytest.mark.parametrize(
    "checkpoint_name",
    ["textsnake_r50_fpn_unet_1200e_ctw1500"],
)
def test_mmocr_text_detection_inference(checkpoint_name):
    mmocr_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmocr_text_detection.checkpoint_name": checkpoint_name,
        },
        pipeline="ocr_text_detection",
    )

    # two dimensions, (num of text lines, 2 * num of coordinate points)
    pred = predictor.predict({"image": [mmocr_image_name]})

    assert len(pred[0]) == 9  # num of text lines
    true_res_list = [751, 477, 757, 809, 977, 1239, 885, 1043, 1039]  # from MMOCR

    for i, line in enumerate(pred[0]):
        assert len(line) == true_res_list[i]  # 2 * num of coordinate points
