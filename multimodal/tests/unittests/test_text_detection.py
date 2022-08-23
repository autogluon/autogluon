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
    [
        "textsnake_r50_fpn_unet_1200e_ctw1500"
    ],
)

def test_mmocr_text_detection_inference(checkpoint_name):
    mmocr_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmocr_image.checkpoint_name": checkpoint_name,
        },
        pipeline= "ocr_text_detection"
    )

    predictor.predict({"image": [mmocr_image_name]})
  

# if __name__ == '__main__':
#     test_mmocr_text_detection_inference("textsnake_r50_fpn_unet_1200e_ctw1500")