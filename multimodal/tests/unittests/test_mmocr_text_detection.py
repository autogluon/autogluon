import numpy as np
import pytest
import requests
from mim.commands.download import download
from mmocr.utils.ocr import MMOCR
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
            "model.mmocr_text.det_ckpt_name": checkpoint_name,
        },
        pipeline="ocr_text",
    )

    # two dimensions, (num of text lines, 2 * num of coordinate points)
    pred = predictor.predict({"image": [mmocr_image_name] * 10})

    # original MMOCR model's output
    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")
    checkpoint = checkpoints[0]
    config_file = checkpoint_name + ".py"
    ocr = MMOCR(det_ckpt=checkpoint, det_config=config_file, recog=None)
    MMOCR_res = ocr.readtext([mmocr_image_name] * 10, output=None)

    # compare the outputs of original model's output and our model
    pred = pred['bbox']
    assert len(pred) == len(MMOCR_res)  # num of text lines

    for p, m in zip(pred, MMOCR_res):
        assert len(p) == len(m["boundary_result"])  # num of bounding boxs

        for i in range(len(p)):
            assert len(p[i]) == len(m["boundary_result"][i]) # 2 * num of coordinate points

            for j in range(len(p[i])):
                assert abs(p[i][j] - m["boundary_result"][i][j]) <= 1e-6