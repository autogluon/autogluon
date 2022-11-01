import numpy as np
import pytest
import requests
from mim.commands.download import download
from mmocr.utils.ocr import MMOCR
from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_text_ocr.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mmocr_image_name = "demo.jpg"
    image.save(mmocr_image_name)

    return mmocr_image_name


@pytest.mark.parametrize(
    "det_ckpt_name,recog_ckpt_name",
    [("textsnake_r50_fpn_unet_1200e_ctw1500", "abinet_academic")],
)
def test_mmocr_text_recognition_inference(det_ckpt_name, recog_ckpt_name):
    mmocr_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmocr_text.det_ckpt_name": det_ckpt_name,
            "model.mmocr_text.recog_ckpt_name": recog_ckpt_name,
        },
        pipeline="ocr_text",
    )

    pred = predictor.predict({"image": [mmocr_image_name] * 10})
    pred = pred["text"]

    # original MMOCR model's output
    checkpoints = download(package="mmocr", configs=[det_ckpt_name, recog_ckpt_name], dest_root=".")
    det_ckpt, recog_ckpt = checkpoints[0], checkpoints[1]
    det_config_file = det_ckpt_name + ".py"
    recog_config_file = recog_ckpt_name + ".py"
    ocr = MMOCR(det_ckpt=det_ckpt, det_config=det_config_file, recog_ckpt=recog_ckpt, recog_config=recog_config_file)
    MMOCR_res = ocr.readtext([mmocr_image_name] * 10, output=None)

    # compare
    assert len(pred) == len(MMOCR_res)  # number of input samples
    for p, m in zip(pred, MMOCR_res):
        for text in p["text"]:
            assert text in m["text"]
