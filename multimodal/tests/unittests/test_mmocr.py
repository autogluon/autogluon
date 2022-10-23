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


# TODO: when using crnn checkpoint, the results are wrong.
@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "abinet_academic",
        "sar_r31_parallel_decoder_academic",
        "seg_r31_1by16_fpnocr_academic",
        "nrtr_r31_1by16_1by8_academic",
    ],
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

    pred = predictor.predict({"image": [mmocr_image_name]})

    # original MMOCR model's output
    checkpoints = download(package="mmocr", configs=[det_ckpt_name, recog_ckpt_name], dest_root=".")
    det_ckpt, recog_ckpt = checkpoints[0], checkpoints[1]
    det_config_file = det_ckpt_name + ".py"
    recog_config_file = recog_ckpt_name + ".py"
    ocr = MMOCR(det_ckpt=det_ckpt, det_config=det_config_file, recog_ckpt=recog_ckpt, recog_config=recog_config_file)
    MMOCR_res = ocr.readtext(mmocr_image_name, output=None)
    print(pred)
    print(MMOCR_res)

    assert len(pred[0]) == len(MMOCR_res[0]["text"])

    for p in pred[0]:
        assert p in MMOCR_res[0]["text"]

    # assert pred[0][0] == MMOCR_res[0]["text"]
    # assert pred[1][0] == MMOCR_res[0]["score"]

if __name__ == '__main__':
    test_mmocr_text_recognition_inference("textsnake_r50_fpn_unet_1200e_ctw1500", "abinet_academic")