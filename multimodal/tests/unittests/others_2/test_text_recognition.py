import numpy as np
import pytest
import requests
from mim.commands.download import download

try:
    from mmocr.utils.ocr import MMOCR
except ImportError:
    pytest.skip(
        'Skip the OCR test because there is no mmocr installed. Try to install it via mim install "mmocr<1.0"',
        allow_module_level=True,
    )

from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_text_recog.jpg"
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
@pytest.mark.skip(
    reason="Output format of OCR shall be changed to match with Object Detection. Since they both have ret_type=BBOX"
)
def test_mmocr_text_recognition_inference(checkpoint_name):
    mmocr_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmocr_text_recognition.checkpoint_name": checkpoint_name,
        },
        problem_type="ocr_text_recognition",
    )

    pred = predictor.predict({"image": [mmocr_image_name]})

    # original MMOCR model's output
    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")
    checkpoint = checkpoints[0]
    config_file = checkpoint_name + ".py"
    ocr = MMOCR(recog_ckpt=checkpoint, recog_config=config_file, det=None)
    MMOCR_res = ocr.readtext(mmocr_image_name, output=None)

    assert pred[0][0] == MMOCR_res[0]["text"]
    assert pred[1][0] == MMOCR_res[0]["score"]
