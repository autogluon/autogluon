import numpy as np
import pytest
import requests
from mim.commands.download import download

try:
    import mmocr
    from mmocr.utils.ocr import MMOCR
except (ImportError, ModuleNotFoundError):
    pytest.skip("MMOCR is not installed. Skip this test.", allow_module_level=True)

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
@pytest.mark.skip(
    reason="Output format of OCR shall be changed to match with Object Detection. Since they both have ret_type=BBOX"
)
def test_mmocr_text_detection_inference(checkpoint_name):
    mmocr_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmocr_text_detection.checkpoint_name": checkpoint_name,
        },
        problem_type="ocr_text_detection",
    )

    # two dimensions, (num of text lines, 2 * num of coordinate points)
    pred = predictor.predict({"image": [mmocr_image_name]})

    # original MMOCR model's output
    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")
    checkpoint = checkpoints[0]
    config_file = checkpoint_name + ".py"
    ocr = MMOCR(det_ckpt=checkpoint, det_config=config_file, recog=None)
    MMOCR_res = ocr.readtext(mmocr_image_name, output=None)

    # compare the outputs of original model's output and our model
    assert len(pred) == len(MMOCR_res[0]["boundary_result"])  # num of text lines

    for i in range(len(pred)):
        p = pred[i]
        m = MMOCR_res[0]["boundary_result"][i]
        assert len(p) == len(m)  # 2 * num of coordinate points

        for j in range(len(p)):
            assert abs(p[j] - m[j]) <= 1e-6
