import os

import pytest
import requests
from PIL import Image

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_coco_or_voc

from autogluon.multimodal.utils.misc import shopee_dataset


def test_classification_str_list_input():
    download_dir = './ag_automm_tutorial_imgcls'
    train_data, test_data = shopee_dataset(download_dir)

    import uuid
    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
    predictor = MultiModalPredictor(label="label", path=model_path)
    predictor.fit(
        train_data=train_data,
        time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]['image']

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)
