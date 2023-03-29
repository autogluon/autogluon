import os
import shutil
import tempfile

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import download
from autogluon.multimodal.utils.misc import shopee_dataset

from ..predictor.test_predictor import verify_predictor_save_load
from ..utils.unittest_datasets import AmazonReviewSentimentCrossLingualDataset
from ..utils.utils import get_home_dir, protected_zip_extraction


def test_load_old_checkpoint_text_only():
    dataset = AmazonReviewSentimentCrossLingualDataset()
    sha1_hash = "4ba096cdf6bd76c06386f2c27140db055e59c91b"
    checkpoint_name = "mdeberta-v3-base-checkpoint"
    save_path = os.path.join(get_home_dir(), "checkpoints")
    file_path = os.path.join(save_path, f"{checkpoint_name}.zip")
    checkpoint_path = os.path.join(get_home_dir(), "checkpoints", checkpoint_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    download(
        url=f"s3://automl-mm-bench/unit-tests-0.4/checkpoints/{checkpoint_name}.zip",
        path=file_path,
        sha1_hash=sha1_hash,
    )
    protected_zip_extraction(
        file_path,
        sha1_hash=sha1_hash,
        folder=save_path,
    )
    predictor = MultiModalPredictor.load(checkpoint_path)
    verify_predictor_save_load(predictor, dataset.test_df.sample(4), cls=MultiModalPredictor)

    # continuous training
    predictor.fit(
        dataset.train_df,
        presets="multilingual",
        time_limit=10,
        hyperparameters={"optimization.top_k_average_method": "uniform_soup", "env.num_gpus": 1},
    )


def test_v0_6_2_checkpoint_timm_image():
    train_df, test_df = shopee_dataset("./ag_automm_tutorial_imgcls")

    file_path = "convnext_nano_shopee_0.6.2_backward_compatible_test.zip"
    save_path = os.path.join(get_home_dir(), "checkpoints_0.6.2_backward_compatible")
    checkpoint_path = os.path.join(save_path, "convnext_nano_shopee")
    sha1_hash = "b0b36fd076b3e0ab599f917c9b2924aef84a02b6"
    download(
        url="https://automl-mm-bench.s3.amazonaws.com/unit-tests/convnext_nano_shopee_0.6.2_backward_compatible_test.zip",
        path=file_path,
        sha1_hash=sha1_hash,
    )
    protected_zip_extraction(
        file_path,
        sha1_hash=sha1_hash,
        folder=save_path,
    )
    predictor = MultiModalPredictor.load(checkpoint_path)
    print(predictor._config)
    assert predictor._config.model.timm_image.image_size is None

    predictor.fit(
        train_df,
        hyperparameters={"model.timm_image.image_size": 288},
        time_limit=10,
    )
    assert predictor._config.model.timm_image.image_size == 288
