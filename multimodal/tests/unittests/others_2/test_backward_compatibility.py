import os
import shutil
import tempfile

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import download

from ..predictor.test_predictor import verify_predictor_save_load
from ..others.unittest_datasets import AmazonReviewSentimentCrossLingualDataset
from ..others.utils import get_home_dir, protected_zip_extraction


@pytest.mark.parametrize("cls", [MultiModalPredictor])
def test_load_old_checkpoint(cls):
    dataset = AmazonReviewSentimentCrossLingualDataset()
    sha1sum_id = "4ba096cdf6bd76c06386f2c27140db055e59c91b"
    checkpoint_name = "mdeberta-v3-base-checkpoint"
    save_path = os.path.join(get_home_dir(), "checkpoints")
    file_path = os.path.join(save_path, f"{checkpoint_name}.zip")
    checkpoint_path = os.path.join(get_home_dir(), "checkpoints", checkpoint_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    download(
        url=f"s3://automl-mm-bench/unit-tests-0.4/checkpoints/{checkpoint_name}.zip",
        path=file_path,
        sha1_hash=sha1sum_id,
    )
    protected_zip_extraction(
        file_path,
        sha1_hash=sha1sum_id,
        folder=save_path,
    )
    predictor = cls.load(checkpoint_path)
    verify_predictor_save_load(predictor, dataset.test_df.sample(4), cls=cls)

    # continuous training
    predictor.fit(
        dataset.train_df,
        presets="multilingual",
        time_limit=10,
        hyperparameters={"optimization.top_k_average_method": "uniform_soup", "env.num_gpus": 1},
    )
