import os
from autogluon.text import TextPredictor
from datasets import AmazonReviewSentimentCrossLingualDataset
from test_automm_predictor import verify_predictor_save_load
from utils import (
    download,
    protected_zip_extraction,
    get_home_dir,
)


def test_load_old_checkpoint():
    dataset = AmazonReviewSentimentCrossLingualDataset()
    sha1sum_id = "4ba096cdf6bd76c06386f2c27140db055e59c91b"
    checkpoint_name = "mdeberta-v3-base-checkpoint"
    save_path = os.path.join(get_home_dir(), "checkpoints")
    file_path = os.path.join(save_path, f"{checkpoint_name}.zip")
    checkpoint_path = os.path.join(get_home_dir(), "checkpoints", checkpoint_name)
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
    predictor = TextPredictor.load(checkpoint_path)
    verify_predictor_save_load(predictor, dataset.test_df)

    # continuous training
    predictor.fit(
        dataset.train_df,
        presets='multilingual',
        time_limit=10,
        hyperparameters={'optimization.top_k_average_method': 'uniform_soup'}
    )
    verify_predictor_save_load(predictor, dataset.test_df)
