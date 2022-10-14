import os
import shutil
import tempfile
import numpy.testing as npt

from autogluon.multimodal.constants import BINARY, MULTICLASS
from autogluon.text import TextPredictor
from unittest_datasets import AmazonReviewSentimentCrossLingualDataset
from utils import (
    download,
    protected_zip_extraction,
    get_home_dir,
)


def verify_predictor_save_load(predictor, df, verify_embedding=True):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df, as_pandas=False)
        loaded_predictor = TextPredictor.load(root)
        predictions2 = loaded_predictor.predict(df, as_pandas=False)
        predictions2_df = loaded_predictor.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2, predictions2_df.to_numpy())
        if predictor.problem_type in [BINARY, MULTICLASS]:
            predictions_prob = predictor.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            embeddings = predictor.extract_embedding(df)
            assert embeddings.shape[0] == len(df)


def test_load_old_checkpoint():
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
    predictor = TextPredictor.load(checkpoint_path)
    verify_predictor_save_load(predictor, dataset.test_df)

    # continuous training
    predictor.fit(
        dataset.train_df,
        presets="multilingual",
        time_limit=30,
        hyperparameters={"optimization.top_k_average_method": "uniform_soup"},
    )
    verify_predictor_save_load(predictor, dataset.test_df)
