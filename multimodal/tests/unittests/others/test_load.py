import os
import shutil

import numpy.testing as npt

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import PetFinderDataset


def test_load_intermediate_ckpt():
    download_dir = "./"
    train_data, test_data = shopee_dataset(download_dir=download_dir)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(train_data=train_data, time_limit=20)
    src_file = os.path.join(predictor.path, "model.ckpt")
    dest_file = os.path.join(predictor.path, "epoch=8-step=18.ckpt")
    shutil.copy(src_file, dest_file)
    loaded_predictor = MultiModalPredictor.load(path=dest_file)

    predictions = predictor.predict(test_data, as_pandas=False)
    predictions2 = loaded_predictor.predict(test_data, as_pandas=False)
    npt.assert_equal(predictions, predictions2)
    predictions_prob = predictor.predict_proba(test_data, as_pandas=False)
    predictions2_prob = loaded_predictor.predict_proba(test_data, as_pandas=False)
    npt.assert_equal(predictions_prob, predictions2_prob)


def test_load_fttransformer_pretrained_ckpt():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "model.names": ["ft_transformer"],
        "model.ft_transformer.checkpoint_name": "https://automl-mm-bench.s3.amazonaws.com/ft_transformer_pretrained_ckpt/iter_2k.ckpt",
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
    }
    predictor.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    predictor.evaluate(dataset.test_df)
