import os
import shutil
import tempfile

import pytest
from transformers import AlbertTokenizer, AlbertTokenizerFast

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import TEXT

from ..utils import AEDataset


@pytest.mark.parametrize(
    "checkpoint_name,use_fast,tokenizer_type",
    [
        (
            "albert-base-v2",
            None,
            AlbertTokenizerFast,
        ),
        (
            "albert-base-v2",
            True,
            AlbertTokenizerFast,
        ),
        (
            "albert-base-v2",
            False,
            AlbertTokenizer,
        ),
    ],
)
def test_tokenizer_use_fast(checkpoint_name, use_fast, tokenizer_type):
    dataset = AEDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "data.categorical.convert_to_text": True,
        "data.numerical.convert_to_text": True,
        "model.hf_text.checkpoint_name": checkpoint_name,
    }
    if use_fast is not None:
        hyperparameters["model.hf_text.use_fast"] = use_fast

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=5,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )
    assert isinstance(predictor._learner._data_processors[TEXT][0].tokenizer, tokenizer_type)
