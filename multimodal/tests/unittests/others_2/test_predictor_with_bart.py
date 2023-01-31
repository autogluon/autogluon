import os
import shutil
import warnings

import numpy.testing as npt
import pytest
from torch import Tensor

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BIT_FIT, IA3, IA3_BIAS, IA3_LORA, LORA_BIAS, LORA_NORM, NORM_FIT

from datasets import load_dataset


@pytest.mark.parametrize("checkpoint_name", ["facebook/bart-base"])
def test_predictor_with_bart(checkpoint_name):
    train_data = load_dataset("glue", 'mrpc')['train'].to_pandas().drop('idx', axis=1)
    test_data = load_dataset("glue", 'mrpc')['validation'].to_pandas().drop('idx', axis=1)
    predictor = MultiModalPredictor(label='label')
    predictor.fit(train_data,
                  hyperparameters={
                      "model.hf_text.checkpoint_name": "yuchenlin/BART0",
                      "optimization.max_epochs": 1
                  },
                  time_limit=180
    )
    predictor.predict(test_data)
