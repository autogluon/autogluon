import os
import shutil
import warnings

import numpy.testing as npt
import pytest
from torch import Tensor

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import IA3_LORA

from datasets import load_dataset


@pytest.mark.parametrize("checkpoint_name", ["facebook/bart-base"])
@pytest.mark.parametrize("efficient_finetune", [None, IA3_LORA])
def test_predictor_with_bart(checkpoint_name, efficient_finetune):
    train_data = load_dataset("glue", 'mrpc')['train'].to_pandas().drop('idx', axis=1)
    test_data = load_dataset("glue", 'mrpc')['validation'].to_pandas().drop('idx', axis=1)
    predictor = MultiModalPredictor(label='label')
    predictor.fit(train_data,
                  hyperparameters={
                      "model.hf_text.checkpoint_name": "yuchenlin/BART0",
                      "optimization.max_epochs": 1,
                      "optimization.efficient_finetune": efficient_finetune,
                      "optimization.top_k": 1,
                      "optimization.top_k_average_method": "best",
                      "env.batch_size": 2,
                  },
                  time_limit=20
    )
    predictor.predict(test_data)
