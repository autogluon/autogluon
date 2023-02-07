import os
import shutil
import warnings

import numpy.testing as npt
import pytest
from datasets import load_dataset
from torch import Tensor

from autogluon.multimodal import MultiModalPredictor


@pytest.mark.parametrize("eval_metric", ["spearmanr", "pearsonr"])
def test_predictor_with_spearman_pearson_eval(eval_metric):
    train_df = load_dataset('SetFit/stsb', split='train').to_pandas()
    predictor = MultiModalPredictor(label="label", eval_metric=eval_metric)
    predictor.fit(train_df, presets="medium_quality", time_limit=5)
    assert predictor._eval_metric_name == eval_metric
