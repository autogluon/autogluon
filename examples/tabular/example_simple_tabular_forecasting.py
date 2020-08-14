""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon.utils.tabular.ml.constants import FORECAST
from autogluon import TabularPrediction as task
import autogluon as ag
# Training time:
# Standard library imports
from functools import partial
from random import randint
from typing import List, Tuple

# Third-party imports
import numpy as np
import pytest
import pandas as pd

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


def make_dummy_datasets(ts_n=1, ts_length=100, freq="D", prediction_length=5):
    targets = []
    for i in range(ts_n):
        for j in range(ts_length):
            targets.append(j)
    ts_index = []
    for i in range(ts_n):
        for j in range(ts_length):
            ts_index.append(i)
    n = ts_n * ts_length
    date = pd.date_range(start="2020", freq=freq, periods=ts_length)
    return pd.DataFrame({"index": ts_index,"date": date, "target": targets, "freq": [freq for i in range(n)], "prediction_length": [prediction_length for i in range(n)]})


train_data = make_dummy_datasets(ts_length=200)
test_data = make_dummy_datasets(ts_length=10)

print(train_data)
hyperparams = {'MQCNN': {'context_length': ag.Int(10, 20)} }
savedir = 'ag_models/'
predictor = task.fit(train_data=train_data, hyperparameters=hyperparams, output_directory=savedir, label="target", problem_type=FORECAST, id_columns=["index"])
# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:  predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, presets='best_quality', eval_metric=YOUR_METRIC_NAME)
results = predictor.fit_summary()

