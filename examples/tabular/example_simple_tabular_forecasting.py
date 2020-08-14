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

# def make_dummy_datasets_with_features(
#     num_ts: int = 5,
#     start: str = "2018-01-01",
#     freq: str = "D",
#     min_length: int = 5,
#     max_length: int = 10,
#     prediction_length: int = 3,
#     cardinality: List[int] = [],
#     num_feat_dynamic_real: int = 0,
#     num_past_feat_dynamic_real: int = 0,
# ) -> Tuple[ListDataset, ListDataset]:
#
#     data_iter_train = []
#     data_iter_test = []
#
#     for k in range(num_ts):
#         ts_length = randint(min_length, max_length)
#         data_entry_train = {
#             FieldName.START: start,
#             FieldName.TARGET: [0.0] * ts_length,
#         }
#         if len(cardinality) > 0:
#             data_entry_train[FieldName.FEAT_STATIC_CAT] = [
#                 randint(0, c) for c in cardinality
#             ]
#         # Since used directly in predict and not in make_evaluate_predictions,
#         # where the test target would be chopped, test and train target have the same lengths
#         data_entry_test = data_entry_train.copy()
#         if num_feat_dynamic_real > 0:
#             data_entry_train[FieldName.FEAT_DYNAMIC_REAL] = [
#                 [float(1 + k)] * ts_length
#                 for k in range(num_feat_dynamic_real)
#             ]
#             data_entry_test[FieldName.FEAT_DYNAMIC_REAL] = [
#                 [float(1 + k)] * (ts_length + prediction_length)
#                 for k in range(num_feat_dynamic_real)
#             ]
#         data_iter_train.append(data_entry_train)
#         data_iter_test.append(data_entry_test)
#
#     return (
#         ListDataset(data_iter=data_iter_train, freq=freq),
#         ListDataset(data_iter=data_iter_test, freq=freq),
#     )
#
# hps = {
#         "seed": 42,
#         "freq": "D",
#         "context_length": 5,
#         "prediction_length": 3,
#         "quantiles": [0.5, 0.1],
#         "epochs": 3,
#         "num_batches_per_epoch": 3,
#     }
#
# train_data, test_data = make_dummy_datasets_with_features(
#     cardinality=[3, 10],
#     num_feat_dynamic_real=2,
#     num_past_feat_dynamic_real=4,
#     freq=hps["freq"],
#     prediction_length=hps["prediction_length"],
# )
print(train_data)
hyperparams = {'MQCNN': {'context_length': ag.Int(10, 20)} }
savedir = 'ag_models/'
predictor = task.fit(train_data=train_data, hyperparameters=hyperparams, output_directory=savedir, label="target", problem_type=FORECAST, id_columns=["index"])
# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:  predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, presets='best_quality', eval_metric=YOUR_METRIC_NAME)
results = predictor.fit_summary()

