""" Example script for defining and using custom models in AutoGluon Tabular """
import time

from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
from autogluon.utils.tabular.data.label_cleaner import LabelCleaner
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.utils import infer_problem_type
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from autogluon.utils.tabular.ml.constants import FORECAST
from autogluon.utils.tabular.ml.models.mqcnn.hyperparameters.parameters import get_default_parameters
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
import json
#########################
# Create a custom model #
#########################


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


class MQCNNModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = get_default_parameters()

    def fit(self, index_column="index", date_column="date", target_column="target", **kwargs):
        kwargs = self._preprocess_fit_args(**kwargs)
        self._fit(index_column=index_column, date_column=date_column, target_column=target_column, **kwargs)

    def _fit(self, X_train, y_train, index_column="index", date_column="date", target_column="target", X_val=None, y_val=None, time_limit=None, **kwargs):
        X_train = self.preprocess(X_train, index_column=index_column, date_column=date_column, target_column=target_column)
        estimator = MQCNNEstimator.from_hyperparameters(**self.params)
        print(estimator)
        self.model = estimator.train(X_train)

    def preprocess(self, X, index_column="index", date_column="date", target_column="target"):
        date_list = sorted(list(set(X[date_column])))

        def transform_dataframe(df):
            data_dic = {index_column: list(set(df[index_column]))}

            for date in date_list:
                tmp = df[df[date_column] == date][[index_column, date_column, target_column]]
                tmp = tmp.pivot(index=index_column, columns=date_column, values=target_column)
                tmp_values = tmp[date].values
                data_dic[date] = tmp_values
            return pd.DataFrame(data_dic)

        X = transform_dataframe(X)

        target_values = X.drop(index_column, axis=1).values
        processed_X = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: pd.Timestamp("2020-01-01", freq="1D"),
            }
            for (target, ) in zip(target_values,)
        ], freq="1D")
        return processed_X

    def predict(self, X, index_column="index", date_column="date", target_column="target", preprocess=True):
        if preprocess:
            X = self.preprocess(X=X, index_column=index_column, date_column=date_column, target_column=target_column)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=X,
            predictor=self.model,
            num_samples=100
        )
        return list(tqdm(forecast_it, total=len(X))), list(tqdm(ts_it, total=len(X))), len(X)

    def score(self, X, y=None, quantiles=[0.5, 0.9], eval_metric=None, metric_needs_y_pred=None, preprocess=True, index_column="index", date_column="date", target_column="target"):
        from gluonts.evaluation import Evaluator
        evaluator = Evaluator(quantiles=quantiles)
        forecasts, tss, num_series = self.predict(X, index_column=index_column, date_column=date_column, target_column=target_column)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        print(json.dumps(agg_metrics, indent=4))
        return agg_metrics["mean_wQuantileLoss"]



# dummy data
# train_data = make_dummy_datasets(ts_length=200)
# test_data = make_dummy_datasets(ts_length=10)
train_data = pd.read_csv("./COV19/processed_train.csv")
test_data = pd.read_csv("./COV19/processed_test.csv")

mqcnn_model = MQCNNModel(path='AutogluonModels/', name='CustomMQCNN', problem_type=FORECAST)
mqcnn_model.fit(X_train=train_data, y_train=None, index_column="name", date_column="Date", target_column="ConfirmedCases")

test_pred = mqcnn_model.predict(test_data, index_column="name", date_column="Date", target_column="ConfirmedCases")

score = mqcnn_model.score(test_data, index_column="name", date_column="Date", target_column="ConfirmedCases")

