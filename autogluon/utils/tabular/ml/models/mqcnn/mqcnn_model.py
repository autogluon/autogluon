import logging
import re

import numpy as np
import pandas as pd
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from ....ml.models.abstract.abstract_model import AbstractModel
from .hyperparameters.parameters import get_default_parameters
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from tqdm.autonotebook import tqdm


class MQCNNModel(AbstractModel):

    def __init__(self, index_column="index", date_column="date", target_column="target", **kwargs):
        super().__init__(**kwargs)
        self.index_column = index_column
        self.date_column = date_column
        self.target_column = target_column
        self.params = get_default_parameters()

    def fit(self, **kwargs):
        kwargs = self._preprocess_fit_args(**kwargs)
        self._fit(**kwargs)

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        X_train = self.preprocess(X_train)
        estimator = MQCNNEstimator.from_hyperparameters(**self.params)
        print(estimator)
        self.model = estimator.train(X_train)

    def preprocess(self, X):
        date_list = sorted(list(set(X[self.date_column])))

        def transform_dataframe(df):
            data_dic = {self.index_column: list(set(df[self.index_column]))}

            for date in date_list:
                tmp = df[df[self.date_column] == date][[self.index_column, self.date_column, self.target_column]]
                tmp = tmp.pivot(index=self.index_column, columns=self.date_column, values=self.target_column)
                tmp_values = tmp[date].values
                data_dic[date] = tmp_values
            return pd.DataFrame(data_dic)

        X = transform_dataframe(X)

        target_values = X.drop(self.index_column, axis=1).values
        processed_X = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: pd.Timestamp("2020-01-22", freq="1D"),
            }
            for (target, ) in zip(target_values,)
        ], freq="1D")
        return processed_X

    def predict(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X=X)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=X,
            predictor=self.model,
            num_samples=100
        )
        return [forecast.forecast_array for forecast in list(tqdm(forecast_it, total=len(X)))]

    def score(self, X, y=None, quantiles=[0.9], eval_metric=None, metric_needs_y_pred=None, preprocess=True, index_column="index", date_column="date", target_column="target"):
        from gluonts.evaluation import Evaluator
        # evaluator = Evaluator(quantiles=quantiles)
        processed_X = self.preprocess(X=X)
        tss = processed_X.list_data
        forecasts = self.predict(processed_X, preprocess=False)
        prediction_length = self.model.prediction_length
        diff = 0
        for idx in range(len(forecasts)):
            forecast = np.mean(forecasts[idx], axis=0)
            true_values = np.array(tss[idx]["target"][-prediction_length:])
            diff += (np.sum((forecast - true_values) ** 2) / prediction_length) ** 0.5
        return diff / len(forecasts)
        # agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        # print(json.dumps(agg_metrics, indent=4))
        # return agg_metrics["mean_wQuantileLoss"]