from ..abstract.abstract_model import AbstractModel
from .default_parameters import get_default_parameters
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
import os
import time
from core import Int, Space
import json
from ..abstract.model_trial import model_trial
from tqdm import tqdm
from core.scheduler.fifo import FIFOScheduler


class MQCNNModel(AbstractModel):

    def __init__(self, hyperparameters=None, model=None):
        super().__init__(hyperparameters, model)
        self.set_default_parameters()
        self.name = "mqcnn"
        if hyperparameters is not None:
            self.params.update(hyperparameters)
        self.best_configs = self.params.copy()

    def set_default_parameters(self):
        self.params = get_default_parameters()

    def create_model(self):
        # print(self.params)
        self.model = MQCNNEstimator.from_hyperparameters(**self.params)

    def fit(self, train_ds):
        # print(self.params)
        self.create_model()
        self.model = self.model.train(train_ds)

    def predict(self, test_ds, num_samples=100):
        # self.model = self.model.predict(test_ds)
        from gluonts.evaluation.backtest import make_evaluation_predictions
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds,
                                                         predictor=self.model,
                                                         num_samples=num_samples)
        return list(tqdm(forecast_it, total=len(test_ds))), list(tqdm(ts_it, total=len(test_ds)))

    def score(self, y, metric=None):
        """
        metric: if metric is None, we will by default use mean_wQuantileLoss for scoring.
                should be one of "MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"
        """
        if metric is None:
            metric = "mean_wQuantileLoss"
        if metric is not None and metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {metric} is not available yet.")

        from gluonts.evaluation import Evaluator
        evaluator = Evaluator(quantiles=self.params["quantiles"])
        forecasts, tss = self.predict(y)
        num_series = len(tss)
        # print(num_series, forecasts, tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)

        # print(json.dumps(agg_metrics, indent=4))
        return agg_metrics[metric]

    def hyperparameter_tune(self, train_data, test_data, metric=None, **kwargs):
        params_copy = self.params.copy()
        util_args = dict(
            train_ds=train_data,
            test_ds=test_data,
            metric=metric,
            model=self,
        )

        model_trial.register_args(util_args=util_args, **params_copy)
        scheduler = FIFOScheduler(model_trial,
                                  searcher="random",
                                  resource={'num_cpus': 1, 'num_gpus': 0},
                                  num_trials=10,
                                  reward_attr='validation_performance',
                                  time_attr='epoch')
        scheduler.run()
        scheduler.join_jobs()
        self.best_configs.update(scheduler.get_best_config())
        return scheduler.get_best_config(), scheduler.get_best_reward()
