from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
from gluonts.nursery.autogluon_tabular.estimator import TabularEstimator
import logging

logger = logging.getLogger(__name__)

"""
Autotabular model from Gluon-TS
"""
class AutoTabularModel(AbstractGluonTSModel):

    def __init__(self, path: str, freq: str, prediction_length: int, name: str = "AutoTabular",
                 eval_metric: str = None, hyperparameters=None, model=None, **kwargs):
        super().__init__(path=path,
                         freq=freq,
                         prediction_length=prediction_length,
                         hyperparameters=hyperparameters,
                         name=name,
                         eval_metric=eval_metric,
                         model=model,
                         **kwargs)

    def create_model(self):
        if len(AbstractGluonTSModel.prev_fitting_time) == 0:
            time_limit = 300
        else:
            time_limit = max(sum(AbstractGluonTSModel.prev_fitting_time) / len(AbstractGluonTSModel.prev_fitting_time), 300)
        if "time_limit" not in self.params:
            self.params["time_limit"] = time_limit
        self.model = TabularEstimator(freq=self.params["freq"],
                                      prediction_length=self.params["prediction_length"],
                                      time_limit=self.params["time_limit"],
                                      last_k_for_val=self.params["prediction_length"])

    def fit(self, train_data, val_data=None, time_limit=None):
        if time_limit is None or time_limit > 0:
            self.create_model()
            if val_data is not None:
                self.model = self.model.train(val_data)
            else:
                self.model = self.model.train(train_data)
        else:
            raise TimeLimitExceeded