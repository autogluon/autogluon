from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
from gluonts.nursery.autogluon_tabular.estimator import TabularEstimator
import logging

logger = logging.getLogger(__name__)


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
        self.model = TabularEstimator.from_hyperparameters(path=self.path, **self.params)
