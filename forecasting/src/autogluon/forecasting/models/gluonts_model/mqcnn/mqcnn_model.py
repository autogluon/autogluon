from ..gluonts_abstract.gluonts_abstract_model import GluonTSAbstractModel
from .default_parameters import get_default_parameters
import core.utils.savers.save_pkl as save_pkl
import core.utils.loaders.load_pkl as load_pkl
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.predictor import Predictor
import os
import time
from core import Int, Space
import json
from gluonts.evaluation import Evaluator
from ..gluonts_abstract.model_trial import model_trial
from tqdm import tqdm
from core.scheduler.fifo import FIFOScheduler
from pathlib import Path
from gluonts.model.predictor import Predictor
from core.utils.exceptions import TimeLimitExceeded


class MQCNNModel(GluonTSAbstractModel):

    gluonts_model_path = "gluon_ts"

    def __init__(self, path: str, freq: str, prediction_length: int, name: str="mqcnn",
                 eval_metric: str = None, hyperparameters=None, model=None):
        super().__init__(path=path,
                         freq=freq,
                         prediction_length=prediction_length,
                         hyperparameters=hyperparameters,
                         name=name,
                         eval_metric=eval_metric,
                         model=model)
        self.best_configs = self.params.copy()

    def set_default_parameters(self):
        # use gluonts default parameters
        self.params = {}

    def create_model(self):
        # print(self.params)
        self.model = MQCNNEstimator.from_hyperparameters(**self.params)

    def save(self, path: str = None,):
        if path is None:
            path = self.path
        weight_path = path + self.gluonts_model_path
        # save gluonts model
        os.makedirs(weight_path, exist_ok=True)
        self.model.serialize(Path(weight_path))
        self.model = None
        # save MQCNN object
        super(MQCNNModel, self).save()

    @classmethod
    def load(cls, path: str, reset_path=True):
        model = super(MQCNNModel, cls).load(path, reset_path)
        weight_path = path + cls.gluonts_model_path
        model.model = Predictor.deserialize(Path(weight_path))
        return model

    def fit(self, train_data, time_limit=None):
        if time_limit is None or time_limit > 0:
            self.create_model()
            self.model = self.model.train(train_data)
        else:
            raise TimeLimitExceeded

    def hyperparameter_tune(self, train_data, test_data, scheduler_options, **kwargs):
        time_start = time.time()
        params_copy = self.params.copy()

        directory = self.path

        dataset_train_filename = 'dataset_train.p'
        train_path = directory + dataset_train_filename
        save_pkl.save(path=train_path, object=train_data)

        dataset_test_filename = 'dataset_test.p'
        test_path = directory + dataset_test_filename
        save_pkl.save(path=test_path, object=test_data)
        scheduler_func, scheduler_options = scheduler_options

        util_args = dict(
            train_data_path=dataset_train_filename,
            test_data_path=dataset_test_filename,
            directory=directory,
            model=self,
            time_start=time_start,
            time_limit=scheduler_options["time_out"]
        )

        model_trial.register_args(util_args=util_args, **params_copy)
        scheduler: FIFOScheduler = scheduler_func(model_trial, **scheduler_options)
        scheduler.run()
        scheduler.join_jobs()
        self.best_configs.update(scheduler.get_best_config())
        return scheduler.get_best_config(), scheduler.get_best_reward()
