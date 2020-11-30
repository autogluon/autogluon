from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
import autogluon.core.utils.savers.save_pkl as save_pkl
import autogluon.core.utils.loaders.load_pkl as load_pkl
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.predictor import Predictor
import os
import time
from autogluon.core import Int, Space
import json
from gluonts.evaluation import Evaluator
from ..abstract_gluonts.model_trial import model_trial
from tqdm import tqdm
from autogluon.core.scheduler.fifo import FIFOScheduler
from pathlib import Path
from gluonts.model.predictor import Predictor
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.task.base.base_predictor import BasePredictor
import logging

logger = logging.getLogger(__name__)


class MQCNNModel(AbstractGluonTSModel):

    gluonts_model_path = "gluon_ts"

    def __init__(self, path: str, freq: str, prediction_length: int, name: str = "MQCNN",
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
        # use gluonts default parameters,
        # you can find them in MQCNNEstimator in gluonts.model.seq2seq
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

    def hyperparameter_tune(self, train_data, val_data, scheduler_options, **kwargs):
        time_start = time.time()
        params_copy = self.params.copy()

        directory = self.path

        dataset_train_filename = 'dataset_train.p'
        train_path = directory + dataset_train_filename
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = 'dataset_val.p'
        val_path = directory + dataset_val_filename
        save_pkl.save(path=val_path, object=val_data)
        scheduler_func, scheduler_options = scheduler_options

        util_args = dict(
            train_data_path=dataset_train_filename,
            val_data_path=dataset_val_filename,
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
        return self._get_hpo_results(scheduler, scheduler_options, time_start)

    def _get_hpo_results(self, scheduler, scheduler_options, time_start):
        # Store results / models from this HPO run:
        best_hp = scheduler.get_best_config()  # best_hp only contains searchable stuff
        hpo_results = {
            'best_reward': scheduler.get_best_reward(),
            'best_config': best_hp,
            'total_time': time.time() - time_start,
            'metadata': scheduler.metadata,
            'training_history': scheduler.training_history,
            'config_history': scheduler.config_history,
            'reward_attr': scheduler._reward_attr,
            'args': model_trial.args
        }

        hpo_results = BasePredictor._format_results(hpo_results)  # results summarizing HPO for this model
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            raise NotImplementedError("need to fetch model files from remote Workers")
            # TODO: need to handle locations carefully: fetch these files and put them into self.path directory:
            # 1) hpo_results['trial_info'][trial]['metadata']['trial_model_file']

        hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['trial_info'].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). How to ID these?
            file_id = "trial_" + str(trial)  # unique identifier to files from this trial
            trial_model_name = self.name + os.path.sep + file_id
            trial_model_path = self.path_root + trial_model_name + os.path.sep
            hpo_models[trial_model_name] = trial_model_path
            hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]

        logger.log(15, "Time for %s model HPO: %s" % (self.name, str(hpo_results['total_time'])))
        logger.log(15, "Best hyperparameter configuration for %s model: " % self.name)
        logger.log(15, str(best_hp))
        return hpo_models, hpo_model_performances, hpo_results
