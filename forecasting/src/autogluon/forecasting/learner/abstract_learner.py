from ..models.gluonts_model.mqcnn.mqcnn_model import MQCNNModel
import random
from ..trainer.abstract_trainer import AbstractTrainer
import os
from autogluon.core.utils.savers import save_pkl, save_json
from autogluon.core.utils.loaders import load_pkl
from gluonts.evaluation import Evaluator


class AbstractLearner:

    learner_file_name = 'learner.pkl'
    learner_info_name = 'info.pkl'
    learner_info_json_name = 'info.json'

    def __init__(self, path_context: str, eval_metric=None, is_trainer_present=False, random_seed=0):
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)

        if eval_metric is not None and eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {eval_metric} is not available yet.")

        if eval_metric is not None:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = "mean_wQuantileLoss"

        self.is_trainer_present = is_trainer_present
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
        self.random_seed = random_seed

        self.trainer: AbstractTrainer = None
        self.trainer_type = None
        self.trainer_path = None
        self.reset_paths = False

    def set_contexts(self, path_context):
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        model_context = path_context + 'models' + os.path.sep
        save_path = path_context + self.learner_file_name
        return path_context, model_context, save_path

    def fit(self, train_data, freq, prediction_length, val_data=None, **kwargs):
        return self._fit(train_data=train_data,
                         freq=freq,
                         prediction_length=prediction_length,
                         val_data=val_data,
                         **kwargs)

    def _fit(self, train_data, freq, prediction_length, val_data=None, scheduler_options=None, hyperparameter_tune=False,
             hyperparameters=None):
        raise NotImplementedError

    def predict(self, data, model=None, for_score=False):
        predict_target = self.load_trainer().predict(data=data, model=model, for_score=for_score)
        return predict_target

    def evaluate(self, forecasts, tss, **kwargs):
        quantiles = kwargs.get("quantiles", None)
        # TODO: difference between evaluate() and score()?
        if self.eval_metric is not None and self.eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric { self.eval_metric} is not available yet.")

        # if quantiles are given, use the given on, otherwise use the default
        if quantiles is not None:
            evaluator = Evaluator(quantiles=quantiles)
        else:
            evaluator = Evaluator()

        num_series = len(tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        return agg_metrics[self.eval_metric]

    def score(self, data, model=None, quantiles=None):
        trainer = self.load_trainer()
        return trainer.score(data, model=model, quantiles=quantiles)

    def leaderboard(self):
        trainer = self.load_trainer()
        return trainer.leaderboard()

    def save(self):
        trainer = None
        if self.trainer is not None:
            if not self.is_trainer_present:
                self.trainer.save()
                trainer = self.trainer
                self.trainer = None
        save_pkl.save(path=self.save_path, object=self)
        self.trainer = trainer

    @classmethod
    def load(cls, path_context, reset_paths=True):
        load_path = path_context + cls.learner_file_name
        obj = load_pkl.load(path=load_path)
        if reset_paths:
            obj.set_contexts(path_context)
            obj.trainer_path = obj.model_context
            obj.reset_paths = reset_paths
            # TODO: Still have to change paths of models in trainer + trainer object path variables
            return obj
        else:
            obj.set_contexts(obj.path_context)
            return obj

    def load_trainer(self) -> AbstractTrainer:
        if self.trainer is not None:
            return self.trainer
        else:
            return self.trainer_type.load(path=self.trainer_path, reset_paths=self.reset_paths)

    def save_trainer(self, trainer):
        if self.is_trainer_present:
            self.trainer = trainer
            self.save()
        else:
            self.trainer_path = trainer.path
            trainer.save()

    @classmethod
    def load_info(cls, path, reset_paths=True, load_model_if_required=True):
        load_path = path + cls.learner_info_name
        try:
            return load_pkl.load(path=load_path)
        except Exception as e:
            if load_model_if_required:
                learner = cls.load(path_context=path, reset_paths=reset_paths)
                return learner.get_info()
            else:
                raise e

    def save_info(self, include_model_info):
        info = self.get_info(include_model_info)

        save_pkl.save(path=self.path + self.learner_info_name, object=info)
        save_json.save(path=self.path + self.learner_info_json_name, obj=info)
        return info

    def get_info(self, include_model_info):
        learner_info = {
            'path': self.path,
            'random_seed': self.random_seed,
        }

        return learner_info