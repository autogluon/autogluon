import os
from typing import Type, Optional

from autogluon.core.trainer import AbstractTrainer
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl, save_json


class AbstractLearner:
    """Abstract class for autogluon Learners. Learners encompass the learning problem end to end,
    including loading initial data, feature generation, model training, model prediction. Similar
    to Trainers, they implement `fit` and `predict` methods. The abstract method also includes
    concrete implementations of serialization/deserialization methods. Refer to individual
    documentations of concrete learner classes for further information.
    """

    learner_info_json_name = "info.json"
    learner_info_name = "info.pkl"
    learner_file_name = "learner.pkl"

    def __init__(self):
        self.save_path = None
        self.model_context = None
        self.path = None
        self.version = None
        self.trainer_path: Optional[str] = None
        self.is_trainer_present: bool = False
        self.trainer: Optional[AbstractTrainer] = None
        self.trainer_type: Optional[Type] = None
        self.reset_paths: bool = False
        self.random_state: int = 0

    def create_contexts(self, path_context):
        model_context = path_context + "models" + os.path.sep
        save_path = path_context + self.learner_file_name
        return path_context, model_context, save_path

    def set_contexts(self, path_context):
        self.path, self.model_context, self.save_path = self.create_contexts(
            path_context
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def leaderboard(self, *args, **kwargs):
        raise NotImplementedError

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
            if obj.trainer_path is not None:
                obj.trainer_path = obj.model_context
            obj.reset_paths = reset_paths
            # TODO: Still have to change paths of models in trainer + trainer object path variables
            return obj
        else:
            obj.set_contexts(obj.path_context)
            return obj

    def save_trainer(self, trainer):
        if self.is_trainer_present:
            self.trainer = trainer
            self.save()
        else:
            self.trainer_path = trainer.path
            trainer.save()

    def load_trainer(self) -> AbstractTrainer:
        if self.trainer is not None:
            return self.trainer
        else:
            if self.trainer_path is None:
                raise AssertionError("Trainer does not exist.")
            return self.trainer_type.load(  # noqa
                path=self.trainer_path, reset_paths=self.reset_paths
            )

    # reset_paths=True if the learner files have changed location since fitting.
    # TODO: Potentially set reset_paths=False inside load function if it is the same path to
    # TODO: avoid re-computing paths on all models path_context -> path for v0.1
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

    def save_info(self, include_model_info=False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.learner_info_name, object=info)
        save_json.save(path=self.path + self.learner_info_json_name, obj=info)
        return info

    def get_info(self, **kwargs):
        learner_info = {
            "path": self.path,
            "random_state": self.random_state,
            "version": self.version,
        }

        return learner_info
