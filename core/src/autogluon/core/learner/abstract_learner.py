import os
import random
import sys
from typing import Optional, Type

from autogluon.core.trainer import AbstractTrainer
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl


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

    def __init__(self, path_context: str, random_state: int = 0, **kwargs):
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)

        self.path_context_og: str = path_context  # Saves path_context used to create the original context of the learner to enable sub-fits.
        self.is_trainer_present: bool = False
        self.trainer: Optional[AbstractTrainer] = None
        self.trainer_type: Optional[Type] = None
        self.trainer_path: Optional[str] = None
        self.reset_paths: bool = False

        if random_state is None:
            random_state = random.randint(0, 1000000)
        self.random_state = random_state

        try:
            from ..version import __version__  # noqa

            self.version = __version__
        except ImportError:
            self.version = None
        self._python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def create_contexts(self, path_context: str):
        """Create and return paths to save model objects, the learner object.

        Parameters
        ----------
        path_context: str
            Top-level directory where models and trainer will be saved.
        """
        model_context = os.path.join(path_context, "models")
        save_path = os.path.join(path_context, self.learner_file_name)
        return path_context, model_context, save_path

    def set_contexts(self, path_context: str):
        """Update the path where model, learner, and trainer objects will be saved.
        Also see `create_contexts`."""
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)

    @property
    def is_fit(self):
        return self.trainer_path is not None or self.trainer is not None

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
        load_path = os.path.join(path_context, cls.learner_file_name)
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
            # trainer_path is used to determine if there's a trained trainer
            # model_context contains the new trainer_path with updated context
            return self.trainer_type.load(path=self.model_context, reset_paths=self.reset_paths)  # noqa

    # reset_paths=True if the learner files have changed location since fitting.
    # TODO: Potentially set reset_paths=False inside load function if it is the same path to
    # TODO: avoid re-computing paths on all models path_context -> path for v0.1
    @classmethod
    def load_info(cls, path, reset_paths=True, load_model_if_required=True):
        load_path = os.path.join(path, cls.learner_info_name)
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

        save_pkl.save(path=os.path.join(self.path, self.learner_info_name), object=info)
        save_json.save(path=os.path.join(self.path, self.learner_info_json_name), obj=info)
        return info

    def get_info(self, **kwargs):
        learner_info = {
            "path": self.path,
            "random_state": self.random_state,
            "version": self.version,
        }

        return learner_info
