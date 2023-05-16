from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Dict, List, Optional, Union

import pandas as pd

from autogluon.multimodal.problem_types import ProblemTypeProperty

logger = logging.getLogger(__name__)


class AbstractLearner(ABC):
    def __init__(
        self,
        **kwargs,
    ):
        pass

    @property
    @abstractmethod
    def path(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def label(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def problem_type(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def problem_property(self) -> Optional[ProblemTypeProperty]:
        pass

    @property
    @abstractmethod
    def class_labels(self) -> Optional[List[str]]:
        pass

    @property
    @abstractmethod
    def positive_class(self) -> Optional[str]:
        pass

    @abstractmethod
    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        max_num_tuning_data: Optional[int] = None,  # TODO: Remove since this is for detection
        # TODO: Remove since this is for matching
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_predictor: Union[str, AbstractLearner] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[dict] = None,
        clean_ckpts: Optional[bool] = True,
        **kwargs,
    ):
        pass

    @abstractmethod
    def predict(self, data: Union[pd.DataFrame, dict, list, str], **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data: Union[pd.DataFrame, dict, list, str], **kwargs):
        pass

    @abstractmethod
    def extract_embedding(self, **kwargs):
        pass

    @abstractmethod
    def fit_summary(self, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str, standalone: bool, **kwargs):
        pass

    @abstractclassmethod
    def load(cls, path: str, resume: Optional[bool] = False, verbosity: Optional[int] = 3, **kwargs):
        pass


if __name__ == "__main__":
    # obj = AbstractMMLearner()
    import ipdb

    ipdb.set_trace()
