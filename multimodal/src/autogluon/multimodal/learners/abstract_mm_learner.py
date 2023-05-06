from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Dict, List, Optional, Union

import pandas as pd

from autogluon.multimodal.problem_types import ProblemTypeProperty

logger = logging.getLogger(__name__)


class AbstractMMLearner(ABC):
    def __init__(
        self,
        **kwargs,
    ):
        pass

    @abstractproperty
    def path(self) -> Optional[str]:
        pass

    @abstractproperty
    def label(self) -> Optional[str]:
        pass

    @abstractproperty
    def problem_type(self) -> Optional[str]:
        pass

    @abstractproperty
    def problem_property(self) -> Optional[ProblemTypeProperty]:
        pass

    @abstractproperty
    def class_labels(self) -> Optional[List[str]]:
        pass

    @abstractproperty
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
        teacher_predictor: Union[str, AbstractMMLearner] = None,
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
    def save(self, **kwargs):
        pass

    @abstractclassmethod
    def load(cls, **kwargs):
        pass


if __name__ == "__main__":
    # obj = AbstractMMLearner()
    import ipdb

    ipdb.set_trace()
