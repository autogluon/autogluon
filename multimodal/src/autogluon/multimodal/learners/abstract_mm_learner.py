import logging
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty, abstractstaticmethod
from typing import List, Optional, Union

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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        validation_metric_name: str,
        minmax_mode: str,  # this is determined solely on validation_metric_name
        save_path: str,
        ckpt_path: str,  # these two are synced
        resume: bool,
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
