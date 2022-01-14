import json
import logging
import os
import random
import time
import warnings
import numpy as np
import pandas as pd

from autogluon.core.constants import QUANTILE
from autogluon.core.utils import try_import_torch

from .tabular_nn_torch import TabularNeuralNetTorchModel

logger = logging.getLogger(__name__)


class TabularNeuralQuantileModel(TabularNeuralNetTorchModel):
    """
    PyTorch neural network models for (multiple) quantile regression with tabular data.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.problem_type != QUANTILE:
            raise ValueError("This neural network is only available for quantile regression")

