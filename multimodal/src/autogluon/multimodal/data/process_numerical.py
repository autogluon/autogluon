import logging
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import nn

from ..constants import COLUMN, NUMERICAL
from .collator import StackCollator

logger = logging.getLogger(__name__)


class NumericalProcessor:
    """
    Prepare numerical data for the model specified by "prefix".
    For multiple models requiring numerical data, we need to create a NumericalProcessor
    for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        merge: Optional[str] = "concat",
        requires_column_info: bool = False,
        dropout: Optional[float] = 0,
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        merge
            How to merge numerical features from multiple columns in a multimodal pd.DataFrame.
            Currently, it only supports one choice:
            - concat
                Concatenate the numerical features.
        requires_column_info
            Whether to require feature column information in dataloader.
        """
        logger.debug(f"initializing numerical processor for model {model.prefix}")
        self.prefix = model.prefix
        self.merge = merge
        self.requires_column_info = requires_column_info
        self.numerical_fill_values = model.numerical_fill_values
        self.dropout = dropout
        assert 0 <= self.dropout <= 1
        if self.dropout > 0:
            logger.debug(f"numerical value dropout probability: {self.dropout}")
            logger.debug(f"dropped values will be replaced by {self.numerical_fill_values}")

    @property
    def numerical_key(self):
        return f"{self.prefix}_{NUMERICAL}"

    @property
    def numerical_column_prefix(self):
        return f"{self.numerical_key}_{COLUMN}"

    def collate_fn(self, numerical_column_names: List) -> Dict:
        """
        Collate individual samples into a batch. Here it stacks numerical features.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for numerical features.
        """
        fn = {}
        if self.requires_column_info:
            assert numerical_column_names, "Empty numerical column names."
            for col_name in numerical_column_names:
                fn[f"{self.numerical_column_prefix}_{col_name}"] = StackCollator()

        fn[self.numerical_key] = StackCollator()

        return fn

    def process_one_sample(
        self,
        numerical_features: Dict[str, float],
        is_training: bool,
    ) -> Dict:
        """
        Process one sample's numerical features.
        Here it converts numerical features to a NumPy array.

        Parameters
        ----------
        numerical_features
            Numerical features of one sample.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing the processed numerical features.
        """
        ret = {}
        if self.requires_column_info:
            # TODO: consider moving this for loop into __init__() since each sample has the same information.
            for i, col_name in enumerate(numerical_features.keys()):
                ret[f"{self.numerical_column_prefix}_{col_name}"] = i

        if is_training and self.dropout > 0:
            numerical_features_copy = dict()
            for k, v in numerical_features.items():
                if random.uniform(0, 1) <= self.dropout:
                    numerical_features_copy[k] = self.numerical_fill_values[k]
                else:
                    numerical_features_copy[k] = v
            numerical_features = numerical_features_copy

        if self.merge == "concat":
            ret[self.numerical_key] = np.array(list(numerical_features.values()), dtype=np.float32)
        else:
            raise ValueError(f"Unknown merging type: {self.merge}")

        return ret

    def __call__(
        self,
        numerical_features: Dict[str, float],
        sub_dtypes: Dict[str, str],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's numerical features and customize it for a specific model.

        Parameters
        ----------
        numerical_features
            Numerical features of one sample.
        sub_dtypes
            The sub data types of all numerical columns.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing one sample's processed numerical features.
        """
        return self.process_one_sample(
            numerical_features=numerical_features,
            is_training=is_training,
        )
