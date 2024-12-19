import logging
import random
from typing import Any, Dict, List, Optional, Union

from torch import nn

from ..constants import CATEGORICAL, COLUMN
from .collator import StackCollator, TupleCollator

logger = logging.getLogger(__name__)


class CategoricalProcessor:
    """
    Prepare categorical data for the model specified by "prefix".
    For multiple models requiring categorical data, we need to create a CategoricalProcessor
    for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        requires_column_info: bool = False,
        dropout: Optional[float] = 0,
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        requires_column_info
            Whether to require feature column information in dataloader.
        """
        logger.debug(f"initializing categorical processor for model {model.prefix}")
        self.prefix = model.prefix
        self.requires_column_info = requires_column_info
        self.num_categories = model.num_categories
        self.dropout = dropout
        assert 0 <= self.dropout <= 1
        if self.dropout > 0:
            logger.debug(f"categorical value dropout probability: {self.dropout}")
            fill_values = {k: v - 1 for k, v in self.num_categories.items()}
            logger.debug(f"dropped values will be replaced by {fill_values}")

    @property
    def categorical_key(self):
        return f"{self.prefix}_{CATEGORICAL}"

    @property
    def categorical_column_prefix(self):
        return f"{self.categorical_key}_{COLUMN}"

    def collate_fn(self, categorical_column_names: Optional[List] = None) -> Dict:
        """
        Collate individual samples into a batch. It stacks categorical features of
        each column independently. This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for categorical features.
        """
        fn = {}
        if self.requires_column_info:
            assert categorical_column_names, "Empty categorical column names."
            for col_name in categorical_column_names:
                fn[f"{self.categorical_column_prefix}_{col_name}"] = StackCollator()

        fn[self.categorical_key] = TupleCollator([StackCollator() for _ in range(len(categorical_column_names))])

        return fn

    def process_one_sample(
        self,
        categorical_features: Dict[str, int],
        is_training: bool,
    ) -> Dict:
        """
        Process one sample's categorical features. Assume the categorical features
        are the encoded labels from sklearn' LabelEncoder().

        Parameters
        ----------
        categorical_features
            Categorical features of one sample.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing the processed categorical features.
        """
        ret = {}
        if self.requires_column_info:
            # TODO: consider moving this for loop into __init__() since each sample has the same information.
            for i, col_name in enumerate(categorical_features.keys()):
                ret[f"{self.categorical_column_prefix}_{col_name}"] = i

        if is_training and self.dropout > 0:
            categorical_features_copy = dict()
            for k, v in categorical_features.items():
                if random.uniform(0, 1) <= self.dropout:
                    categorical_features_copy[k] = self.num_categories[k] - 1
                else:
                    categorical_features_copy[k] = v
            categorical_features = categorical_features_copy

        # make sure keys are in the same order
        assert list(categorical_features.keys()) == list(self.num_categories.keys())
        ret[self.categorical_key] = list(categorical_features.values())

        return ret

    def __call__(
        self,
        categorical_features: Dict[str, int],
        sub_dtypes: Dict[str, str],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's categorical features and customize it for a specific model.

        Parameters
        ----------
        categorical_features
            Categorical features of one sample.
        sub_dtypes
            The sub data types of all categorical columns.
        is_training
            Whether to do processing in the training mode.

        Returns
        -------
        A dictionary containing one sample's processed categorical features.
        """
        return self.process_one_sample(
            categorical_features=categorical_features,
            is_training=is_training,
        )
