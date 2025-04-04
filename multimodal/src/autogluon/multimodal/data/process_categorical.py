from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import nn

from ..constants import CATEGORICAL, COLUMN
from .collator import StackCollator, TupleCollator


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
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        requires_column_info
            Whether to require feature column information in dataloader.
        """
        self.prefix = model.prefix
        self.requires_column_info = requires_column_info

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
    ) -> Dict:
        """
        Process one sample's categorical features. Assume the categorical features
        are the encoded labels from sklearn' LabelEncoder().

        Parameters
        ----------
        categorical_features
            Categorical features of one sample.

        Returns
        -------
        A dictionary containing the processed categorical features.
        """
        ret = {}
        if self.requires_column_info:
            # TODO: consider moving this for loop into __init__() since each sample has the same information.
            for i, col_name in enumerate(categorical_features.keys()):
                ret[f"{self.categorical_column_prefix}_{col_name}"] = i

        ret[self.categorical_key] = list(categorical_features.values())

        return ret

    def __call__(
        self,
        categorical_features: Dict[str, int],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's categorical features and customize it for a specific model.

        Parameters
        ----------
        categorical_features
            Categorical features of one sample.
        feature_modalities
            The modality of the feature columns.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's processed categorical features.
        """
        return self.process_one_sample(categorical_features)
