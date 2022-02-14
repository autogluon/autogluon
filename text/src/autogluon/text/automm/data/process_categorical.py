from typing import Optional, List, Any
import numpy as np
from nptyping import NDArray
from ..constants import CATEGORICAL
from .collator import Stack, Tuple


class CategoricalProcessor:
    """
    Prepare categorical data for the model specified by "prefix".
    For multiple models requiring categorical data, we need to create a CategoricalProcessor
    for each related model so that they will have independent input.
    """

    def __init__(
            self,
            prefix: str,
            num_categorical_columns: int,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        num_categorical_columns
            Number of categorical columns in a multimodal pd.DataFrame.
        """
        self.prefix = prefix
        self.num_categorical_columns = num_categorical_columns

    def collate_fn(self) -> dict:
        """
        Collate individual samples into a batch. It stacks categorical features of
        each column independently. This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for categorical features.
        """
        fn = [Stack() for _ in range(self.num_categorical_columns)]
        return {f"{self.prefix}_{CATEGORICAL}": Tuple(fn)}

    def process_one_sample(
            self,
            categorical_features: List[int],
    ) -> dict:
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
        return {
            f"{self.prefix}_{CATEGORICAL}": categorical_features
        }

    def __call__(
            self,
            all_categorical_features: List[NDArray[(Any,), np.int32]],
            idx: int,
            is_training: bool,
    ) -> dict:
        """
        Extract one sample's categorical features and customize it for a specific model.

        Parameters
        ----------
        all_categorical_features
            All the categorical features in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's processed categorical features.
        """
        per_sample_features = [per_column_features[idx] for per_column_features in all_categorical_features]
        return self.process_one_sample(per_sample_features)
