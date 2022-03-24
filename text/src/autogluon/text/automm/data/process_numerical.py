from typing import Optional, List, Any
import numpy as np
from nptyping import NDArray
from ..constants import NUMERICAL
from .collator import Stack


class NumericalProcessor:
    """
    Prepare numerical data for the model specified by "prefix".
    For multiple models requiring numerical data, we need to create a NumericalProcessor
    for each related model so that they will have independent input.
    """

    def __init__(
            self,
            prefix: str,
            merge: Optional[str] = "concat",
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        merge
            How to merge numerical features from multiple columns in a multimodal pd.DataFrame.
            Currently, it only supports one choice:
            - concat
                Concatenate the numerical features.
        """
        self.prefix = prefix
        self.merge = merge

    def collate_fn(self) -> dict:
        """
        Collate individual samples into a batch. Here it stacks numerical features.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for numerical features.
        """
        fn = {f"{self.prefix}_{NUMERICAL}": Stack()}
        return fn

    def process_one_sample(
            self,
            numerical_features: List[float],
    ) -> dict:
        """
        Process one sample's numerical features.
        Here it converts numerical features to a NumPy array.

        Parameters
        ----------
        numerical_features
            Numerical features of one sample.

        Returns
        -------
        A dictionary containing the processed numerical features.
        """
        if self.merge == "concat":
            return {
                f"{self.prefix}_{NUMERICAL}": np.array(numerical_features, dtype=np.float32)
            }
        else:
            raise ValueError(f"Unknown merging type: {self.merge}")

    def __call__(
            self,
            all_numerical_features: List[NDArray[(Any,), np.float32]],
            idx: int,
            is_training: bool,
    ) -> dict:
        """
        Extract one sample's numerical features and customize it for a specific model.

        Parameters
        ----------
        all_numerical_features
            All the numerical features in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's processed numerical features.
        """
        per_sample_features = [per_column_features[idx] for per_column_features in all_numerical_features]
        return self.process_one_sample(per_sample_features)
