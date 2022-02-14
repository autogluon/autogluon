from typing import Optional, List, Any, Union
from nptyping import NDArray
from ..constants import LABEL
from .collator import Stack


class LabelProcessor:
    """
    Prepare ground-truth labels for the model specified by "prefix".
    For multiple models, we need to create a LabelProcessor for each model so that
    each model will have independent labels.
    """

    def __init__(
            self,
            prefix: str,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        """
        self.prefix = prefix

    def collate_fn(self) -> dict:
        """
        Collate individual labels into a batch. Here it stacks labels.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for labels.
        """
        fn = {f"{self.prefix}_{LABEL}": Stack()}
        return fn

    def process_one_sample(
            self,
            labels: List[Union[int, float]],
    ) -> dict:
        """
        Process one sample's labels. Here it only picks the first label.
        New rules can be added if necessary.

        Parameters
        ----------
        labels
            One sample may have multiple labels.
        Returns
        -------
        A dictionary containing one sample's label.
        """
        return {
            f"{self.prefix}_{LABEL}": labels[0]
        }

    def __call__(
            self,
            all_labels: List[NDArray[(Any,), Any]],
            idx: int,
            is_training: bool,
    ) -> dict:
        """
        Extract one sample's labels and customize them for a specific model.

        Parameters
        ----------
        all_labels
            All labels in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's processed label.
        """
        per_sample_labels = [per_column_labels[idx] for per_column_labels in all_labels]
        return self.process_one_sample(per_sample_labels)
