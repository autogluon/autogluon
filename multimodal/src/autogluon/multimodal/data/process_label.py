from typing import Any, Dict, List, Optional, Union

from torch import nn

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
        model: nn.Module,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        """
        self.prefix = model.prefix

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def collate_fn(self, label_column_names: Optional[List] = None) -> Dict:
        """
        Collate individual labels into a batch. Here it stacks labels.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for labels.
        """
        fn = {self.label_key: Stack()}
        return fn

    def process_one_sample(
        self,
        labels: Dict[str, Union[int, float]],
    ) -> Dict:
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
            self.label_key: labels[next(iter(labels))],  # get the first key's value
        }

    def __call__(
        self,
        labels: Dict[str, Union[int, float]],
        is_training: bool,
    ) -> Dict:
        """
        Extract one sample's labels and customize them for a specific model.

        Parameters
        ----------
        labels
            Labels of one sample.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.

        Returns
        -------
        A dictionary containing one sample's processed label.
        """
        return self.process_one_sample(labels)
