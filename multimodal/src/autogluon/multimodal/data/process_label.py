import logging
from typing import Any, Dict, List, Optional, Union

from torch import nn

from ..constants import LABEL, MMDET_IMAGE
from .collator import ListCollator, StackCollator

logger = logging.getLogger(__name__)


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
        model
            The model for which this processor would be created.
        """
        logger.debug(f"initializing label processor for model {model.prefix}")
        self.prefix = model.prefix

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def collate_fn(self, label_column_names: Optional[List] = None, per_gpu_batch_size: Optional[int] = None) -> Dict:
        """
        Collate individual labels into a batch. Here it stacks labels.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for labels.
        """
        if self.prefix == MMDET_IMAGE:
            fn = {self.label_key: ListCollator()}
        else:
            fn = {self.label_key: StackCollator()}
        return fn

    def process_one_sample(
        self,
        labels: Dict[str, Union[int, float, list]],
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
        sub_dtypes: Dict[str, str],
        is_training: bool,
        load_only: bool = False,  # TODO: refactor mmdet_image and remove this
    ) -> Dict:
        """
        Extract one sample's labels and customize them for a specific model.

        Parameters
        ----------
        labels
            Labels of one sample.
        sub_dtypes
            The sub data types of all label columns.
        is_training
            Whether to do processing in the training mode. This unused flag is for the API compatibility.
        load_only
            Whether to only load the data. Other processing steps may happen in dataset.__getitem__.

        Returns
        -------
        A dictionary containing one sample's processed label.
        """
        return self.process_one_sample(labels)
