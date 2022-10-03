from typing import Any, Dict, List, Optional, Union

from nptyping import NDArray
from torch import nn

from ..constants import LABEL, NER, NER_ANNOTATION, TEXT
from .collator import Stack
from .utils import process_ner_annotations


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
        self.tokenizer = None
        self.model = model
        if self.prefix == NER:
            self.tokenizer = model.tokenizer

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
        if self.prefix == NER:
            ner_annotation = labels[NER_ANNOTATION]
            ner_text = labels[TEXT]
            # online label generation
            return {
                self.label_key: process_ner_annotations(ner_annotation, ner_text, self.tokenizer),
            }
        else:
            return {
                self.label_key: labels[next(iter(labels))],  # get the first key's value
            }

    def __call__(
        self,
        all_labels: Dict[str, Union[NDArray[(Any,), Any], list]],
        idx: int,
        is_training: bool,
    ) -> Dict:
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
        per_sample_labels = {
            per_column_name: per_column_labels[idx] for per_column_name, per_column_labels in all_labels.items()
        }
        return self.process_one_sample(per_sample_labels)
