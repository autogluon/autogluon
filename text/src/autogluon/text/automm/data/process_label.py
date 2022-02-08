from typing import Optional, List, Any, Union
from nptyping import NDArray
from ..constants import LABEL
from .collator import Stack


class LabelProcessor:
    def __init__(
            self,
            prefix: str,
    ):
        self.prefix = prefix

    def collate_fn(self) -> dict:
        fn = {f"{self.prefix}_{LABEL}": Stack()}
        return fn

    def process_one_sample(
            self,
            labels: List[Union[int, float]],
    ) -> dict:
        # Only use the first label
        return {
            f"{self.prefix}_{LABEL}": labels[0]
        }

    def __call__(
            self,
            all_labels: List[NDArray[(Any,), Any]],
            idx: int,
            is_training: bool,
    ) -> dict:
        per_sample_labels = [per_column_labels[idx] for per_column_labels in all_labels]
        return self.process_one_sample(per_sample_labels)
