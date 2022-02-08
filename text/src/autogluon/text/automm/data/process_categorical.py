from typing import Optional, List, Any
import numpy as np
from nptyping import NDArray
from ..constants import CATEGORICAL
from .collator import Stack, Tuple


class CategoricalProcessor:
    def __init__(
            self,
            prefix: str,
            num_categorical_columns: int,
    ):
        self.prefix = prefix
        self.num_categorical_columns = num_categorical_columns

    def collate_fn(self) -> dict:
        fn = [Stack() for _ in range(self.num_categorical_columns)]
        return {f"{self.prefix}_{CATEGORICAL}": Tuple(fn)}

    def process_one_sample(
            self,
            categorical_features: List[int],
    ) -> dict:
        return {
            f"{self.prefix}_{CATEGORICAL}": categorical_features
        }

    def __call__(
            self,
            all_categorical_features: List[NDArray[(Any,), np.int32]],
            idx: int,
            is_training: bool,
    ) -> dict:
        per_sample_features = [per_column_features[idx] for per_column_features in all_categorical_features]
        return self.process_one_sample(per_sample_features)
