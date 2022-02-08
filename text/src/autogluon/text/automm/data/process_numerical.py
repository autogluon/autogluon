from typing import Optional, List, Any
import numpy as np
from nptyping import NDArray
from ..constants import NUMERICAL
from .collator import Stack


class NumericalProcessor:
    def __init__(
            self,
            prefix: str,
            merge: Optional[str] = "concat",
    ):
        self.prefix = prefix
        self.merge = merge

    def collate_fn(self) -> dict:
        fn = {f"{self.prefix}_{NUMERICAL}": Stack()}
        return fn

    def process_one_sample(
            self,
            numerical_features: List[float],
    ) -> dict:
        if self.merge == "concat":
            return {
                f"{self.prefix}_{NUMERICAL}": np.array(numerical_features, dtype=np.float32)
            }

    def __call__(
            self,
            all_numerical_features: List[NDArray[(Any,), np.float32]],
            idx: int,
            is_training: bool,
    ) -> dict:
        per_sample_features = [per_column_features[idx] for per_column_features in all_numerical_features]
        return self.process_one_sample(per_sample_features)