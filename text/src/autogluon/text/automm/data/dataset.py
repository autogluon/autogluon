import torch
import pandas as pd
from .preprocess_dataframe import MultiModalFeaturePreprocessor


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: MultiModalFeaturePreprocessor,
        processors: dict,
        is_training: bool = False,
    ):
        super().__init__()
        self.processors = processors
        self.is_training = is_training

        self.lengths = []
        for per_modality, per_modality_processors in processors.items():
            if len(per_modality_processors) > 0:
                per_modality_features = getattr(preprocessor, f"transform_{per_modality}")(data)
                setattr(self, f"{per_modality}", per_modality_features)
                self.lengths.append(len(per_modality_features[0]))
        assert len(set(self.lengths)) == 1

    def __len__(self):
        return self.lengths[0]

    def __getitem__(self, idx):
        ret = dict()
        for per_modality, per_modality_processors in self.processors.items():
            for per_model_processor in per_modality_processors:
                ret.update(per_model_processor(getattr(self, per_modality), idx, self.is_training))

        return ret

