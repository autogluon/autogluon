import logging
import torch
import pandas as pd
from .preprocess_dataframe import MultiModalFeaturePreprocessor
from ..constants import (
    GET_ITEM_ERROR_RETRY, AUTOMM
)
logger = logging.getLogger(AUTOMM)


class BaseDataset(torch.utils.data.Dataset):
    """
    A Pytorch DataSet class to process a multimodal pd.DataFrame. It first uses a preprocessor to
    produce model-agnostic features. Then, each processor prepares customized data for one modality
    per model. For code simplicity, here we treat ground-truth label as one modality. This class is
    independent of specific data modalities and models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: MultiModalFeaturePreprocessor,
        processors: dict,
        is_training: bool = False,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing multimodal features.
        preprocessor
            A multimodal feature preprocessor generating model-agnostic features.
        processors
            Data processors customizing data for each modality per model.
        is_training
            Whether in training mode. Some data processing may be different between training
            and validation/testing/prediction, e.g., image data augmentation is used only in
            training.
        """
        super().__init__()
        self.processors = processors
        self.is_training = is_training
        self._consecutive_errors = 0

        self.lengths = []
        for per_modality, per_modality_processors in processors.items():
            per_modality_features = getattr(preprocessor, f"transform_{per_modality}")(data)
            setattr(self, f"{per_modality}", per_modality_features)
            self.lengths.append(len(per_modality_features[0]))
        assert len(set(self.lengths)) == 1

    def __len__(self):
        """
        Assume that all modalities have the same sample number.

        Returns
        -------
        Sample number in this dataset.
        """
        return self.lengths[0]

    def __getitem__(self, idx):
        """
        Iterate through all data processors to prepare model inputs. The data processors are
        organized first by modalities and then by models.

        Parameters
        ----------
        idx
            Index of sample to process.

        Returns
        -------
        Input data formatted as a dictionary.
        """
        ret = dict()
        try:
            for per_modality, per_modality_processors in self.processors.items():
                for per_model_processor in per_modality_processors:
                    ret.update(per_model_processor(getattr(self, per_modality), idx, self.is_training))
        except Exception as e:
            logger.debug(f"Skipping sample {idx} due to '{e}'")
            self._consecutive_errors += 1
            if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                return self.__getitem__((idx + 1) % self.__len__())
            else:
                raise e
        self._consecutive_errors = 0
        return ret
