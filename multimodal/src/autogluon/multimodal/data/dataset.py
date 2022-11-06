import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import torch

from ..constants import AUTOMM, GET_ITEM_ERROR_RETRY
from .preprocess_dataframe import MultiModalFeaturePreprocessor
from .utils import apply_data_processor, apply_df_preprocessor, get_per_sample_features

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
        preprocessor: List[MultiModalFeaturePreprocessor],
        processors: List[dict],
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        is_training: bool = False,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing multimodal features.
        preprocessor
            A list of multimodal feature preprocessors generating model-agnostic features.
        processors
            Data processors customizing data for each modality per model.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response indexes instead of their contents.
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
        for i, (per_preprocessor, per_processors_group) in enumerate(zip(preprocessor, processors)):
            modality_features, modality_types, length = apply_df_preprocessor(
                data=data,
                df_preprocessor=per_preprocessor,
                modalities=per_processors_group.keys(),
            )
            self.lengths.append(length)
            setattr(self, f"modality_features_{i}", modality_features)
            setattr(self, f"modality_types_{i}", modality_types)

        assert len(set(self.lengths)) == 1

        self.id_mappings = id_mappings

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
            for group_id, per_processors_group in enumerate(self.processors):
                per_sample_features = get_per_sample_features(
                    modality_features=getattr(self, f"modality_features_{group_id}"),
                    modality_types=getattr(self, f"modality_types_{group_id}"),
                    idx=idx,
                    id_mappings=self.id_mappings,
                )
                per_ret = apply_data_processor(
                    per_sample_features=per_sample_features,
                    data_processors=per_processors_group,
                    feature_modalities=getattr(self, f"modality_types_{group_id}"),
                    is_training=self.is_training,
                )
                ret.update(per_ret)
        except Exception as e:
            logger.debug(f"Skipping sample {idx} due to '{e}'")
            self._consecutive_errors += 1
            if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                return self.__getitem__((idx + 1) % self.__len__())
            else:
                raise e
        self._consecutive_errors = 0
        return ret
