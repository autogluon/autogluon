from typing import Dict, List, Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..constants import PREDICT, TEST, TRAIN, VALIDATE
from .dataset import BaseDataset
from .preprocess_dataframe import MultiModalFeaturePreprocessor
from .utils import get_collate_fn


class BaseDataModule(LightningDataModule):
    """
    Set up Pytorch DataSet and DataLoader objects to prepare data for single-modal/multimodal training,
    validation, testing, and prediction. We organize the multimodal data using pd.DataFrame.
    For some modalities, e.g, image, that cost much memory, we only store their disk path to do lazy loading.
    This class inherits from the Pytorch Lightning's LightningDataModule:
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
        data_processors: Union[dict, List[dict]],
        per_gpu_batch_size: int,
        num_workers: int,
        train_data: Optional[pd.DataFrame] = None,
        validate_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        predict_data: Optional[pd.DataFrame] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        val_use_training_mode: bool = False,
    ):
        """
        Parameters
        ----------
        df_preprocessor
            One or a list of dataframe preprocessors. The preprocessing of one modality is generic so that
            the preprocessed data can be used by different models requiring the modality.
            For example, formatting input data as strings is a valid preprocessing operation for text.
            However, tokenizing strings into ids is invalid since different models generally
            use different tokenizers.
        data_processors
            The data processors to prepare customized data for each model. Each processor is only charge of
            one modality of one model. This helps scale up training arbitrary combinations of models.
        per_gpu_batch_size
            Mini-batch size for each GPU.
        num_workers
            Number of workers for Pytorch DataLoader.
        train_data
            Training data.
        validate_data
            Validation data.
        test_data
            Test data.
        predict_data
            Prediction data. No labels required in it.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response indexes instead of their contents.
        val_use_training_mode
             whether we are triggering is_training when creating the dataset for validation.
             This is used when we want to use val_loss as val metric, and thus we'll use data pipeline
             for training instead of for inference during validation.
        """
        super().__init__()
        self.prepare_data_per_node = True

        if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
            df_preprocessor = [df_preprocessor]
        if isinstance(data_processors, dict):
            data_processors = [data_processors]

        self.df_preprocessor = df_preprocessor
        self.data_processors = data_processors
        self.per_gpu_batch_size = per_gpu_batch_size
        self.num_workers = num_workers
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.predict_data = predict_data
        self.id_mappings = id_mappings
        self.val_use_training_mode = val_use_training_mode

    def set_dataset(self, split):
        data_split = getattr(self, f"{split}_data")
        if self.val_use_training_mode:
            is_training = split in [TRAIN, VALIDATE]
        else:
            is_training = split == TRAIN
        dataset = BaseDataset(
            data=data_split,
            preprocessor=self.df_preprocessor,
            processors=self.data_processors,
            id_mappings=self.id_mappings,
            is_training=is_training,
        )

        setattr(self, f"{split}_dataset", dataset)

    def setup(self, stage):
        """
        Set up datasets for different stages: "fit" (training and validation), "test", and "predict".
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup

        Parameters
        ----------
        stage
            Stage name including choices:
                - fit (For the fitting stage)
                - test (For the test stage)
                - predict (For the prediction stage)
        """
        if stage == "fit":
            self.set_dataset(TRAIN)
            self.set_dataset(VALIDATE)
        elif stage == "validate":
            self.set_dataset(VALIDATE)
        elif stage == "test":
            self.set_dataset(TEST)
        elif stage == "predict":
            self.set_dataset(PREDICT)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        """
        Create the dataloader for training.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#train-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=get_collate_fn(
                df_preprocessor=self.df_preprocessor,
                data_processors=self.data_processors,
                per_gpu_batch_size=self.per_gpu_batch_size,
            ),
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def val_dataloader(self):
        """
        Create the dataloader for validation.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#val-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        loader = DataLoader(
            self.validate_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=get_collate_fn(
                df_preprocessor=self.df_preprocessor,
                data_processors=self.data_processors,
                per_gpu_batch_size=self.per_gpu_batch_size,
            ),
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def test_dataloader(self):
        """
        Create the dataloader for test.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#test-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=get_collate_fn(
                df_preprocessor=self.df_preprocessor,
                data_processors=self.data_processors,
                per_gpu_batch_size=self.per_gpu_batch_size,
            ),
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def predict_dataloader(self):
        """
        Create the dataloader for prediction.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#predict-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=get_collate_fn(
                df_preprocessor=self.df_preprocessor,
                data_processors=self.data_processors,
                per_gpu_batch_size=self.per_gpu_batch_size,
            ),
            persistent_workers=self.num_workers > 0,
        )
        return loader
