from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from typing import Optional
from .dataset import BaseDataset
from .collator import Dict
from ..constants import TRAIN, VAL, TEST, PREDICT
from .preprocess_dataframe import MultiModalFeaturePreprocessor


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            df_preprocessor: MultiModalFeaturePreprocessor,
            data_processors: dict,
            per_gpu_batch_size: int,
            num_workers: int,
            train_data: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None,
            predict_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.df_preprocessor = df_preprocessor
        self.data_processors = data_processors
        self.per_gpu_batch_size = per_gpu_batch_size
        self.num_workers = num_workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data

    def set_dataset(self, split):
        data_split = getattr(self, f"{split}_data")
        dataset = BaseDataset(
            data=data_split,
            preprocessor=self.df_preprocessor,
            processors=self.data_processors,
            is_training=split == TRAIN,
        )

        setattr(self, f"{split}_dataset", dataset)

    def setup(self, stage):
        if stage == "fit":
            self.set_dataset(TRAIN)
            self.set_dataset(VAL)
        elif stage == "test":
            self.set_dataset(TEST)
        elif stage == "predict":
            self.set_dataset(PREDICT)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def get_collate_fn(self,):
        collate_fn = {}
        for d_type in self.data_processors:
            for per_model_processor in self.data_processors[d_type]:
                collate_fn.update(per_model_processor.collate_fn())
        return Dict(collate_fn)

