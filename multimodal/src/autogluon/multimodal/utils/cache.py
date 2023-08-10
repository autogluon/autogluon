import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from ..constants import WEIGHT

logger = logging.getLogger(__name__)


class DDPPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Optional[str], write_interval: Optional[str], sleep_time=5):
        """
        Parameters
        ----------
        output_dir
            The directory to save predictions.
        write_interval
            When to write. Could be "batch" or "epoch".
            See Lightning's source code at
            https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/prediction_writer.html#BasePredictionWriter
        sleep_time
            If other process does not finish writing, sleep for a few seconds and recheck.
        """
        super().__init__(write_interval)
        assert isinstance(
            output_dir, (str, Path)
        ), f"Only str and pathlib.Path types are supported for path, but got {output_dir} of type {type(output_dir)}."
        self.sleep_time = sleep_time
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.output_dir = os.path.join(output_dir, f"predictions_{uuid.uuid4().hex}")
        try:
            os.makedirs(self.output_dir, exist_ok=False)
        except FileExistsError:  # assume the string is unique
            raise Exception(
                f'Output path {self.output_dir} already exists! Please clean all the outdated folders in {output_dir}."'
            )

    def get_predictions_cache_dir(self, global_rank: int):
        """
        Parameters
        ----------
        global_rank
            Global rank of current process.
        """
        return os.path.join(self.output_dir, f"predictions_rank_{global_rank}.pt")

    def get_batch_indices_cache_dir(self, global_rank: int):
        """
        Parameters
        ----------
        global_rank
            Global rank of current process.
        """
        return os.path.join(self.output_dir, f"sample_indices_rank_{global_rank}.pt")

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ):
        """
        Parameters
        ----------
        trainer
            Pytorch Lightning trainer.
        pl_module
            Pytorch Lightning module.
        predictions
            The predicted results.
        batch_indices
            The corresponding batch indices for prediction results.
        """
        # this will create N (num processes) files in `cache_dir` each containing
        # the predictions of its respective rank
        torch.save(predictions, self.get_predictions_cache_dir(trainer.global_rank))
        # here we save `batch_indices` to get the information about the data index
        # from prediction data
        torch.save(batch_indices, self.get_batch_indices_cache_dir(trainer.global_rank))

    def read_single_gpu_results(self, global_rank: Optional[int]):
        """
        Parameters
        ----------
        global_rank
            Global rank of current process.
        """
        sample_indices_file = self.get_batch_indices_cache_dir(global_rank)
        predictions_file = self.get_predictions_cache_dir(global_rank)
        while (not os.path.exists(sample_indices_file)) or (not os.path.exists(predictions_file)):
            logger.info(f"waiting for rank #{global_rank} to finish saving predictions...")
            time.sleep(self.sleep_time)
        sample_indices = torch.load(sample_indices_file)
        predictions = torch.load(predictions_file)
        os.remove(sample_indices_file)
        os.remove(predictions_file)
        return sample_indices, predictions

    def flatten(self, x):
        """
        Parameters
        ----------
        x
            A nested list to be flattened.
        """
        if isinstance(x, list):
            return [a for i in x for a in self.flatten(i)]
        else:
            return [x]

    def collect_all_gpu_results(self, num_gpus):
        """
        Parameters
        ----------
        num_gpus
            Number of gpus used.
        """
        sample_indices = []
        predictions = []
        for global_rank in range(num_gpus):
            sample_indices_per_rank, predictions_per_rank = self.read_single_gpu_results(global_rank)
            sample_indices += self.flatten(sample_indices_per_rank)
            predictions += self.flatten(predictions_per_rank)

        assert sorted(sample_indices) == list(range(len(sample_indices)))
        output_keys = [k for k in predictions[0].keys() if k != WEIGHT]  # weight is not needed in prediction outputs

        sorted_predictions = dict()
        for k in output_keys:
            predictions_per_key = torch.cat([batch_preds[k] for batch_preds in predictions])
            assert len(predictions_per_key) == len(sample_indices)
            sorted_predictions[k] = [
                x for _, x in sorted(zip(sample_indices, predictions_per_key), key=lambda ele: ele[0])
            ]
            sorted_predictions[k] = torch.stack(sorted_predictions[k])

        os.rmdir(self.output_dir)

        return [sorted_predictions]
