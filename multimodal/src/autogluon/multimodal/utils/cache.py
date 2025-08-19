import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from ..constants import (
    AUG_LOGITS,
    BBOX,
    LOGIT_SCALE,
    MULTIMODAL_FEATURES,
    MULTIMODAL_FEATURES_POST_AUG,
    MULTIMODAL_FEATURES_PRE_AUG,
    ORI_LOGITS,
    VAE_MEAN,
    VAE_VAR,
    WEIGHT,
)

logger = logging.getLogger(__name__)


class DDPPredictionWriter(BasePredictionWriter):
    def __init__(
        self, output_dir: Optional[str], write_interval: Optional[str], strategy: Optional[str], sleep_time=5
    ):
        """
        Parameters
        ----------
        output_dir
            The directory to save predictions.
        write_interval
            When to write. Could be "batch" or "epoch".
            See Lightning's source code at
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html
        sleep_time
            If other process does not finish writing, sleep for a few seconds and recheck.
        """
        super().__init__(write_interval)
        if output_dir is None:  # TODO: give the predictor a default save_path before calling DDPPredictionWriter
            output_dir = "temp_cache_for_ddp_prediction"
            logging.warning(
                f"Current predictor's save_path is None, using a default cache folder which may cause an error in prediction I/O. Try init the predictor with a save_path."
            )
        assert isinstance(output_dir, (str, Path)), (
            f"Only str and pathlib.Path types are supported for path, but got {output_dir} of type {type(output_dir)}."
        )
        self.sleep_time = sleep_time
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if "spawn" in strategy:
            self.output_dir = os.path.join(output_dir, f"predictions_{uuid.uuid4().hex}")
        else:
            self.output_dir = os.path.join(output_dir, "ddp_prediction_cache")
        try:
            os.makedirs(self.output_dir, exist_ok=False)
        except FileExistsError:  # assume the string is unique
            logger.warning(
                f"{self.output_dir} already exists. This could be caused by DDP subprocess. Just make sure the previous cache is removed in {self.output_dir}."
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
        torch.save(predictions, self.get_predictions_cache_dir(trainer.global_rank))  # nosec B614
        # here we save `batch_indices` to get the information about the data index
        # from prediction data
        torch.save(batch_indices, self.get_batch_indices_cache_dir(trainer.global_rank))  # nosec B614

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
        sample_indices = torch.load(sample_indices_file)  # nosec B614
        predictions = torch.load(predictions_file)  # nosec B614

        return sample_indices, predictions

    def flatten(self, x):
        """
        Flatten nested lists into one list.

        Parameters
        ----------
        x
            A nested list to be flattened.
        """
        if isinstance(x, list):
            return [a for i in x for a in self.flatten(i)]
        else:
            return [x]

    def collate(self, x: List[Dict]):
        """
        Collate a list of dictionaries into one.
        Each value is a tensor concatenated from a list of batch tensors.

        Parameters
        ----------
        x
            A list of batch results. Each batch is one (nested) dictionary.
        """
        if BBOX in x[0]:  # for detection outputs just sort the batches
            return x

        results = dict()
        if len(x[0]) == 0:  # dict is empty
            return dict()

        for k, v in x[0].items():
            if k in [
                WEIGHT,
                LOGIT_SCALE,
                MULTIMODAL_FEATURES,
                MULTIMODAL_FEATURES_PRE_AUG,
                MULTIMODAL_FEATURES_POST_AUG,
                ORI_LOGITS,
                AUG_LOGITS,
                VAE_MEAN,
                VAE_VAR,
            ]:  # ignore the keys
                continue
            elif isinstance(v, dict):
                results[k] = self.collate([i[k] for i in x])
            elif isinstance(v, torch.Tensor):
                results[k] = torch.cat([i[k] for i in x])
            else:
                raise ValueError(
                    f"Unsupported data type {type(v)} is found in prediction results. "
                    f"We only support dictionary with torch tensor values."
                )

        return results

    def sort(self, x: Dict, indices: List):
        """
        Sort the values of each key according to the indices.
        This is to make sure that prediction results follow the order of input samples.

        Parameters
        ----------
        x
            A dictionary with all the predictions.
        indices
            A list of indices.
        """
        if isinstance(x, list) and BBOX in x[0]:  # for detection outputs just sort the batches
            return [xi for _, xi in sorted(zip(indices, x), key=lambda ele: ele[0])]

        results = dict()
        for k, v in x.items():
            if isinstance(v, dict):
                results[k] = self.sort(v, indices)
            else:
                assert len(indices) == len(v), (
                    f"Size mismatch, {k}: {v} of len {len(v)} and indices {indices} of length {len(indices)}"
                )
                results[k] = [x for _, x in sorted(zip(indices, v), key=lambda ele: ele[0])]
                results[k] = torch.stack(results[k])

        return results

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
        predictions = self.collate(predictions)
        sorted_predictions = self.sort(predictions, indices=sample_indices)
        shutil.rmtree(self.output_dir)

        return [sorted_predictions]
