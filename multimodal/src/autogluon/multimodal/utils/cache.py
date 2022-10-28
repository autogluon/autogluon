import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


# TODO: doc for this file
class DDPCacheWriter(BasePredictionWriter):
    def __init__(self, pipeline, write_interval, sleep_time=5):
        super().__init__(write_interval)
        self._pipeline = pipeline
        self._sleep_time = sleep_time
        self.set_cache_path()

    def set_cache_path(self):
        # TODO: Enable running multiple DDP programs at the same time.
        self.cache_dir = f"AutogluonDDPCache_{self._pipeline}"
        assert isinstance(
            self.cache_dir, (str, Path)
        ), f"Only str and pathlib.Path types are supported for path, got {self.cache_dir} of type {type(self.cache_dir)}."
        try:
            os.makedirs(self.cache_dir, exist_ok=False)
        except FileExistsError:
            logger.warning(
                f'Warning: cache path already exists! It should be created by other process. Please make sure not running multiple DDP programs simutaneously! path="{self.cache_dir}"'
            )
        self.cache_dir = os.path.expanduser(self.cache_dir)  # replace ~ with absolute path if it exists
        if self.cache_dir[-1] != os.path.sep:
            self.cache_dir = self.cache_dir + os.path.sep

    def get_predictions_cache_dir(self, global_rank):
        return os.path.join(self.cache_dir, f"predictions_{global_rank}.pt")

    def get_batch_indices_cache_dir(self, global_rank):
        return os.path.join(self.cache_dir, f"batch_indices_{global_rank}.pt")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `cache_dir` each containing
        # the predictions of its respective rank
        torch.save(predictions, self.get_predictions_cache_dir(trainer.global_rank))
        # here we save `batch_indices` to get the information about the data index
        # from prediction data
        torch.save(batch_indices, self.get_batch_indices_cache_dir(trainer.global_rank))

    def read_single_gpu_result(self, gpu_idx):
        batch_indices_file = self.get_batch_indices_cache_dir(gpu_idx)
        predictions_file = self.get_predictions_cache_dir(gpu_idx)
        while (not os.path.exists(batch_indices_file)) or (not os.path.exists(predictions_file)):
            logger.info(f"waiting for gpu #{gpu_idx}...")
            time.sleep(self._sleep_time)
        batch_indices = torch.load(batch_indices_file)[0]
        predictions = torch.load(predictions_file)[0]
        os.remove(batch_indices_file)
        os.remove(predictions_file)
        return batch_indices, predictions

    def collect_all_gpu_results(self, num_gpus):

        outputs = defaultdict(dict)
        output_keys = []
        for gpu_idx in range(num_gpus):
            batch_indices, predictions = self.read_single_gpu_result(gpu_idx)
            if not output_keys:
                output_keys = list(predictions[0].keys())
            for group_idx, batch_group in enumerate(batch_indices):
                for in_group_batch_idx, batch_idx in enumerate(batch_group):
                    # TODO: check if this is available to other tasks as well
                    for output_key in output_keys:
                        outputs[batch_idx][output_key] = predictions[group_idx][output_key][in_group_batch_idx]

        os.rmdir(self.cache_dir)

        outputs = [outputs[i] for i in range(len(outputs))]
        return outputs
