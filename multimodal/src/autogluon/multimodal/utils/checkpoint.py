import logging
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import _METRIC

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


def average_checkpoints(
    checkpoint_paths: List[str],
):
    """
    Average a list of checkpoints' state_dicts.
    Reference: https://github.com/rwightman/pytorch-image-models/blob/master/avg_checkpoints.py

    Parameters
    ----------
    checkpoint_paths
        A list of model checkpoint paths.

    Returns
    -------
    The averaged state_dict.
    """
    if len(checkpoint_paths) > 1:
        avg_state_dict = {}
        avg_counts = {}
        for per_path in checkpoint_paths:
            state_dict = torch.load(per_path, map_location=torch.device("cpu"))["state_dict"]
            for k, v in state_dict.items():
                if k not in avg_state_dict:
                    avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                    avg_counts[k] = 1
                else:
                    avg_state_dict[k] += v.to(dtype=torch.float64)
                    avg_counts[k] += 1
            del state_dict

        for k, v in avg_state_dict.items():
            v.div_(avg_counts[k])

        # convert to float32.
        float32_info = torch.finfo(torch.float32)
        for k in avg_state_dict:
            avg_state_dict[k].clamp_(float32_info.min, float32_info.max).to(dtype=torch.float32)
    else:
        avg_state_dict = torch.load(checkpoint_paths[0], map_location=torch.device("cpu"))["state_dict"]

    return avg_state_dict


class AutoMMModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    Class that inherits pl.callbacks.ModelCheckpoint. The purpose is to resolve the potential issues in lightning.

    - Issue1:

    It solves the issue described in https://github.com/PyTorchLightning/pytorch-lightning/issues/5582.
    For ddp_spawn, the checkpoint_callback.best_k_models will be empty.
    Here, we resolve it by storing the best_models to "SAVE_DIR/best_k_models.yaml".

    """

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, _METRIC],
    ) -> None:
        super(AutoMMModelCheckpoint, self)._update_best_and_save(
            current=current, trainer=trainer, monitor_candidates=monitor_candidates
        )
        self.to_yaml()
