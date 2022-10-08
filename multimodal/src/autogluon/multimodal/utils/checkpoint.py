import logging
import os
import re
import shutil
from symbol import parameters
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import _METRIC, _PATH

from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import _PATH

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


class AutoMMModelCheckpointIO(pl.plugins.CheckpointIO):
    """
    Class that customizes how checkpoints are saved. Only save parameters that have been explicitly updated during training. Reduces memory footprint substentially when training very large models using parameter-efficient finetuning methods.
    Class is based on pl.plugins.TorchCheckpointIO.
    """

    def __init__(self, trainable_param_names):
        super().__init__()
        self.trainable_param_names = trainable_param_names

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )

        if self.trainable_param_names:
            updated_params = {}
            for name, param in checkpoint["state_dict"].items():
                if any([re.match(trainable_param_name, name) for trainable_param_name in self.trainable_param_names]):
                    updated_params[name] = param
        else:
            updated_params = checkpoint["state_dict"]

        checkpoint["state_dict"] = updated_params

        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # write the checkpoint dictionary on the file
            atomic_save(checkpoint, path)
        except AttributeError as err:
            # todo (sean): is this try catch necessary still?
            # https://github.com/Lightning-AI/lightning/pull/431
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            atomic_save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, map_location: Optional[Any] = None) -> Dict[str, Any]:
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")

        return pl_load(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            logger.debug(f"Removed checkpoint: {path}")


class AutoMMModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    Class that inherits pl.callbacks.ModelCheckpoint. The purpose is to resolve the potential issues in lightning.

    - Issue1:

    It solves the issue described in https://github.com/PyTorchLightning/pytorch-lightning/issues/5582.
    For ddp_spawn, the checkpoint_callback.best_k_models will be empty.
    Here, we resolve it by storing the best_models to "SAVE_DIR/best_k_models.yaml".

    """

    def _save_checkpoint(self, trainer, filepath):

        trainer.save_checkpoint(filepath, self.save_weights_only)

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

        if (
            trainer.strategy.strategy_name == "deepspeed"
        ):  # Deepspeed saves sharded model and optimizer states. Merging them default but does not maintain optimizer/lr-scheduler states.
            from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

            current_save_path = self.kth_best_model_path
            convert_zero_checkpoint_to_fp32_state_dict(current_save_path, current_save_path + ".tmp")
            shutil.rmtree(current_save_path)
            os.rename(current_save_path + ".tmp", current_save_path)
            client_state = torch.load(current_save_path, map_location=torch.device("cpu"))
            state_dict = client_state["state_dict"]
            state_dict_new = {}
            # for name, parameter in state_dict.items():
            #     state_dict_new[name.replace("model.", "", 1)] = parameter # Remove model prefix to remain consistent.
            client_state["state_dict"] = state_dict
            torch.save(client_state, current_save_path)
