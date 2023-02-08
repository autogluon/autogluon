import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from ..constants import AUTOMM, DEEPSPEED_STRATEGY
from .cloud_io import _atomic_save
from .cloud_io import _load as pl_load
from .cloud_io import get_filesystem

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
    Class that customizes how checkpoints are saved. Saves either the entire model or only parameters that have been explicitly updated during training. The latter reduces memory footprint substentially when training very large models with parameter-efficient finetuning methods.
    Class is based on pl.plugins.TorchCheckpointIO.

    """

    def __init__(self, trainable_param_names, model_name_to_id):
        """
        Parameters
        ----------
        trainable_param_names
            A list of regular expressions or exact names of layers to filter which parameters should be saved. If empty save entire model.
        model_name_to_id
            A dictionary mapping the layer names (keys) of the model to their ids (values).
        """
        super().__init__()
        self.trainable_param_names = trainable_param_names
        self.model_name_to_id = model_name_to_id

    def save_checkpoint(self, checkpoint: Dict[str, Any], path, storage_options: Optional[Any] = None) -> None:
        """
        Save model/training states as a checkpoint file through state-dump and file-write.

        Parameters
        ----------
        checkpoint
            dict containing model and trainer state
        path
            write-target path
        storage_options
            Optional parameters when saving the model/training states. Not currently considered.
        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`."
            )

        if "state_dict" in checkpoint:
            if self.trainable_param_names:
                updated_params = {}
                for name, param in checkpoint["state_dict"].items():
                    adjusted_name = name.replace("model.", "", 1)
                    if adjusted_name in self.model_name_to_id and self.model_name_to_id[adjusted_name] == 0:
                        updated_params[name] = param
                    if any(
                        [re.match(trainable_param_name, name) for trainable_param_name in self.trainable_param_names]
                    ):
                        updated_params[name] = param
            else:
                updated_params = checkpoint["state_dict"]

            checkpoint["state_dict"] = updated_params

        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # write the checkpoint dictionary on the file
            _atomic_save(checkpoint, path)
        except AttributeError as err:
            # todo (sean): is this try catch necessary still?
            # https://github.com/Lightning-AI/lightning/pull/431
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            _atomic_save(checkpoint, path)

    def load_checkpoint(self, path, map_location: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages.

        Parameters
        ----------
        path
            Path to checkpoint
        map_location
            a function, torch.device, string or a dict specifying how to remap storage locations.
        """

        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")

        return pl_load(path, map_location=map_location)

    def remove_checkpoint(self, path) -> None:
        """
        Remove checkpoint file from the filesystem.

        Parameters
        ----------
        path
            Path to checkpoint
        """
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
        monitor_candidates: Dict[str, torch.Tensor],
    ) -> None:

        super(AutoMMModelCheckpoint, self)._update_best_and_save(
            current=current, trainer=trainer, monitor_candidates=monitor_candidates
        )
        self.to_yaml()

        if (
            trainer.strategy.strategy_name == DEEPSPEED_STRATEGY
        ):  # Deepspeed saves model and optimizer states in a sharded state in seperate folder (even when using single GPU). Merging folder to single checkpoint file.
            from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

            current_save_path = self.kth_best_model_path
            convert_zero_checkpoint_to_fp32_state_dict(current_save_path, current_save_path + ".tmp")
            shutil.rmtree(current_save_path)
            os.rename(current_save_path + ".tmp", current_save_path)
            client_state = torch.load(current_save_path, map_location=torch.device("cpu"))
            state_dict = client_state["state_dict"]
            client_state["state_dict"] = state_dict
            torch.save(client_state, current_save_path)
