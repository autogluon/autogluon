"""
Some utilities are copied from
https://github.com/Lightning-AI/lightning/blob/master/src/lightning/fabric/utilities/cloud_io.py
to address warnings:
LightningDeprecationWarning: lightning.pytorch.utilities.cloud_io.atomic_save has been
deprecated in v1.8.0 and will be removed in v1.10.0. This function is internal but you
can copy over its implementation.
"""

import io
import logging
import os
import re
import shutil
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union

import fsspec
import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from .env import get_filesystem

_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]

logger = logging.getLogger(__name__)


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
            if os.path.isdir(per_path + "-dir"):  # deepspeed save checkpoints into a directory
                from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

                convert_zero_checkpoint_to_fp32_state_dict(per_path + "-dir", per_path)
                shutil.rmtree(per_path + "-dir")
                state_dict = torch.load(per_path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]  # nosec B614
            else:
                state_dict = torch.load(per_path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]  # nosec B614
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
        avg_state_dict = torch.load(checkpoint_paths[0], map_location=torch.device("cpu"), weights_only=False)[
            "state_dict"
        ]  # nosec B614

    return avg_state_dict


def pl_load(
    path_or_url: Union[IO, str, Path],
    map_location: _MAP_LOCATION_TYPE = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)  # nosec B614
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)  # nosec B614


def pl_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)  # nosec B614
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())


class AutoMMModelCheckpointIO(pl.plugins.CheckpointIO):
    """
    Class that customizes how checkpoints are saved. Saves either the entire model or only parameters that have been explicitly updated during training. The latter reduces memory footprint substantially when training very large models with parameter-efficient finetuning methods.
    Class is based on plugins.TorchCheckpointIO.

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
            pl_save(checkpoint, path)
        except AttributeError as err:
            # todo (sean): is this try catch necessary still?
            # https://github.com/Lightning-AI/lightning/pull/431
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            pl_save(checkpoint, path)

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
    Class that inherits callbacks.ModelCheckpoint. The purpose is to resolve the potential issues in lightning.

    - Issue1:

    It solves the issue described in https://github.com/Lightning-AI/lightning/issues/5582.
    For ddp_spawn, the checkpoint_callback.best_k_models will be empty.
    Here, we resolve it by storing the best_models to "SAVE_DIR/best_k_models.yaml".

    """

    def _save_checkpoint(self, trainer, filepath):
        # Deepspeed saves model and optimizer states in a shared state in a separate folder
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            trainer.save_checkpoint(filepath + "-dir", self.save_weights_only)
        else:
            trainer.save_checkpoint(filepath, self.save_weights_only)

        # Required to avoid redundant evaluation and checkpointing
        self._last_global_step_saved = trainer.global_step

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
