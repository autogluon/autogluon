import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from autogluon.common.utils.utils import setup_outputdir

from ..constants import AUTOMM, HF_MODELS, LAST_CHECKPOINT
from ..data import TextProcessor

logger = logging.getLogger(AUTOMM)


def save_pretrained_model_configs(
    model: nn.Module,
    config: DictConfig,
    path: str,
) -> DictConfig:
    """
    Save the pretrained model configs to local to make future loading not dependent on Internet access.
    By initializing models with local configs, Huggingface doesn't need to download pretrained weights from Internet.

    Parameters
    ----------
    model
        One model.
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The path to save pretrained model configs.
    """
    # TODO? Fix hardcoded model names.
    requires_saving = any([model_name.lower().startswith(HF_MODELS) for model_name in config.model.names])
    if not requires_saving:
        return config

    if (
        len(config.model.names) == 1
    ):  # TODO: Not sure this is a sufficient check. Hyperparameter "model.names" : ["hf_text", "fusion_mlp"] fails here.
        model = nn.ModuleList([model])
    else:  # assumes the fusion model has a model attribute, a nn.ModuleList
        model = model.model
    for per_model in model:
        if per_model.prefix.lower().startswith(HF_MODELS):
            per_model.config.save_pretrained(os.path.join(path, per_model.prefix))
            model_config = getattr(config.model, per_model.prefix)
            model_config.checkpoint_name = os.path.join("local://", per_model.prefix)

    return config


def save_text_tokenizers(
    text_processors: List[TextProcessor],
    path: str,
) -> List[TextProcessor]:
    """
    Save all the text tokenizers and record their relative paths, which are
    the corresponding model names, e.g, hf_text.

    Parameters
    ----------
    text_processors
        A list of text processors with tokenizers.
    path
        The root path.

    Returns
    -------
    A list of text processors with tokenizers replaced by their local relative paths.
    """
    for per_text_processor in text_processors:
        per_path = os.path.join(path, per_text_processor.prefix)
        per_text_processor.tokenizer.save_pretrained(per_path)
        per_text_processor.tokenizer = per_text_processor.prefix

    return text_processors


def process_save_path(path, resume: Optional[bool] = False, raise_if_exist: Optional[bool] = True):
    """
    Convert the provided path to an absolute path and check whether it is valid.
    If a path exists, either raise error or return None.
    A None path can be identified by the `setup_outputdir` to generate a random path.

    Parameters
    ----------
    path
        A provided path.
    resume
        Whether this is a path to resume training.
    raise_if_exist
        Whether to raise error if the path exists.

    Returns
    -------
    A complete and verified path or None.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if resume:
        assert os.path.isfile(os.path.join(path, LAST_CHECKPOINT)), (
            f"Trying to resume training from '{path}'. "
            f"However, it does not contain the last checkpoint file: '{LAST_CHECKPOINT}'. "
            "Are you using a correct path?"
        )
    elif os.path.isdir(path) and len(os.listdir(path)) > 0:
        if raise_if_exist:
            raise ValueError(
                f"Path {path} already exists."
                "Specify a new path to avoid accidentally overwriting a saved predictor."
            )
        else:
            logger.warning(
                "A new predictor save path is created."
                "This is to prevent you to overwrite previous predictor saved here."
                "You could check current save path at predictor._save_path."
                "If you still want to use this path, set resume=True"
            )
            path = None

    return path


def setup_save_path(
    resume: Optional[bool] = None,
    old_save_path: Optional[str] = None,
    proposed_save_path: Optional[str] = None,
    hyperparameter_tune_kwargs: Optional[dict] = None,
    warn_if_exist: Optional[bool] = True,
    raise_if_exist: Optional[bool] = False,
    model_loaded: Optional[bool] = None,
    fit_called: Optional[bool] = None,
):
    # TODO: remove redundant folders in DDP mode
    rank = int(os.environ.get("LOCAL_RANK", 0))
    save_path = None
    if resume:
        assert hyperparameter_tune_kwargs is None, "You can not resume training with HPO"
        save_path = process_save_path(path=old_save_path, resume=True)
    elif proposed_save_path is not None:  # TODO: distinguish DDP and existed predictor
        save_path = process_save_path(path=proposed_save_path, raise_if_exist=(raise_if_exist and rank == 0))
    elif old_save_path is not None:
        if model_loaded or fit_called:
            save_path = process_save_path(path=old_save_path, raise_if_exist=False)
        else:
            save_path = os.path.abspath(os.path.expanduser(old_save_path))

    if not resume:
        save_path = setup_outputdir(
            path=save_path,
            warn_if_exist=warn_if_exist,
        )

    save_path = os.path.abspath(os.path.expanduser(save_path))
    logger.debug(f"save path: {save_path}")

    return save_path
