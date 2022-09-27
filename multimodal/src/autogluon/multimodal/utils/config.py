import copy
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import AUTOMM, HF_MODELS, REGRESSION, VALID_CONFIG_KEYS
from ..presets import get_automm_presets, get_basic_automm_config

logger = logging.getLogger(AUTOMM)


def filter_search_space(hyperparameters: dict, keys_to_filter: Union[str, List[str]]):
    """
    Filter search space within hyperparameters without the given keys as prefixes.
    Hyperparameters that are not search space will not be filtered.

    Parameters
    ----------
    hyperparameters
        A dictionary containing search space and overrides to config.
    keys_to_filter
        Keys that needs to be filtered out

    Returns
    -------
        hyperparameters being filtered
    """
    assert any(
        key.startswith(valid_keys) for valid_keys in VALID_CONFIG_KEYS for key in keys_to_filter
    ), f"Invalid keys: {keys_to_filter}. Valid options are {VALID_CONFIG_KEYS}"
    from ray.tune.sample import Domain

    from autogluon.core.space import Space

    hyperparameters = copy.deepcopy(hyperparameters)
    if isinstance(keys_to_filter, str):
        keys_to_filter = [keys_to_filter]
    for hyperparameter, value in hyperparameters.copy().items():
        if not isinstance(value, (Space, Domain)):
            continue
        for key in keys_to_filter:
            if hyperparameter.startswith(key):
                del hyperparameters[hyperparameter]
    return hyperparameters


def get_config(
    presets: Optional[str] = None,
    config: Optional[Union[dict, DictConfig]] = None,
    overrides: Optional[Union[str, List[str], Dict]] = None,
    is_distill: Optional[bool] = False,
):
    """
    Construct configurations for model, data, optimization, and environment.
    It supports to overrides some default configurations.

    Parameters
    ----------
    presets
        Name of the presets.
    config
        A dictionary including four keys: "model", "data", "optimization", and "environment".
        If any key is not not given, we will fill in with the default value.

        The value of each key can be a string, yaml path, or DictConfig object. For example:
        config = {
                        "model": "fusion_mlp_image_text_tabular",
                        "data": "default",
                        "optimization": "adamw",
                        "environment": "default",
                    }
            or
            config = {
                        "model": "/path/to/model/config.yaml",
                        "data": "/path/to/data/config.yaml",
                        "optimization": "/path/to/optimization/config.yaml",
                        "environment": "/path/to/environment/config.yaml",
                    }
            or
            config = {
                        "model": OmegaConf.load("/path/to/model/config.yaml"),
                        "data": OmegaConf.load("/path/to/data/config.yaml"),
                        "optimization": OmegaConf.load("/path/to/optimization/config.yaml"),
                        "environment": OmegaConf.load("/path/to/environment/config.yaml"),
                    }
    overrides
        This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            overrides = "model.hf_text.checkpoint_name=google/electra-small-discriminator
            model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            overrides = ["model.hf_text.checkpoint_name=google/electra-small-discriminator",
            "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            overrides = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }
    is_distill
        Whether in the distillation mode.

    Returns
    -------
    Configurations as a DictConfig object
    """
    if config is None:
        config = {}

    if not config and not presets:
        presets = "default"

    if not isinstance(config, DictConfig):
        basic_config = get_basic_automm_config(is_distill=is_distill)
        if presets is None:
            preset_overrides = None
        else:
            preset_overrides = get_automm_presets(presets=presets)

        for k, default_value in basic_config.items():
            if k not in config:
                config[k] = default_value

        all_configs = []
        for k, v in config.items():
            if isinstance(v, dict):
                per_config = OmegaConf.create(v)
            elif isinstance(v, DictConfig):
                per_config = v
            elif isinstance(v, str):
                if v.lower().endswith((".yaml", ".yml")):
                    per_config = OmegaConf.load(os.path.expanduser(v))
                else:
                    cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    config_path = os.path.join(cur_path, "configs", k, f"{v}.yaml")
                    per_config = OmegaConf.load(config_path)
            else:
                raise ValueError(f"Unknown configuration type: {type(v)}")

            all_configs.append(per_config)

        config = OmegaConf.merge(*all_configs)
        # apply the preset's overrides
        if preset_overrides:
            config = apply_omegaconf_overrides(config, overrides=preset_overrides, check_key_exist=True)

    verify_model_names(config.model)
    logger.debug(f"overrides: {overrides}")
    if overrides is not None:
        # avoid manipulating the user-provided overrides
        overrides = copy.deepcopy(overrides)
        # apply customized model names
        overrides = parse_dotlist_conf(overrides)  # convert to a dict
        config.model = customize_model_names(
            config=config.model,
            customized_names=overrides.get("model.names", None),
        )
        # remove `model.names` from overrides since it's already applied.
        overrides.pop("model.names", None)
        # apply the user-provided overrides
        config = apply_omegaconf_overrides(config, overrides=overrides, check_key_exist=True)
    verify_model_names(config.model)
    return config


def verify_model_names(config: DictConfig):
    """
    Verify whether provided model names are valid.

    Parameters
    ----------
    config
        Config should have a attribute `names`, which contains a list of
        attribute names, e.g., ["timm_image", "hf_text"]. And each string in
        `config.names` should also be a attribute of `config`, e.g, `config.timm_image`.
    """
    # must have attribute `names`
    assert hasattr(config, "names")
    # return if no names available
    if not config.names:
        return
    # assure no duplicate names
    assert len(config.names) == len(set(config.names))
    # verify that strings in `config.names` match the keys of `config`.
    keys = list(config.keys())
    keys.remove("names")
    assert set(config.names).issubset(set(keys)), f"`{config.names}` do not match config keys {keys}"

    # verify that no name starts with another one
    names = sorted(config.names, key=lambda ele: len(ele), reverse=True)
    for i in range(len(names)):
        if names[i].startswith(tuple(names[i + 1 :])):
            raise ValueError(f"name {names[i]} starts with one of another name: {names[i+1:]}")


def get_name_prefix(
    name: str,
    prefixes: List[str],
):
    """
    Get a name's prefix from some available candidates.

    Parameters
    ----------
    name
        A name string
    prefixes
        Available prefixes.

    Returns
    -------
        Prefix of the name.
    """
    search_results = [pre for pre in prefixes if name.lower().startswith(pre)]
    if len(search_results) == 0:
        return None
    elif len(search_results) >= 2:
        raise ValueError(
            f"Model name `{name}` is mapped to multiple models, "
            f"which means some names in `{prefixes}` have duplicate prefixes."
        )
    else:
        return search_results[0]


def customize_model_names(
    config: DictConfig,
    customized_names: Union[str, List[str]],
):
    """
    Customize attribute names of `config` with the provided names.
    A valid customized name string should start with one available name
    string in `config`.

    Parameters
    ----------
    config
        Config should have a attribute `names`, which contains a list of
        attribute names, e.g., ["timm_image", "hf_text"]. And each string in
        `config.names` should also be a attribute of `config`, e.g, `config.timm_image`.
    customized_names
        The provided names to replace the existing ones in `config.names` as well as
        the corresponding attribute names. For example, if `customized_names` is
        ["timm_image_123", "hf_text_abc"], then `config.timm_image` and `config.hf_text`
        are changed to `config.timm_image_123` and `config.hf_text_abc`.

    Returns
    -------
        A new config with its first-level attributes customized by the provided names.
    """
    if not customized_names:
        return config

    if isinstance(customized_names, str):
        customized_names = OmegaConf.from_dotlist([f"names={customized_names}"]).names

    new_config = OmegaConf.create()
    new_config.names = []
    available_prefixes = list(config.keys())
    available_prefixes.remove("names")
    for per_name in customized_names:
        per_prefix = get_name_prefix(
            name=per_name,
            prefixes=available_prefixes,
        )
        if per_prefix:
            per_config = getattr(config, per_prefix)
            setattr(new_config, per_name, copy.deepcopy(per_config))
            new_config.names.append(per_name)
        else:
            logger.debug(f"Removing {per_name}, which doesn't start with any of these prefixes: {available_prefixes}.")

    if len(new_config.names) == 0:
        raise ValueError(
            f"No customized name in `{customized_names}` starts with name prefixes in `{available_prefixes}`."
        )

    return new_config


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


def get_local_pretrained_config_paths(config: DictConfig, path: str) -> DictConfig:
    """
    Get the local config paths of hugginface pretrained models. With a local config,
    Hugginface can initialize a model without having to download its pretrained weights.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The saving path to the pretrained model configs.
    """
    for model_name in config.model.names:
        if model_name.lower().startswith(HF_MODELS):
            model_config = getattr(config.model, model_name)
            if model_config.checkpoint_name.startswith("local://"):
                model_config.checkpoint_name = os.path.join(path, model_config.checkpoint_name[len("local://") :])
                assert os.path.exists(
                    os.path.join(model_config.checkpoint_name, "config.json")
                )  # guarantee the existence of local configs

    return config


def parse_dotlist_conf(conf):
    """
    Parse the config files that is potentially in the dotlist format to a dictionary.

    Parameters
    ----------
    conf
        Apply the conf stored as dotlist, e.g.,
         'aaa=a, bbb=b' or ['aaa=a, ', 'bbb=b'] to {'aaa': 'a', 'bbb': b}

    Returns
    -------
    new_conf
    """
    if isinstance(conf, str):
        conf = conf.split()
        need_parse = True
    elif isinstance(conf, (list, tuple)):
        need_parse = True
    elif isinstance(conf, dict):
        need_parse = False
    else:
        raise ValueError(f"Unsupported format of conf={conf}")
    if need_parse:
        new_conf = dict()
        curr_key = None
        curr_value = ""
        for ele in conf:
            if "=" in ele:
                key, v = ele.split("=")
                if curr_key is not None:
                    new_conf[curr_key] = curr_value
                curr_key = key
                curr_value = v
            else:
                if curr_key is None:
                    raise ValueError(f"Cannot parse the conf={conf}")
                curr_value = curr_value + " " + ele
        if curr_key is not None:
            new_conf[curr_key] = curr_value
        return new_conf
    else:
        return conf


def apply_omegaconf_overrides(
    conf: DictConfig,
    overrides: Union[List, Tuple, str, Dict, DictConfig],
    check_key_exist=True,
):
    """
    Apply omegaconf overrides.

    Parameters
    ----------
    conf
        The base configuration.
    overrides
        The overrides can be a string or a list.
    check_key_exist
        Whether to check if all keys in the overrides must exist in the conf.

    Returns
    -------
    new_conf
        The updated configuration.
    """
    overrides = parse_dotlist_conf(overrides)

    def _check_exist_dotlist(C, key_in_dotlist):
        if not isinstance(key_in_dotlist, list):
            key_in_dotlist = key_in_dotlist.split(".")
        if key_in_dotlist[0] in C:
            if len(key_in_dotlist) > 1:
                return _check_exist_dotlist(C[key_in_dotlist[0]], key_in_dotlist[1:])
            else:
                return True
        else:
            return False

    if check_key_exist:
        for ele in overrides.items():
            if not _check_exist_dotlist(conf, ele[0]):
                raise KeyError(
                    f'"{ele[0]}" is not found in the config. You may need to check the overrides. '
                    f"overrides={overrides}"
                )
    override_conf = OmegaConf.from_dotlist([f"{ele[0]}={ele[1]}" for ele in overrides.items()])
    conf = OmegaConf.merge(conf, override_conf)
    return conf


def update_config_by_rules(
    problem_type: str,
    config: DictConfig,
):
    """
    Modify configs based on the need of loss func.
    Now it support changing the preprocessing of numerical label into Minmaxscaler while using BCEloss.

    Parameters
    ----------
    problem_type
        The type of the problem of the project.
    config
        The config of the project. It is a Dictconfig object.

    Returns
    -------
    The modified config.
    """
    loss_func = OmegaConf.select(config, "optimization.loss_function")
    if loss_func is not None:
        if problem_type == REGRESSION and "bce" in loss_func.lower():
            # We are using BCELoss for regression problems. Need to first scale the labels.
            config.data.label.numerical_label_preprocessing = "minmaxscaler"
        elif loss_func != "auto":
            warnings.warn(
                f"Received loss function={loss_func} for problem={problem_type}. "
                "Currently, we only support using BCE loss for regression problems and choose "
                "the loss_function automatically otherwise.",
                UserWarning,
            )
    return config
