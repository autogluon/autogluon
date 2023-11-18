import copy
import logging
import os
import re
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from packaging import version
from torch import nn

from ..constants import (
    AUTOMM,
    DATA,
    FT_TRANSFORMER,
    FUSION_TRANSFORMER,
    HF_MODELS,
    MODEL,
    REGRESSION,
    VALID_CONFIG_KEYS,
)
from ..presets import get_automm_presets, get_basic_automm_config
from .data import get_detected_data_types

logger = logging.getLogger(__name__)


def filter_search_space(hyperparameters: Dict, keys_to_filter: Union[str, List[str]]):
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
    if isinstance(keys_to_filter, str):
        keys_to_filter = [keys_to_filter]

    assert any(
        key.startswith(valid_keys) for valid_keys in VALID_CONFIG_KEYS for key in keys_to_filter
    ), f"Invalid keys: {keys_to_filter}. Valid options are {VALID_CONFIG_KEYS}"
    from ray.tune.search.sample import Domain

    from autogluon.common import space

    hyperparameters = copy.deepcopy(hyperparameters)
    for hyperparameter, value in hyperparameters.copy().items():
        if not isinstance(value, (space.Space, Domain)):
            continue
        for key in keys_to_filter:
            if hyperparameter.startswith(key):
                del hyperparameters[hyperparameter]
    return hyperparameters


def get_default_config(config: Optional[Union[Dict, DictConfig]] = None, extra: Optional[List[str]] = None):
    """
    Get the default config.

    Parameters
    ----------
    config
        A dictionary including four keys: "model", "data", "optimization", and "environment".
        If any key is not given, we will fill in with the default value.
    extra
        A list of extra config keys.

    Returns
    -------
    The default config.
    """
    if isinstance(config, DictConfig):
        return config

    if config is None:
        config = {}

    basic_config = get_basic_automm_config(extra=extra)
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

    return config


def get_config(
    problem_type: Optional[str] = None,
    presets: Optional[str] = None,
    config: Optional[Union[Dict, DictConfig]] = None,
    overrides: Optional[Union[str, List[str], Dict]] = None,
    extra: Optional[List[str]] = None,
):
    """
    Construct configurations for model, data, optimization, and environment.
    It supports to overrides some default configurations.

    Parameters
    ----------
    problem_type
        Problem type.
    presets
        Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
    config
        A dictionary including four keys: "model", "data", "optimization", and "environment".
        If any key is not given, we will fill in with the default value.

        The value of each key can be a string, yaml path, or DictConfig object. For example:
        config = {
                        "model": "default",
                        "data": "default",
                        "optimization": "default",
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
    extra
        A list of extra config keys.

    Returns
    -------
    Configurations as a DictConfig object
    """

    if not config and not presets:
        presets = "default"

    if not isinstance(config, DictConfig):
        if presets is None:
            preset_overrides = None
        else:
            preset_overrides, _ = get_automm_presets(problem_type=problem_type, presets=presets)

        config = get_default_config(config, extra=extra)
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
        config.model, _ = customize_model_names(
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
    advanced_hyperparameters: Optional[Dict] = None,
):
    """
    Customize attribute names of `config` with the provided names.
    A valid customized name string should start with one available name
    string in `config`. Customizing the model names in advanced_hyperparameters
    is only used for matcher query and response models currently.

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
    advanced_hyperparameters
        The hyperparameters whose values are complex objects, which can't be stored in config.

    Returns
    -------
        A new config with its first-level attributes customized by the provided names.
    """
    if not customized_names:
        return config, advanced_hyperparameters

    if isinstance(customized_names, str):
        customized_names = OmegaConf.from_dotlist([f"names={customized_names}"]).names

    if advanced_hyperparameters:
        new_advanced_hyperparameters = copy.deepcopy(advanced_hyperparameters)
    else:
        new_advanced_hyperparameters = dict()

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

            if advanced_hyperparameters:
                for k, v in advanced_hyperparameters.items():
                    if k.startswith(f"{MODEL}.{per_prefix}"):
                        new_k = k.replace(f"{MODEL}.{per_prefix}", f"{MODEL}.{per_name}")
                        new_advanced_hyperparameters.pop(k)
                        new_advanced_hyperparameters[new_k] = v
        else:
            logger.debug(f"Removing {per_name}, which doesn't start with any of these prefixes: {available_prefixes}.")

    if len(new_config.names) == 0:
        raise ValueError(
            f"No customized name in `{customized_names}` starts with name prefixes in `{available_prefixes}`."
        )

    return new_config, new_advanced_hyperparameters


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


def upgrade_config(config, loaded_version):
    """Upgrade outdated configurations

    Parameters
    ----------
    config
        The configuration
    loaded_version
        The version of the config that has been loaded

    Returns
    -------
    config
        The upgraded configuration
    """
    # backward compatibility for variable image size.
    if version.parse(loaded_version) <= version.parse("0.6.2"):
        logger.info(f"Start to upgrade the previous configuration trained by AutoMM version={loaded_version}.")
        if OmegaConf.select(config, "model.timm_image") is not None:
            logger.warning(
                "Loading a model that has been trained via AutoGluon Multimodal<=0.6.2. "
                "Setting config.model.timm_image.image_size = None."
            )
            config.model.timm_image.image_size = None
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
            # To use BCELoss for regression problems, need to first scale the labels.
            config.data.label.numerical_label_preprocessing = "minmaxscaler"

    return config


def update_tabular_config_by_resources(
    config: DictConfig,
    num_numerical_columns: Optional[int] = 0,
    num_categorical_columns: Optional[int] = 0,
    resource: Optional[int] = 16,
):
    """
    Modify configs based on the dataset statistics.
    Use Additive attention with large column count and tune batch size accordingly.
    Parameters
    ----------
    config
        The config of the project. It is a Dictconfig object.
    num_numerical_columns
        The number of numerical columns.
    num_categorical_columns
        The number of categorical columns.
    resource
        The maximum resource (memory in GB) a single GPU has.
    Returns
    -------
    The modified config.
    """
    columns_per_model = {
        FUSION_TRANSFORMER: num_categorical_columns + num_numerical_columns,
        FT_TRANSFORMER: num_categorical_columns + num_numerical_columns,
    }

    # Threshold is expected to be ~= batch_size * num_tokens, for additive attention.
    # The multiplier 2e4 is a heuristic found from AutoML Benchmark.
    # TODO: determine the threshold/batch_size on training data directly
    threshold = resource * 2e4
    per_gpu_batch_size = config.env.per_gpu_batch_size
    for model in columns_per_model:
        if model in config.model.names:
            model_ = getattr(config.model, model)
            if columns_per_model[model] > 300 and model_.additive_attention == "auto":
                model_.additive_attention = True
                model_.share_qv_weights = True if model_.share_qv_weights == "auto" else model_.share_qv_weights
                warnings.warn(
                    f"Dataset contains >300 features, using additive attention for efficiency",
                    UserWarning,
                )
                if columns_per_model[model] * per_gpu_batch_size > threshold:
                    per_gpu_batch_size = int(threshold / columns_per_model[model])

            model_.additive_attention = False if model_.additive_attention == "auto" else model_.additive_attention
            model_.share_qv_weights = False if model_.share_qv_weights == "auto" else model_.share_qv_weights

    per_gpu_batch_size = max(per_gpu_batch_size, 1)
    if per_gpu_batch_size < config.env.per_gpu_batch_size:
        config.env.per_gpu_batch_size = per_gpu_batch_size
        warnings.warn(
            f"Setting  per_gpu_batch_size to {per_gpu_batch_size} to fit into GPU memory",
            UserWarning,
        )

    return config


def get_pretrain_configs_dir(subfolder: Optional[str] = None):
    import autogluon.multimodal

    pretrain_config_dir = os.path.join(autogluon.multimodal.__path__[0], "configs", "pretrain")
    if subfolder:
        pretrain_config_dir = os.path.join(pretrain_config_dir, subfolder)
    return pretrain_config_dir


def filter_timm_pretrained_cfg(cfg, remove_source=False, remove_null=True):
    filtered_cfg = {}
    keep_null = {"pool_size", "first_conv", "classifier"}  # always keep these keys, even if none
    for k, v in cfg.items():
        if remove_source and k in {"url", "file", "hf_hub_id", "hf_hub_id", "hf_hub_filename", "source"}:
            continue
        if remove_null and v is None and k not in keep_null:
            continue
        filtered_cfg[k] = v
    return filtered_cfg


def update_hyperparameters(
    problem_type,
    presets,
    provided_hyperparameters,
    provided_hyperparameter_tune_kwargs,
):
    """
    Update preset hyperparameters hyperparameter_tune_kwargs by the provided.
    Currently, this is mainly used for HPO presets, which define some searchable hyperparameters.
    We need to combine these searchable hyperparameters with ones provided by users.

    Parameters
    ----------
    problem_type
        Problem type.
    presets
        A preset string regarding modality quality or hpo.
    provided_hyperparameters
        The hyperparameters provided by users.
    provided_hyperparameter_tune_kwargs
        The hyperparameter_tune_kwargs provided by users.

    Returns
    -------
    The updated hyperparameters and hyperparameter_tune_kwargs.
    """
    hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(problem_type=problem_type, presets=presets)

    if hyperparameter_tune_kwargs and provided_hyperparameter_tune_kwargs:
        hyperparameter_tune_kwargs.update(provided_hyperparameter_tune_kwargs)
    elif provided_hyperparameter_tune_kwargs:
        hyperparameter_tune_kwargs = provided_hyperparameter_tune_kwargs

    if hyperparameter_tune_kwargs:
        if provided_hyperparameters:
            hyperparameters.update(provided_hyperparameters)
    else:  # use the provided hyperparameters if no hpo. The preset hyperparameters will be also used later in get_config.
        hyperparameters = provided_hyperparameters

    if hyperparameter_tune_kwargs:
        assert isinstance(
            hyperparameters, dict
        ), "Please provide hyperparameters as a dictionary if you want to do HPO"

    return hyperparameters, hyperparameter_tune_kwargs


def filter_hyperparameters(
    hyperparameters: Dict,
    column_types: Dict,
    config: Optional[Union[Dict, DictConfig]] = None,
    fit_called: Optional[bool] = False,
):
    """
    Filter out the hyperparameters that have no effect for HPO.

    Parameters
    ----------
    hyperparameters
        The hyperparameters to override the default config.
    column_types
        Dataframe's column types.
    config
        A config provided by users or from the previous training.
    fit_called
        Whether fit() has been called.

    Returns
    -------
    The filtered hyperparameters.
    """
    model_names_key = f"{MODEL}.names"
    keys_to_filter = []

    if model_names_key in hyperparameters:
        # If continuous training or config is provided, make sure models are in config.model.
        config = get_default_config(config)
        selected_model_names = []
        config_model_names = list(config.model.keys())
        config_model_names.remove("names")
        for name in hyperparameters[model_names_key]:
            if name in config_model_names:
                selected_model_names.append(name)
        hyperparameters[model_names_key] = selected_model_names
        assert (
            len(selected_model_names) > 0
        ), f"hyperparameters['model.names'] {hyperparameters[model_names_key]} doesn't match any config model names {config_model_names}."

        # Filter models that are not in hyperparameters[model_names_key]
        # Avoid key not in config error when applying the overrides later.
        model_keys = [k for k in hyperparameters.keys() if k.startswith(MODEL)]
        if model_keys and model_names_key in model_keys:
            model_keys.remove(model_names_key)
            for k in model_keys:
                if k.split(".")[1] not in hyperparameters[model_names_key] and k not in keys_to_filter:
                    keys_to_filter.append(k)

        # Filter models whose data types are not detected.
        # Avoid sampling unused checkpoints, e.g., hf_text models for image classification, to run jobs,
        # which wastes resources and time.
        detected_data_types = get_detected_data_types(column_types)
        selected_model_names = []
        for model_name in hyperparameters[model_names_key]:
            model_config = config.model[model_name]
            if model_config.data_types:
                model_data_status = [d_type in detected_data_types for d_type in model_config.data_types]
                if not all(model_data_status):
                    keys_to_filter.append(f"{MODEL}.{model_name}")
                else:
                    selected_model_names.append(model_name)
            else:  # keep the fusion model, which will be handled by select_model().
                selected_model_names.append(model_name)
        hyperparameters[model_names_key] = selected_model_names
        assert (
            len(selected_model_names) > 0
        ), f"Model {hyperparameters[model_names_key]} can't handle the data with column types {column_types}"

    # Filter keys for continuous training.
    # Model and data processors would be reused.
    if fit_called:
        warnings.warn(
            "HPO while continuous training."
            "Hyperparameters related to Model and Data will NOT take effect."
            "We will filter them out from the search space."
        )
        keys_to_filter.extend([MODEL, DATA])

    for key in keys_to_filter:
        hyperparameters = {k: v for k, v in hyperparameters.items() if not k.startswith(key)}

    return hyperparameters


def split_hyperparameters(hyperparameters: Dict):
    """
    Split out some advanced hyperparameters whose values are complex objects instead of strings or numbers.

    Parameters
    ----------
    hyperparameters
        The user provided hyperparameters.

    Returns
    -------
    Hyperparameters and advanced hyperparameters.
    """
    if not isinstance(hyperparameters, dict):  # only support complex objects in dict.
        return hyperparameters, dict()

    if not hyperparameters:
        return hyperparameters, dict()

    advanced_hyperparameters = dict()
    for k, v in hyperparameters.items():
        if re.search("^model.*train_transforms$", k) or re.search("^model.*val_transforms$", k):
            if all([isinstance(trans, str) for trans in hyperparameters[k]]):
                pass
            elif all([isinstance(trans, Callable) for trans in hyperparameters[k]]):
                advanced_hyperparameters[k] = copy.deepcopy(v)
                hyperparameters[k] = str(v)  # get the objects' class strings
            else:
                raise ValueError(f"transform_types {v} contain neither all strings nor all callable objects.")

    return hyperparameters, advanced_hyperparameters
