import copy
import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import (
    AUTOMM,
    CATEGORICAL_TRANSFORMER,
    FUSION_TRANSFORMER,
    HF_MODELS,
    NUMERICAL_TRANSFORMER,
    REGRESSION,
    MODEL,
    DATA,
)
from ..models import TimmAutoModelForImagePrediction
from ..presets import get_automm_presets, get_basic_automm_config
from .data import get_detected_data_types

logger = logging.getLogger(AUTOMM)


def get_default_config(config, extra: Optional[List[str]] = None):

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
    config: Optional[Union[dict, DictConfig]] = None,
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
        config.model = customize_model_names(
            config=config.model,
            customized_names=overrides.get("model.names", None),
        )
        # remove `model.names` from overrides since it's already applied.
        overrides.pop("model.names", None)
        # apply the user-provided overrides
        # raise ValueError(f"config: {OmegaConf.to_yaml(config)} \n overrides: {overrides}")
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
        NUMERICAL_TRANSFORMER: num_numerical_columns,
        CATEGORICAL_TRANSFORMER: num_categorical_columns,
        FUSION_TRANSFORMER: num_categorical_columns + num_numerical_columns,
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


def _filter_timm_pretrained_cfg(cfg, remove_source=False, remove_null=True):
    filtered_cfg = {}
    keep_null = {"pool_size", "first_conv", "classifier"}  # always keep these keys, even if none
    for k, v in cfg.items():
        if remove_source and k in {"url", "file", "hf_hub_id", "hf_hub_id", "hf_hub_filename", "source"}:
            continue
        if remove_null and v is None and k not in keep_null:
            continue
        filtered_cfg[k] = v
    return filtered_cfg


def save_timm_config(
    model: TimmAutoModelForImagePrediction,
    config_path: str,
):
    """
    Save TIMM image model configs to a local file.

    Parameters
    ----------
    model
        A TimmAutoModelForImagePrediction model object.
    config_path:
        A file to where the config is written to.
    """
    config = {}
    pretrained_cfg = _filter_timm_pretrained_cfg(model.config, remove_source=True, remove_null=True)
    # set some values at root config level
    config["architecture"] = pretrained_cfg.pop("architecture")
    config["num_classes"] = model.num_classes
    config["num_features"] = model.out_features

    global_pool_type = getattr(model, "global_pool", None)
    if isinstance(global_pool_type, str) and global_pool_type:
        config["global_pool"] = global_pool_type

    config["pretrained_cfg"] = pretrained_cfg

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        logger.info(f"Timm config saved to {config_path}.")


def update_hyperparameters(
    problem_type,
    presets,
    provided_hyperparameters,
    provided_hyperparameter_tune_kwargs,
    teacher_predictor: Optional[str] = None,
):
    hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(problem_type=problem_type, presets=presets)

    if hyperparameter_tune_kwargs and provided_hyperparameter_tune_kwargs:
        hyperparameter_tune_kwargs.update(provided_hyperparameter_tune_kwargs)
    elif provided_hyperparameter_tune_kwargs:
        hyperparameter_tune_kwargs = provided_hyperparameter_tune_kwargs

    if hyperparameter_tune_kwargs:
        if provided_hyperparameters:
            hyperparameters.update(provided_hyperparameters)
    else:
        hyperparameters = provided_hyperparameters

    if hyperparameter_tune_kwargs:
        assert isinstance(
            hyperparameters, dict
        ), "Please provide hyperparameters as a dictionary if you want to do HPO"
        if teacher_predictor is not None:
            assert isinstance(
                teacher_predictor, str
            ), "HPO with distillation only supports passing a path to the predictor"

    return hyperparameters, hyperparameter_tune_kwargs


def filter_hyperparameters(hyperparameters: Dict, column_types: Dict, config: Union[Dict, DictConfig], fit_called: bool):
    model_names_key = f"{MODEL}.names"
    keys_to_filter = []
    # Filter models that are not in model.names
    # Avoid key not in config error.
    if model_names_key in hyperparameters:
        model_keys = [k for k in hyperparameters.keys() if k.startswith(MODEL)]
        model_keys.remove(model_names_key)
        for k in model_keys:
            if k.split(".")[1] not in hyperparameters[model_names_key] and k not in keys_to_filter:
                keys_to_filter.append(k)

    config = get_default_config(config)
    valid_names = []
    for model_name in hyperparameters[model_names_key]:
        if hasattr(config.model, model_name):
            valid_names.append(model_name)
    hyperparameters[model_names_key] = valid_names

    # Filter models whose data types are not detected.
    # Avoid sampling unused checkpoints to run jobs, which wastes resources and time.
    detected_data_types = get_detected_data_types(column_types)
    for model_name in hyperparameters[model_names_key]:
        model_config = config.model[model_name]
        if model_config.data_types:  # skip fusion model
            model_data_status = [d_type in detected_data_types for d_type in model_config.data_types]
            if not all(model_data_status):
                keys_to_filter.append(f"{MODEL}.{model_name}")

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
        hyperparameters = {k:v for k, v in hyperparameters.items() if not k.startswith(key)}

    return hyperparameters
