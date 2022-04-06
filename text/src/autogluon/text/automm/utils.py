import pytz
import datetime
import os
import functools
import logging
import pandas as pd
import pickle
import collections
import copy
import torch
from torch.nn.modules.container import ModuleList
import warnings
from contextlib import contextmanager
from typing import Optional, List, Any, Dict, Tuple, Union
from nptyping import NDArray
from omegaconf import OmegaConf, DictConfig
from autogluon.core.metrics import get_metric

from .models import (
    HFAutoModelForTextPrediction,
    TimmAutoModelForImagePrediction,
    CLIPForImageText,
    CategoricalMLP,
    NumericalMLP,
    MultimodalFusionMLP,
)
from .data import (
    ImageProcessor,
    TextProcessor,
    CategoricalProcessor,
    NumericalProcessor,
    LabelProcessor,
    MultiModalFeaturePreprocessor,
)
from .constants import (
    ACCURACY, RMSE, R2, ALL_MODALITIES,
    IMAGE, TEXT, CATEGORICAL, NUMERICAL,
    LABEL, MULTICLASS, BINARY, REGRESSION,
    Y_PRED_PROB, Y_PRED, Y_TRUE, AUTOMM,
    CLIP, TIMM_IMAGE, HF_TEXT, NUMERICAL_MLP,
    CATEGORICAL_MLP, FUSION_MLP,
)
from .presets import (
    list_model_presets,
    get_preset,
)

logger = logging.getLogger(AUTOMM)


def infer_metrics(
        problem_type: Optional[str] = None,
        eval_metric_name: Optional[str] = None,
):
    """
    Infer the validation metric and the evaluation metric if not provided.
    Validation metric is for early-stopping and selecting the best model checkpoints.
    Evaluation metric is to report performance to users.
    If the evaluation metric is provided, then we use it as the validation metric.
    But there are some exceptions that validation metric is different from evaluation metric.
    For example, if the provided evaluation metric is `r2`, we set the validation metric as `rmse`
    since `torchmetrics.R2Score` may encounter errors for per gpu batch size 1.

    Parameters
    ----------
    problem_type
        Type of problem.
    eval_metric_name
        Name of evaluation metric provided by users.

    Returns
    -------
    validation_metric_name
        Name of validation metric.
    eval_metric_name
        Name of evaluation metric.
    """
    if eval_metric_name is not None:
        if eval_metric_name.lower() == R2:
            validation_metric_name = RMSE
        else:
            validation_metric_name = eval_metric_name
        return validation_metric_name, eval_metric_name

    if problem_type in [MULTICLASS, BINARY]:
        eval_metric_name = ACCURACY
    elif problem_type == REGRESSION:
        eval_metric_name = RMSE
    else:
        raise NotImplementedError(
            f"Problem type: {problem_type} is not supported yet!"
        )

    validation_metric_name = eval_metric_name

    return validation_metric_name, eval_metric_name


def get_config(
        config: Union[dict, DictConfig],
        overrides: Optional[Union[str, List[str], Dict]] = None,
):
    """
    Construct configurations for model, data, optimization, and environment.
    It supports to overrides some default configurations.

    Parameters
    ----------
    config
        A dictionary including four keys: "model", "data", "optimization", and "environment".
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

    Returns
    -------
    Configurations as a DictConfig object
    """
    if config is None:
        config = get_preset(list_model_presets()[0])
    if not isinstance(config, DictConfig):
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
                    cur_path = os.path.dirname(os.path.abspath(__file__))
                    config_path = os.path.join(cur_path, "configs", k, f"{v}.yaml")
                    per_config = OmegaConf.load(config_path)
            else:
                raise ValueError(f"Unknown configuration type: {type(v)}")

            all_configs.append(per_config)

        config = OmegaConf.merge(*all_configs)
    verify_config_names(config.model)
    logger.debug(f"overrides: {overrides}")
    if overrides is not None:
        # avoid manipulating user-provided overrides
        overrides = copy.deepcopy(overrides)
        # apply customized model names
        overrides = parse_dotlist_conf(overrides)  # convert to a dict
        config.model = customize_config_names(
            config=config.model,
            customized_names=overrides.get("model.names", None),
        )
        # remove `model.names` from overrides since it's already applied.
        overrides.pop("model.names", None)
        # apply all the overrides
        config = apply_omegaconf_overrides(config, overrides=overrides, check_key_exist=True)
    verify_config_names(config.model)
    return config


def verify_config_names(config: DictConfig):
    """
    Verify whether provided names are valid for a config.

    Parameters
    ----------
    config
        Config should have a attribute `names`, which contains a list of
        attribute names, e.g., ["timm_image", "hf_text"]. And each string in
        `config.names` should also be a attribute of `config`, e.g, `config.timm_image`.
    """
    # must have attribute `names`
    assert hasattr(config, "names")
    # assure no duplicate names
    assert len(config.names) == len(set(config.names))
    # verify that strings in `config.names` match the keys of `config`.
    keys = list(config.keys())
    keys.remove("names")
    assert sorted(config.names) == sorted(keys), \
        f"`{config.names}` do not match config keys {keys}"

    # verify that no name starts with another one
    names = sorted(config.names, key=lambda ele: len(ele), reverse=True)
    for i in range(len(names)):
        if names[i].startswith(tuple(names[i+1:])):
            raise ValueError(
                f"name {names[i]} starts with one of another name: {names[i+1:]}"
            )


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


def customize_config_names(
        config: DictConfig,
        customized_names: List[str],
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
            logger.debug(
                f"Removing {per_name}, which doesn't start with any of these prefixes: {available_prefixes}."
            )

    if len(new_config.names) == 0:
        raise ValueError(
            f"No customized name in `{customized_names}` starts with name prefixes in `{available_prefixes}`."
        )

    return new_config


def select_model(
        config: DictConfig,
        df_preprocessor: MultiModalFeaturePreprocessor,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the "fusion_mlp_image_text_tabular" as the model config template
    but the training data don't have images, this function will filter out all the models
    using images, such as Swin Transformer and CLIP.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_path_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        if model_config.data_types is None:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    assert len(fusion_model_name) <= 1
    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    else:  # remove the fusion model's config make `config.model.names` and the keys of `config.model` consistent.
        if len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
            delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")

    return config


def init_df_preprocessor(
        config: DictConfig,
        column_types: collections.OrderedDict,
        label_column: str,
        train_df_x: pd.DataFrame,
        train_df_y: pd.Series,
):
    """
    Initialize the dataframe preprocessor by calling .fit().

    Parameters
    ----------
    config
        A DictConfig containing only the data config.
    column_types
        A dictionary that maps column names to their data types.
        For example: `column_types = {"item_name": "text", "image": "image_path",
        "product_description": "text", "height": "numerical"}`
        may be used for a table with columns: "item_name", "brand", "product_description", and "height".
    label_column
        Name of the column that contains the target variable to predict.
    train_df_x
        A pd.DataFrame containing only the feature columns.
    train_df_y
        A pd.Series object containing only the label column.
    Returns
    -------
    Initialized dataframe preprocessor.
    """
    df_preprocessor = MultiModalFeaturePreprocessor(
        config=config,
        column_types=column_types,
        label_column=label_column,
    )
    df_preprocessor.fit(
        X=train_df_x,
        y=train_df_y,
    )

    return df_preprocessor


def init_data_processors(
        config: DictConfig,
        num_categorical_columns: int,
):
    """
    Create the data processors according to the model config. This function creates one processor for
    each modality of each model. For example, if one model config contains BERT, ViT, and CLIP, then
    BERT would have its own text processor, ViT would have its own image processor, and CLIP would have
    its own text and image processors. This is to support training arbitrary combinations of single-modal
    and multimodal models since two models may share the same modality but have different processing. Text
    sequence length is a good example. BERT's sequence length is generally 512, while CLIP uses sequences of
    length 77.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_categorical_columns
        The number of categorical columns in the training dataframe.

    Returns
    -------
    A dictionary with modalities as the keys. Each modality has a list of processors.
    Note that "label" is also treated as a modality for convenience.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]

    data_processors = {
        IMAGE: [],
        TEXT: [],
        CATEGORICAL: [],
        NUMERICAL: [],
        LABEL: [],
    }
    for model_name in names:
        model_config = getattr(config.model, model_name)
        # each model has its own label processor
        data_processors[LABEL].append(
            LabelProcessor(prefix=model_name)
        )
        if model_config.data_types is None:
            continue
        for d_type in model_config.data_types:
            if d_type == IMAGE:
                data_processors[IMAGE].append(
                    ImageProcessor(
                        prefix=model_name,
                        checkpoint_name=model_config.checkpoint_name,
                        train_transform_types=model_config.train_transform_types,
                        val_transform_types=model_config.val_transform_types,
                        norm_type=model_config.image_norm,
                        size=model_config.image_size,
                        max_img_num_per_col=model_config.max_img_num_per_col,
                        missing_value_strategy=config.data.image.missing_value_strategy,
                    )
                )
            elif d_type == TEXT:
                data_processors[TEXT].append(
                    TextProcessor(
                        prefix=model_name,
                        tokenizer_name=model_config.tokenizer_name,
                        checkpoint_name=model_config.checkpoint_name,
                        max_len=model_config.max_text_len,
                        insert_sep=model_config.insert_sep,
                        text_segment_num=model_config.text_segment_num,
                        stochastic_chunk=model_config.stochastic_chunk,
                    )
                )
            elif d_type == CATEGORICAL:
                data_processors[CATEGORICAL].append(
                    CategoricalProcessor(
                        prefix=model_name,
                        num_categorical_columns=num_categorical_columns,
                    )
                )
            elif d_type == NUMERICAL:
                data_processors[NUMERICAL].append(
                    NumericalProcessor(
                        prefix=model_name,
                        merge=model_config.merge,
                    )
                )
            else:
                raise ValueError(f"unknown data type: {d_type}")

    assert len(data_processors[LABEL]) > 0

    # Only keep the modalities with non-empty processors.
    data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}
    return data_processors


def create_model(
        config: DictConfig,
        num_classes: int,
        num_numerical_columns: Optional[int] = None,
        num_categories: Optional[List[int]] = None,
        pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    all_models = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        if model_name.lower().startswith(CLIP):
            model = CLIPForImageText(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(TIMM_IMAGE):
            model = TimmAutoModelForImagePrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
                mix_choice=model_config.mix_choice,
                pretrained=pretrained,
            )
        elif model_name.lower().startswith(HF_TEXT):
            model = HFAutoModelForTextPrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(NUMERICAL_MLP):
            model = NumericalMLP(
                prefix=model_name,
                in_features=num_numerical_columns,
                hidden_features=model_config.hidden_size,
                out_features=model_config.hidden_size,
                num_layers=model_config.num_layers,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(CATEGORICAL_MLP):
            model = CategoricalMLP(
                prefix=model_name,
                num_categories=num_categories,
                out_features=model_config.hidden_size,
                num_layers=model_config.num_layers,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                num_classes=num_classes,
            )
        elif model_name.lower().startswith(FUSION_MLP):
            fusion_model = functools.partial(
                MultimodalFusionMLP,
                prefix=model_name,
                hidden_features=model_config.hidden_sizes,
                num_classes=num_classes,
                adapt_in_features=model_config.adapt_in_features,
                activation=model_config.activation,
                dropout_prob=model_config.drop_rate,
                normalization=model_config.normalization,
                loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
            )
            continue
        else:
            raise ValueError(f"unknown model name: {model_name}")

        all_models.append(model)

    if len(all_models) > 1:
        # must have one fusion model if there are multiple independent models
        return fusion_model(models=all_models)
    elif len(all_models) == 1:
        return all_models[0]
    else:
        raise ValueError(f"No available models for {names}")


def save_pretrained_configs(
    model: ModuleList,
    config: DictConfig,
    path: str
) -> DictConfig:
    '''
    Saving the pretrained configs for offline deployment. 
    It is called by setting "standalone=True" in "AutoMMPredictor.load()". 

    Parameters
    ----------
    model
        A modulelist containing all models.
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The saving path to the pretained weights i.e. config.json for "clip" and "hf_text"
    '''
    for idx, model_name in enumerate(config.model.names):
        if model_name.lower().startswith((CLIP, HF_TEXT)):
            model[idx].model.save_pretrained(os.path.join(path, model_name))
            model_config = getattr(config.model, model_name)
            model_config.checkpoint_name = os.path.join('local://', model_name)
    return config


def convert_checkpoint_name(
        config: DictConfig,
        path: str
) -> DictConfig:  
    '''
    Convert the checkpoint name from relative path to absolute path for loading the pretrained weights in offline deployment. 
    It is called by setting "standalone=True" in "AutoMMPredictor.load()". 

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    path
        The saving path to the pretained weights i.e. config.json for "clip" and "hf_text", 'pkl' for "timm_image"
    '''
    for model_name in config.model.names:
        if model_name.lower().startswith((CLIP, HF_TEXT)):
            model_config = getattr(config.model, model_name)
            if model_config.checkpoint_name.startswith('local://'):
                model_config.checkpoint_name = os.path.join(path, model_config.checkpoint_name[len('local://'):])
                assert os.path.exists(os.path.join(model_config.checkpoint_name, 'config.json')) # guarantee the existence of local configs
                assert os.path.exists(os.path.join(model_config.checkpoint_name, 'pytorch_model.bin'))
                
    return config    


def make_exp_dir(
        root_path: str,
        job_name: str,
        create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.,: root_path/2022_01_01/job_name_12_00_00/
    This function is to better organize the training runs. It is recommended to call this
    function and pass the returned "exp_dir" to "AutoMMPredictor.fit(save_path=exp_dir)".

    Parameters
    ----------
    root_path
        The basic path where to create saving directories for training runs.
    job_name
        The job names to name training runs.
    create
        Whether to make the directory.

    Returns
    -------
    The formatted directory path.
    """
    tz = pytz.timezone('US/Pacific')
    ct = datetime.datetime.now(tz=tz)
    date_stamp = ct.strftime("%Y_%m_%d")
    time_stamp = ct.strftime("%H_%M_%S")

    # Group logs by day first
    exp_dir = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    exp_dir = os.path.join(exp_dir, "_".join([job_name, time_stamp]))

    if create:
        os.makedirs(exp_dir, mode=0o777, exist_ok=False)

    return exp_dir


def gather_top_k_ckpts(
        ckpt_dir: Optional[str] = None,
        ckpt_paths: Optional[List[str]] = None,
):
    """
    Gather the state_dicts of top k models. If "ckpt_paths" is not an empty list, it loads the models
    from its paths. Otherwise, it will find available checkpoints in the "ckpt_dir". After loading all the
    top k checkpoints, it cleans them. The lastest checkpoint "last.ckpt" is also removed since "last.ckpt"
    is for resuming training in the middle, but the the training should be done when calling this function.

    Parameters
    ----------
    ckpt_dir
        The directory where we save all the top k checkpoints.
    ckpt_paths
        A list of top k checkpoint paths.

    Returns
    -------
    all_state_dicts
        A list of state_dicts
    """
    if not ckpt_paths:
        ckpt_paths = []
        for file_name in sorted(os.listdir(ckpt_dir), reverse=True):
            if file_name.startswith("epoch"):
                ckpt_paths.append(os.path.join(ckpt_dir, file_name))

    all_state_dicts = []
    for path in ckpt_paths:
        checkpoint = torch.load(path)
        all_state_dicts.append(checkpoint["state_dict"])
        os.remove(path)

    if ckpt_dir is not None:
        for file_name in os.listdir(ckpt_dir):
            if file_name.startswith("epoch"):
                os.remove(os.path.join(ckpt_dir, file_name))
        last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            os.remove(last_ckpt_path)

    logger.debug(f"ckpt num: {len(all_state_dicts)}")
    return all_state_dicts


def average_checkpoints(
        all_state_dicts: List[Dict],
        out_path: Optional[str] = None,
):
    """
    Average the state_dicts of top k checkpoints.

    Parameters
    ----------
    all_state_dicts
        A list of Pytorch state_dicts.
    out_path
        The path to save the averaged checkpoint.

    Returns
    -------
    The averaged state_dict.
    """
    avg_state_dict = {}
    for key in all_state_dicts[0]:
        arr = [state_dict[key] for state_dict in all_state_dicts]
        avg_state_dict[key] = sum(arr) / len(arr)

    if out_path:
        checkpoint = {"state_dict": avg_state_dict}
        torch.save(checkpoint, out_path)

    return avg_state_dict


def compute_score(
        metric_data: dict,
        metric_name: str,
) -> float:
    """
    Use sklearn to compute the score of one metric.

    Parameters
    ----------
    metric_data
        A dictionary with the groundtruth (Y_TRUE) and predicted values (Y_PRED, Y_PRED_PROB).
        The predicted class probabilities are required to compute the roc_auc score.
    metric_name
        The name of metric to compute.

    Returns
    -------
    Computed score.
    """
    metric = get_metric(metric_name)
    if metric.name in ["roc_auc", "average_precision"]:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED_PROB][:, 1])
    else:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED])


def parse_dotlist_conf(conf):
    """Parse the config files that is potentially in the dotlist format to a dictionary

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
        raise ValueError(f'Unsupported format of conf={conf}')
    if need_parse:
        new_conf = dict()
        curr_key = None
        curr_value = ''
        for ele in conf:
            if '=' in ele:
                key, v = ele.split('=')
                if curr_key is not None:
                    new_conf[curr_key] = curr_value
                curr_key = key
                curr_value = v
            else:
                if curr_key is None:
                    raise ValueError(f'Cannot parse the conf={conf}')
                curr_value = curr_value + ' ' + ele
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
            key_in_dotlist = key_in_dotlist.split('.')
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
                raise KeyError(f'"{ele[0]}" is not found in the config. You may need to check the overrides. '
                               f'overrides={overrides}')
    override_conf = OmegaConf.from_dotlist([f'{ele[0]}={ele[1]}' for ele in overrides.items()])
    conf = OmegaConf.merge(conf, override_conf)
    return conf


class LogFilter(logging.Filter):
    """
    Filter log messages with patterns.
    """

    def __init__(self, blacklist: Union[str, List[str]]):
        """
        Parameters
        ----------
        blacklist
            Patterns to be suppressed in logging.
        """
        super().__init__()
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        self._blacklist = blacklist

    def filter(self, record):
        """
        Check whether to suppress a logging message.

        Parameters
        ----------
        record
            A logging message.

        Returns
        -------
        If True, no pattern exists in the message, hence printed out.
        If False, some pattern is in the message, hence filtered out.
        """
        matches = [pattern not in record.msg for pattern in self._blacklist]
        return all(matches)


def add_log_filter(target_logger, log_filter):
    """
    Add one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.addFilter(log_filter)


def remove_log_filter(target_logger, log_filter):
    """
    Remove one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.removeFilter(log_filter)


@contextmanager
def apply_log_filter(log_filter):
    """
    User contextmanager to control the scope of applying one log filter.
    Currently, it is to filter some pytorch lightning's log messages.
    But we can easily extend it to cover more loggers.

    Parameters
    ----------
    log_filter
        Log filter.
    """
    try:
        add_log_filter(logging.getLogger(), log_filter)
        add_log_filter(logging.getLogger("pytorch_lightning"), log_filter)
        yield

    finally:
        remove_log_filter(logging.getLogger(), log_filter)
        remove_log_filter(logging.getLogger("pytorch_lightning"), log_filter)
