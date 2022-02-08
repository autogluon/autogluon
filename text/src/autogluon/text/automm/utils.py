import pytz
import datetime
import os
import functools
import numpy as np
import pandas as pd
import pickle
import collections
import torch
import warnings
from typing import Optional, List, Any, Dict, Tuple, Union
from nptyping import NDArray
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    cohen_kappa_score,
    roc_auc_score,
)
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
    ACCURACY, RMSE, ALL_MODALITIES,
    IMAGE, TEXT, CATEGORICAL, NUMERICAL,
    LABEL, MULTICLASS, BINARY, REGRESSION,
    Y_PRED_PROB, Y_PRED, Y_TRUE,
)


def infer_eval_metric(problem_type: str):

    if problem_type in [MULTICLASS, BINARY]:
        eval_metric = ACCURACY
    elif problem_type == REGRESSION:
        eval_metric = RMSE
    else:
        raise NotImplementedError(
            f"Problem type: {problem_type} is not supported yet!"
        )

    return eval_metric


def get_config(
        config: dict,
        overrides: Optional[List[str]] = None,
):
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
    print('overrides=', overrides)
    if overrides is not None:
        config = apply_omegaconf_overrides(config, overrides=overrides, check_key_exist=True)

    return config


def select_model(
        config: DictConfig,
        df_preprocessor: MultiModalFeaturePreprocessor,
):
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

    # only allow no more than 1 fusion model
    assert len(fusion_model_name) <= 1
    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)

    config.model.names = selected_model_names
    print(f"selected models: {selected_model_names}")
    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    return config


def init_df_preprocessor(
        config: DictConfig,
        column_types: collections.OrderedDict,
        label_column: str,
        train_df_x: pd.DataFrame,
        train_df_y: pd.Series,
):

    df_preprocessor = MultiModalFeaturePreprocessor(
        cfg=config,
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
    names = config.model.names
    if isinstance(names, str):
        names = [names]

    image_processors = []
    text_processors = []
    categorical_processors = []
    numerical_processors = []
    label_processors = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        # each model has its own label processor
        label_processors.append(
            LabelProcessor(prefix=model_name)
        )
        if model_config.data_types is None:
            continue
        for d_type in model_config.data_types:
            if d_type == IMAGE:
                image_processors.append(
                    ImageProcessor(prefix=model_name,
                                   checkpoint_name=model_config.checkpoint_name,
                                   train_transform_types=model_config.train_transform_types,
                                   val_transform_types=model_config.val_transform_types,
                                   norm_type=model_config.image_norm,
                                   size=model_config.image_size,
                                   max_img_num_per_col=model_config.max_img_num_per_col)
                )
            elif d_type == TEXT:
                text_processors.append(
                    TextProcessor(prefix=model_name,
                                  tokenizer_name=model_config.tokenizer_name,
                                  checkpoint_name=model_config.checkpoint_name,
                                  max_len=model_config.max_text_len,
                                  insert_sep=model_config.insert_sep,
                                  text_segment_num=model_config.text_segment_num,
                                  stochastic_chunk=model_config.stochastic_chunk)
                )
            elif d_type == CATEGORICAL:
                categorical_processors.append(
                    CategoricalProcessor(prefix=model_name,
                                         num_categorical_columns=num_categorical_columns)
                )
            elif d_type == NUMERICAL:
                numerical_processors.append(
                    NumericalProcessor(prefix=model_name,
                                       merge=model_config.merge)
                )
            else:
                raise ValueError(f"unknown data type: {d_type}")

    assert len(label_processors) > 0

    return {
        IMAGE: image_processors,
        TEXT: text_processors,
        CATEGORICAL: categorical_processors,
        NUMERICAL: numerical_processors,
        LABEL: label_processors
    }


def create_model(
        config: DictConfig,
        num_classes: int,
        num_numerical_columns: Optional[int] = None,
        num_categories: Optional[List[int]] = None,
):
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    print(f"output_shape: {num_classes}")
    all_models = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        if model_name == "clip":
            model = CLIPForImageText(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name == "timm_image":
            model = TimmAutoModelForImagePrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
                mix_choice=model_config.mix_choice,
            )
        elif "hf_text" in model_name:
            model = HFAutoModelForTextPrediction(
                prefix=model_name,
                checkpoint_name=model_config.checkpoint_name,
                num_classes=num_classes,
            )
        elif model_name == "numerical_mlp":
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
        elif model_name == "categorical_mlp":
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
        elif model_name == "fusion_mlp":
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


def setup_save_dir(
        path: Optional[str] = None,
        job_name: Optional[str] = "job",
        warn_if_exist=True,
):
    if path is None:
        path = os.path.join("~", "exp")
        path = os.path.expanduser(path)
        path = make_exp_dir(
            root_path=path,
            job_name=job_name,
        )
    else:
        if os.path.isdir(path) and warn_if_exist:
            warnings.warn(
                f'Warning: path already exists! '
                f'This predictor may overwrite an existing predictor! path="{path}"'
            )
        os.makedirs(path, mode=0o777, exist_ok=True)

    return path


def make_exp_dir(
        root_path: str,
        job_name: str,
        create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.:
        experiments/2017_01_01/job_name_12_00_00/
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

    if ckpt_paths is None:
        ckpt_paths = []
        for file_name in os.listdir(ckpt_dir):
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

    print(f"ckpt num: {len(all_state_dicts)}")
    return all_state_dicts, checkpoint


def average_checkpoints(
        all_state_dicts: List[Dict],
        out_path: str,
        ckpt_template: dict,
):

    avg_state_dict = dict()
    for key in all_state_dicts[0]:
        arr = [state_dict[key] for state_dict in all_state_dicts]
        avg_state_dict[key] = sum(arr) / len(arr)

    ckpt_template["state_dict"] = avg_state_dict
    torch.save(ckpt_template, out_path)

    return avg_state_dict


def compute_score(
        metric_data: dict,
        metric_name: str,
) -> float:
    metric_name = metric_name.lower()
    if metric_name in ["acc", "accuracy"]:
        score = accuracy_score(
            y_true=metric_data[Y_TRUE],
            y_pred=metric_data[Y_PRED],
        )
    elif metric_name == "quadratic_kappa":
        score = cohen_kappa_score(
            metric_data[Y_PRED],
            metric_data[Y_TRUE],
            weights="quadratic",
        )
    elif metric_name == "r2":
        score = r2_score(
            y_true=metric_data[Y_TRUE],
            y_pred=metric_data[Y_PRED],
        )
    elif metric_name == "roc_auc":  # this is only for binary classification
        score = roc_auc_score(
            y_true=metric_data[Y_TRUE],
            y_score=metric_data[Y_PRED_PROB][:, 1],
        )
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")

    return score


def apply_omegaconf_overrides(
        conf: Union[List, Tuple, str, Dict],
        overrides,
        check_key_exist=True,
):
    """Apply omegaconf overrides

    Parameters
    ----------
    conf
        The base configuration
    overrides
        The overrides can be a string or a list
    check_key_exist
        Whether to check if all keys in the overrides must exist in the conf

    Returns
    -------
    new_conf
        The updated configuration
    """
    if isinstance(overrides, str):
        overrides = overrides.split()
        need_parse_overrides = True
    elif isinstance(overrides, (list, tuple)):
        need_parse_overrides = True
    elif isinstance(overrides, dict):
        need_parse_overrides = False
    else:
        raise ValueError(f'Unsupported format of overrides. Overrides={overrides}')

    if need_parse_overrides:
        kv_l = []
        curr_key = None
        curr_value = ''
        for ele in overrides:
            if '=' in ele:
                key, v = ele.split('=')
                if curr_key is not None:
                    kv_l.append((curr_key, curr_value))
                curr_key = key
                curr_value = v
            else:
                if curr_key is None:
                    raise ValueError(f'Cannot parse the overrides. overrides={overrides}')
                curr_value = curr_value + ' ' + ele
        if curr_key is not None:
            kv_l.append((curr_key, curr_value))
    else:
        kv_l = sorted(list(overrides.items()))
    if check_key_exist:
        for ele in kv_l:
            OmegaConf.select(conf, ele[0], throw_on_missing=True)
    override_conf = OmegaConf.from_dotlist([f'{ele[0]}={ele[1]}' for ele in kv_l])
    conf = OmegaConf.merge(conf, override_conf)
    return conf
