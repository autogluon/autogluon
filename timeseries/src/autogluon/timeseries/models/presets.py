import copy
import logging
import re
from typing import Any, Dict, List, Optional, Type, Union

import autogluon.timeseries as agts
from autogluon.common import space
from autogluon.core import constants

from . import (
    ARIMAModel,
    AutoARIMAModel,
    AutoETSModel,
    DeepARModel,
    DirectTabularModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    NaiveModel,
    RecursiveTabularModel,
    SeasonalNaiveModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    ThetaModel,
    ThetaStatsmodelsModel,
)
from .abstract import AbstractTimeSeriesModel
from .multi_window.multi_window_model import MultiWindowBacktestingModel

logger = logging.getLogger(__name__)

# define the model zoo with their aliases
MODEL_TYPES = dict(
    SimpleFeedForward=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    TemporalFusionTransformer=TemporalFusionTransformerModel,
    RecursiveTabular=RecursiveTabularModel,
    DirectTabular=DirectTabularModel,
    Naive=NaiveModel,
    SeasonalNaive=SeasonalNaiveModel,
    AutoETS=AutoETSModel,
    AutoARIMA=AutoARIMAModel,
    DynamicOptimizedTheta=DynamicOptimizedThetaModel,
    Theta=ThetaModel,
    ARIMA=ARIMAModel,
    ETS=ETSModel,
    ThetaStatsmodels=ThetaStatsmodelsModel,
)
if agts.MXNET_INSTALLED:
    from .gluonts.mx import (
        DeepARMXNetModel,
        MQCNNMXNetModel,
        MQRNNMXNetModel,
        SimpleFeedForwardMXNetModel,
        TemporalFusionTransformerMXNetModel,
        TransformerMXNetModel,
    )

    MODEL_TYPES.update(
        dict(
            DeepARMXNet=DeepARMXNetModel,
            SimpleFeedForwardMXNet=SimpleFeedForwardMXNetModel,
            MQCNNMXNet=MQCNNMXNetModel,
            MQRNNMXNet=MQRNNMXNetModel,
            TransformerMXNet=TransformerMXNetModel,
            TemporalFusionTransformerMXNet=TemporalFusionTransformerMXNetModel,
        )
    )

DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    Naive=100,
    SeasonalNaive=100,
    ETS=90,
    Theta=90,
    RecursiveTabular=80,
    ARIMA=70,
    DirectTabular=70,
    DeepAR=60,
    TemporalFusionTransformer=50,
    SimpleFeedForward=40,
    AutoARIMA=50,
    AutoETS=70,
    DynamicOptimizedTheta=60,
    # Models below are not included in any presets
    DeepARMXNet=50,
    SimpleFeedForwardMXNet=30,
    TemporalFusionTransformerMXNet=50,
    TransformerMXNet=30,
    MQCNNMXNet=10,
    MQRNNMXNet=10,
)
DEFAULT_CUSTOM_MODEL_PRIORITY = 0

VALID_AG_ARGS_KEYS = {
    "name",
    "name_prefix",
    "name_suffix",
}


def get_default_hps(key):
    default_model_hps = {
        "local_only": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {"max_num_samples": 100_000},
        },
        "medium_quality": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "AutoETS": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "DeepAR": {},
        },
        "high_quality": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "AutoETS": {},
            "AutoARIMA": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "DeepAR": {},
            "SimpleFeedForward": {},
            "TemporalFusionTransformer": {},
        },
        "best_quality": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "AutoETS": {},
            "AutoARIMA": {},
            "DynamicOptimizedTheta": {},
            "Theta": {},
            "RecursiveTabular": {
                "tabular_hyperparameters": {
                    "NN_TORCH": {"proc.impute_strategy": "constant"},
                    "GBM": [{}, {"extra_trees": True, "ag_args": {"name_suffix": "XT"}}],
                },
            },
            "DirectTabular": {},
            "DeepAR": {
                "num_layers": space.Int(1, 3, default=2),
                "hidden_size": space.Int(40, 80, default=40),
            },
            "SimpleFeedForward": {
                "hidden_dimensions": space.Categorical([40], [40, 40], [120]),
            },
            "TemporalFusionTransformer": {},
        },
    }

    # For backwards compatibility
    default_model_hps["default"] = default_model_hps["medium_quality"]
    default_model_hps["default_hpo"] = default_model_hps["best_quality"]

    return default_model_hps[key]


def normalize_model_type_name(model_name: str) -> str:
    """Remove 'Model' suffix from the end of the string, if it's present."""
    if model_name.endswith("Model"):
        model_name = model_name[: -len("Model")]
    return model_name


def get_preset_models(
    freq: str,
    prediction_length: int,
    path: str,
    eval_metric: str,
    eval_metric_seasonal_period: Optional[int],
    hyperparameters: Union[str, Dict, None],
    hyperparameter_tune: bool,
    all_assigned_names: List[str],
    excluded_model_types: List[str],
    multi_window: bool = False,
    **kwargs,
):
    """
    Create a list of models according to hyperparameters. If hyperparamaters=None,
    will create models according to presets.
    """
    models = []
    if hyperparameters is None:
        hp_string = "default_hpo" if hyperparameter_tune else "default"
        hyperparameters = copy.deepcopy(get_default_hps(hp_string))
    elif isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(get_default_hps(hyperparameters))
    elif isinstance(hyperparameters, dict):
        hyperparameters = copy.deepcopy(hyperparameters)
    else:
        raise ValueError(
            f"hyperparameters must be a dict, a string or None (received {type(hyperparameters)}). "
            f"Please see the documentation for TimeSeriesPredictor.fit"
        )
    # Handle model names ending with "Model", e.g., "DeepARModel" is mapped to "DeepAR"
    hyperparameters_clean = {}
    for key, value in hyperparameters.items():
        if isinstance(key, str):
            key = normalize_model_type_name(key)
        hyperparameters_clean[key] = value
    hyperparameters = hyperparameters_clean

    excluded_models = set()
    if excluded_model_types is not None and len(excluded_model_types) > 0:
        if not isinstance(excluded_model_types, list):
            raise ValueError(f"`excluded_model_types` must be a list, received {type(excluded_model_types)}")
        logger.info(f"Excluded model types: {excluded_model_types}")
        for model in excluded_model_types:
            if not isinstance(model, str):
                raise ValueError(f"Each entry in `excluded_model_types` must be a string, received {type(model)}")
            excluded_models.add(normalize_model_type_name(model))

    if hyperparameter_tune:
        verify_contains_at_least_one_searchspace(hyperparameters)
    else:
        verify_contains_no_searchspaces(hyperparameters)

    all_assigned_names = set(all_assigned_names)

    model_priority_list = sorted(hyperparameters.keys(), key=lambda x: DEFAULT_MODEL_PRIORITY.get(x, 0), reverse=True)

    for model in model_priority_list:
        if isinstance(model, str):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            if model in excluded_models:
                logger.info(
                    f"\tFound '{model}' model in hyperparameters, but '{model}' "
                    "is present in `excluded_model_types` and will be removed."
                )
                continue
            model_type = MODEL_TYPES[model]
        elif isinstance(model, type):
            if not issubclass(model, AbstractTimeSeriesModel):
                raise ValueError(f"Custom model type {model} must inherit from `AbstractTimeSeriesModel`.")
            model_type = model
        else:
            raise ValueError(
                f"Keys of the `hyperparameters` dictionary must be strings or types, received {type(model)}."
            )

        model_hps_list = hyperparameters[model]
        if not isinstance(model_hps_list, list):
            model_hps_list = [model_hps_list]

        for model_hps in model_hps_list:
            ag_args = model_hps.pop(constants.AG_ARGS, {})
            for key in ag_args:
                if key not in VALID_AG_ARGS_KEYS:
                    raise ValueError(
                        f"Model {model_type} received unknown ag_args key: {key} (valid keys {VALID_AG_ARGS_KEYS})"
                    )
            model_name_base = get_model_name(ag_args, model_type)

            model_type_kwargs = dict(
                name=model_name_base,
                path=path,
                freq=freq,
                prediction_length=prediction_length,
                eval_metric=eval_metric,
                eval_metric_seasonal_period=eval_metric_seasonal_period,
                hyperparameters=model_hps,
                **kwargs,
            )

            # add models while preventing name collisions
            model = model_type(**model_type_kwargs)

            model_type_kwargs.pop("name", None)
            increment = 1
            while model.name in all_assigned_names:
                increment += 1
                model = model_type(name=f"{model_name_base}_{increment}", **model_type_kwargs)

            if multi_window:
                model = MultiWindowBacktestingModel(model_base=model, name=model.name, **model_type_kwargs)

            all_assigned_names.add(model.name)
            models.append(model)

    return models


def get_model_name(ag_args: Dict[str, Any], model_type: Type[AbstractTimeSeriesModel]) -> str:
    name = ag_args.get("name")
    if name is None:
        name_stem = re.sub(r"Model$", "", model_type.__name__)
        name_prefix = ag_args.get("name_prefix", "")
        name_suffix = ag_args.get("name_suffix", "")
        name = name_prefix + name_stem + name_suffix
    return name


def contains_searchspace(model_hyperparameters: Dict[str, Any]) -> bool:
    for hp_value in model_hyperparameters.values():
        if isinstance(hp_value, space.Space):
            return True
    return False


def verify_contains_at_least_one_searchspace(hyperparameters: Dict[str, Dict[str, Any]]):
    for model, model_hps_list in hyperparameters.items():
        if not isinstance(model_hps_list, list):
            model_hps_list = [model_hps_list]

        for model_hps in model_hps_list:
            if contains_searchspace(model_hps):
                return

    raise ValueError(
        f"Hyperparameter tuning specified, but no model contains a hyperparameter search space. "
        f"Please disable hyperparameter tuning with `hyperparameter_tune_kwargs=None` or provide a search space "
        f"for at least one model."
    )


def verify_contains_no_searchspaces(hyperparameters: Dict[str, Dict[str, Any]]):
    for model, model_hps_list in hyperparameters.items():
        if not isinstance(model_hps_list, list):
            model_hps_list = [model_hps_list]

        for model_hps in model_hps_list:
            if contains_searchspace(model_hps):
                raise ValueError(
                    f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. "
                    f"However, for model {model} hyperparameters {model_hps} contain a search space."
                )
