import copy
import logging
from typing import Any, Dict, List, Union

import autogluon.core as ag
import autogluon.timeseries as agts

from . import (
    ARIMAModel,
    AutoARIMAModel,
    AutoETSModel,
    AutoGluonTabularModel,
    DeepARModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    NaiveModel,
    SeasonalNaiveModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    ThetaModel,
)
from .abstract import AbstractTimeSeriesModel, AbstractTimeSeriesModelFactory

logger = logging.getLogger(__name__)

# define the model zoo with their aliases
MODEL_TYPES = dict(
    SimpleFeedForward=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    TemporalFusionTransformer=TemporalFusionTransformerModel,
    # Prophet=ProphetModel,
    ETS=ETSModel,
    ARIMA=ARIMAModel,
    Theta=ThetaModel,
    AutoGluonTabular=AutoGluonTabularModel,
    Naive=NaiveModel,
    SeasonalNaive=SeasonalNaiveModel,
    AutoETS=AutoETSModel,
    AutoARIMA=AutoARIMAModel,
    DynamicOptimizedTheta=DynamicOptimizedThetaModel,
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

if agts.SKTIME_INSTALLED:
    from .sktime import ARIMASktimeModel, AutoARIMASktimeModel, AutoETSSktimeModel

    MODEL_TYPES.update(
        dict(
            ARIMASktime=ARIMASktimeModel,
            AutoARIMASktime=AutoARIMASktimeModel,
            AutoETSSktime=AutoETSSktimeModel,
        )
    )

DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    Naive=100,
    SeasonalNaive=100,
    ETS=90,
    Theta=90,
    ARIMA=80,
    AutoGluonTabular=70,
    DeepAR=60,
    TemporalFusionTransformer=50,
    SimpleFeedForward=40,
    TransformerMXNet=30,
    AutoARIMA=50,
    AutoETS=70,
    DynamicOptimizedTheta=60,
    # Models below are not included in any presets
    AutoETSSktime=60,
    ARIMASktime=50,
    DeepARMXNet=50,
    SimpleFeedForwardMXNet=30,
    TemporalFusionTransformerMXNet=50,
    AutoARIMASktime=20,
    MQCNNMXNet=10,
    MQRNNMXNet=10,
)
DEFAULT_CUSTOM_MODEL_PRIORITY = 0
MINIMUM_CONTEXT_LENGTH = 10


def get_default_hps(key, prediction_length):
    context_length = max(prediction_length * 2, MINIMUM_CONTEXT_LENGTH)
    default_model_hps = {
        "local_only": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "Theta": {},
        },
        "medium_quality": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "AutoETS": {},
            "Theta": {},
            "AutoGluonTabular": {},
            "DeepAR": {
                "context_length": context_length,
            },
        },
        "high_quality": {
            "Naive": {},
            "SeasonalNaive": {},
            "ARIMA": {},
            "ETS": {},
            "AutoETS": {},
            "AutoARIMA": {},
            "Theta": {
                "deseasonalize": ag.Categorical(True, False),
            },
            "AutoGluonTabular": {},
            "DeepAR": {
                "context_length": context_length,
            },
            "SimpleFeedForward": {
                "context_length": context_length,
            },
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
            "Theta": {
                "deseasonalize": ag.Categorical(True, False),
            },
            "DeepAR": {
                "context_length": context_length,
                "num_layers": ag.Int(1, 3, default=2),
                "hidden_size": ag.Int(40, 80, default=40),
            },
            "SimpleFeedForward": {
                "context_length": context_length,
                "hidden_dimensions": ag.Categorical([40], [40, 40], [120]),
            },
            "TemporalFusionTransformer": {},
        },
    }

    # update with MXNet if installed
    if agts.MXNET_INSTALLED:
        mxnet_default_updates = {
            "best_quality": {
                "TransformerMXNet": {"context_length": context_length},
            },
        }
        for k in default_model_hps:
            default_model_hps[k] = dict(**default_model_hps[k], **mxnet_default_updates.get(k, {}))

    # For backwards compatibility
    default_model_hps["default"] = default_model_hps["medium_quality"]
    default_model_hps["default_hpo"] = default_model_hps["best_quality"]

    return default_model_hps[key]


def get_preset_models(
    freq: str,
    prediction_length: int,
    path: str,
    eval_metric: str,
    hyperparameters: Union[str, Dict],
    hyperparameter_tune: bool,
    invalid_model_names: List[str],
    **kwargs,
):
    """
    Create a list of models according to hyperparameters. If hyperparamaters=None,
    will create models according to presets.
    """
    models = []
    if hyperparameters is None:
        hp_string = "default_hpo" if hyperparameter_tune else "default"
        hyperparameters = copy.deepcopy(get_default_hps(hp_string, prediction_length))
    elif isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(get_default_hps(hyperparameters, prediction_length))
    elif isinstance(hyperparameters, dict):
        default_hps = copy.deepcopy(get_default_hps("default", prediction_length))
        updated_hyperparameters = {}
        # Only train models from `hyperparameters`, overload default HPs if provided
        for model, hps in hyperparameters.items():
            updated_hyperparameters[model] = default_hps.get(model, {})
            updated_hyperparameters[model].update(hps)
        hyperparameters = copy.deepcopy(updated_hyperparameters)
    else:
        raise ValueError(
            f"hyperparameters must be a dict, a string or None (received {type(hyperparameters)}). "
            f"Please see the documentation for TimeSeriesPredictor.fit"
        )

    if hyperparameter_tune:
        verify_contains_at_least_one_searchspace(hyperparameters)
    else:
        verify_contains_no_searchspaces(hyperparameters)

    invalid_model_names = set(invalid_model_names)
    all_assigned_names = set(invalid_model_names)

    model_priority_list = sorted(hyperparameters.keys(), key=lambda x: DEFAULT_MODEL_PRIORITY.get(x, 0), reverse=True)

    for model in model_priority_list:
        model_hps = hyperparameters[model]
        if isinstance(model, str):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            model_type = MODEL_TYPES[model]
        elif isinstance(model, AbstractTimeSeriesModelFactory):
            model_type = model
        elif not issubclass(model, AbstractTimeSeriesModel):
            logger.warning(f"Customized model {model} does not inherit from {AbstractTimeSeriesModel}")
            model_type = model
        else:
            logger.log(20, f"Custom Model Type Detected: {model}")
            model_type = model

        model_type_kwargs = dict(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
            hyperparameters=model_hps,
            **kwargs,
        )

        # add models while preventing name collisions
        model = model_type(**model_type_kwargs)
        name_stem = model.name

        model_type_kwargs.pop("name", None)
        increment = 1
        while model.name in all_assigned_names:
            increment += 1
            model = model_type(name=f"{name_stem}_{increment}", **model_type_kwargs)

        all_assigned_names.add(model.name)
        models.append(model)

    return models


def contains_searchspace(model_hyperparameters: Dict[str, Any]) -> bool:
    for hp_value in model_hyperparameters.values():
        if isinstance(hp_value, ag.space.Space):
            return True
    return False


def verify_contains_at_least_one_searchspace(hyperparameters: Dict[str, Dict[str, Any]]):
    if not any(contains_searchspace(model_hps) for model_hps in hyperparameters.values()):
        raise ValueError(
            f"Hyperparameter tuning specified, but no model contains a hyperparameter search space. "
            f"Please disable hyperparameter tuning with `hyperparameter_tune_kwargs=None` or provide a search space "
            f"for at least one model."
        )


def verify_contains_no_searchspaces(hyperparameters: Dict[str, Dict[str, Any]]):
    for model, model_hps in hyperparameters.items():
        if contains_searchspace(model_hps):
            raise ValueError(
                f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. "
                f"However, for model {model} hyperparameters {model_hps} contain a search space."
            )
