import copy
import logging
from typing import Dict, List, Union

import autogluon.core as ag

from .abstract import AbstractTimeSeriesModel
from .abstract.abstract_timeseries_model import AbstractTimeSeriesModelFactory
from .gluonts import (
    AutoTabularModel,
    DeepARModel,
    MQCNNModel,
    MQRNNModel,
    ProphetModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    TransformerModel,
)
from .sktime import SktimeARIMAModel, SktimeAutoARIMAModel, SktimeAutoETSModel
from .statsmodels import ARIMAModel, ETSModel, ThetaModel

logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    MQRNN=MQRNNModel,
    SimpleFeedForward=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    AutoTabular=AutoTabularModel,
    Prophet=ProphetModel,
    Transformer=TransformerModel,
    TemporalFusionTransformer=TemporalFusionTransformerModel,
    SktimeARIMA=SktimeARIMAModel,
    SktimeAutoARIMA=SktimeAutoARIMAModel,
    SktimeAutoETS=SktimeAutoETSModel,
    ETS=ETSModel,
    ARIMA=ARIMAModel,
    Theta=ThetaModel,
)
DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=40,
    MQRNN=40,
    SimpleFeedForward=50,
    Transformer=40,
    TemporalFusionTransformer=45,
    DeepAR=50,
    Prophet=10,
    AutoTabular=10,
    SktimeAutoARIMA=20,
    SktimeARIMA=50,
    SktimeAutoETS=60,
    ARIMA=50,
    ETS=60,
    Theta=60,
)
DEFAULT_CUSTOM_MODEL_PRIORITY = 0
MINIMUM_CONTEXT_LENGTH = 10


def get_default_hps(key, prediction_length):
    context_length = max(prediction_length * 2, MINIMUM_CONTEXT_LENGTH)
    default_model_hps = {
        "local_only": {
            "ARIMA": {},
            "ETS": {},
            "Theta": {},
        },
        "default": {
            "ARIMA": {},
            "ETS": {},
            "Theta": {},
            "SimpleFeedForward": {
                "context_length": context_length,
            },
            "DeepAR": {
                "context_length": context_length,
            },
            "TemporalFusionTransformer": {
                "context_length": context_length,
            },
        },
        "default_hpo": {
            "ETS": {
                "trend": ag.Categorical("add", None),
                "seasonal": ag.Categorical("add", None),
            },
            "ARIMA": {
                "order": ag.Categorical((2, 0, 1), (2, 1, 0), (2, 1, 1), (1, 1, 1)),
                "seasonal_order": ag.Categorical((0, 0, 0), (1, 0, 0)),
            },
            "DeepAR": {
                "cell_type": ag.Categorical("gru", "lstm"),
                "num_layers": ag.Int(1, 3),
                "num_cells": ag.Categorical(40, 80),
                "context_length": context_length,
            },
            "SimpleFeedForward": {
                "num_hidden_dimensions": ag.Categorical([40], [40, 40], [120]),
                "batch_size": 64,
                "context_length": context_length,
            },
            "Transformer": {
                "model_dim": ag.Categorical(32, 64),
                "batch_size": 64,
                "context_length": context_length,
            },
            "TemporalFusionTransformer": {
                "hidden_dim": ag.Categorical(32, 64),
                "batch_size": 64,
                "context_length": context_length,
            },
            "Theta": {
                "deseasonalize": ag.Categorical(True, False),
                "method": ag.Categorical("auto", "additive"),
            },
        },
    }
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
    if isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(get_default_hps(hyperparameters, prediction_length))
    else:
        hp_str = "default" if not hyperparameter_tune else "default_hpo"
        default_hps = copy.deepcopy(get_default_hps(hp_str, prediction_length))

        if hyperparameters is not None:
            # filter only default_hps for models with hyperparameters provided
            default_hps = {model: default_hps.get(model, {}) for model in hyperparameters}
            for model in hyperparameters:
                default_hps[model].update(hyperparameters[model])
        hyperparameters = copy.deepcopy(default_hps)

    if hyperparameter_tune:
        verify_contains_searchspace(hyperparameters)
    else:
        verify_no_searchspace(hyperparameters)

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


def verify_contains_searchspace(hyperparameters):
    for model in hyperparameters:
        model_contains_searchspace = False
        model_hps = hyperparameters[model]
        for hp in model_hps:
            hp_value = model_hps[hp]
            if isinstance(hp_value, ag.space.Space):
                model_contains_searchspace = True
                break
        if not model_contains_searchspace:
            raise ValueError(
                f"Hyperparameter tuning specified, but no hyperparameter search space provided for {model}. "
                f"Please convert one of the fixed hyperparameter values of this model to a search space and "
                f"try again, or do not specify hyperparameter tuning."
            )


def verify_no_searchspace(hyperparameters):
    for model in hyperparameters:
        model_hps = hyperparameters[model]
        for hp in model_hps:
            hp_value = model_hps[hp]
            if isinstance(hp_value, ag.space.Space):
                raise ValueError(
                    f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. For "
                    f"{model}, hyperparameter {hp} currently given as search space: {hp_value}."
                )
