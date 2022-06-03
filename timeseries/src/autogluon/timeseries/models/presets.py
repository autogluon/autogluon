import copy
import logging
from typing import Union, Dict, List

import autogluon.core as ag

from .abstract import AbstractForecastingModel
from .abstract.abstract_forecasting_model import AbstractForecastingModelFactory
from .gluonts import (
    DeepARModel,
    MQCNNModel,
    ProphetModel,
    SimpleFeedForwardModel,
    AutoTabularModel,
)

logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    SimpleFeedForward=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    AutoTabular=AutoTabularModel,
    Prophet=ProphetModel,
)
DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=50,
    SimpleFeedForward=30,
    DeepAR=50,
    Prophet=50,
    AutoTabular=10,
)
DEFAULT_CUSTOM_MODEL_PRIORITY = 0


def get_default_hps(key, prediction_length):
    default_model_hps = {
        "toy": {
            "SimpleFeedForward": {
                "epochs": 5,
                "num_batches_per_epoch": 10,
                "context_length": 5,
            },
            "MQCNN": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
            "DeepAR": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
        },
        "toy_hpo": {
            "SimpleFeedForward": {
                "epochs": 5,
                "num_batches_per_epoch": 10,
                "context_length": ag.Int(5, 25),
            },
            "MQCNN": {
                "epochs": 5,
                "num_batches_per_epoch": 10,
                "context_length": ag.Int(5, 25),
            },
            "DeepAR": {
                "epochs": 5,
                "num_batches_per_epoch": 10,
                "context_length": ag.Int(5, 25),
            },
        },
        "default": {
            "SimpleFeedForward": {},
            "MQCNN": {},
            "DeepAR": {},
            # "AutoTabular": {} # AutoTabular model is experimental.
        },
        "default_hpo": {
            "MQCNN": {
                "context_length": ag.Int(
                    min(prediction_length, max(10, 2 * prediction_length), 250),
                    max(min(500, 12 * prediction_length), 4 * prediction_length),
                    default=prediction_length * 4,
                ),
            },
            "DeepAR": {
                "context_length": ag.Int(
                    min(prediction_length, max(10, 2 * prediction_length), 250),
                    max(min(500, 12 * prediction_length), prediction_length),
                    default=prediction_length,
                ),
            },
            "SimpleFeedForward": {
                "context_length": ag.Int(
                    min(prediction_length, max(10, 2 * prediction_length), 250),
                    max(min(500, 12 * prediction_length), prediction_length),
                    default=prediction_length,
                ),
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
        hyperparameters = copy.deepcopy(
            get_default_hps(hyperparameters, prediction_length)
        )
    else:
        hp_str = "default" if not hyperparameter_tune else "default_hpo"
        default_hps = copy.deepcopy(get_default_hps(hp_str, prediction_length))

        if hyperparameters is not None:
            # filter only default_hps for models with hyperparameters provided
            default_hps = {
                model: default_hps.get(model, {}) for model in hyperparameters
            }
            for model in hyperparameters:
                default_hps[model].update(hyperparameters[model])
        hyperparameters = copy.deepcopy(default_hps)

    if hyperparameter_tune:
        verify_contains_searchspace(hyperparameters)
    else:
        verify_no_searchspace(hyperparameters)

    invalid_model_names = set(invalid_model_names)
    all_assigned_names = set(invalid_model_names)

    for model, model_hps in hyperparameters.items():
        if isinstance(model, str):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            model_type = MODEL_TYPES[model]
        elif isinstance(model, AbstractForecastingModelFactory):
            model_type = model
        elif not issubclass(model, AbstractForecastingModel):
            logger.warning(
                f"Customized model {model} does not inherit from {AbstractForecastingModel}"
            )
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
