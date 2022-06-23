import copy
import logging
from typing import Union, Dict, List

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
    TransformerModel,
)
from .sktime import AutoARIMAModel, AutoETSModel


logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    MQRNN=MQRNNModel,
    SimpleFeedForward=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    AutoTabular=AutoTabularModel,
    Prophet=ProphetModel,
    Transformer=TransformerModel,
    AutoARIMA=AutoARIMAModel,
    AutoETS=AutoETSModel
)
DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=40,
    MQRNN=40,
    SimpleFeedForward=50,
    Transformer=40,
    DeepAR=50,
    Prophet=10,
    AutoTabular=10,
    AutoARIMA=20,
    AutoETS=60,
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
            "AutoETS": {},
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
            "AutoETS": {},
            # "AutoARIMA": {},
            "SimpleFeedForward": {},
            "MQCNN": {},
            "MQRNN": {},
            "DeepAR": {},
            "Transformer": {},
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
            "AutoETS": {"error": ag.Categorical("add", "mul")},
            # "AutoARIMA": {"max_p": ag.Int(2, 4)}
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

    model_priority_list = sorted(
        hyperparameters.keys(),
        key=lambda x: DEFAULT_MODEL_PRIORITY.get(x, 0),
        reverse=True
    )

    for model in model_priority_list:
        model_hps = hyperparameters[model]
        if isinstance(model, str):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            model_type = MODEL_TYPES[model]
        elif isinstance(model, AbstractTimeSeriesModelFactory):
            model_type = model
        elif not issubclass(model, AbstractTimeSeriesModel):
            logger.warning(
                f"Customized model {model} does not inherit from {AbstractTimeSeriesModel}"
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
