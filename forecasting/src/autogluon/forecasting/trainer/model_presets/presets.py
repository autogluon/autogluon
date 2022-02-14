import copy
import inspect
import logging

import autogluon.core as ag

from ...models.abstract.abstract_forecasting_model import AbstractForecastingModel
from ...models.gluonts_model.mqcnn import MQCNNModel
from ...models.gluonts_model.sff import SimpleFeedForwardModel
from ...models.gluonts_model.deepar import DeepARModel
from ...models.gluonts_model.auto_tabular import (
    AutoTabularModel,
)  # TODO: this module is not fully prepared.

logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    SFF=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    AutoTabular=AutoTabularModel,
)
DEFAULT_MODEL_NAMES = {v: k for k, v in MODEL_TYPES.items()}
DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=50,
    SFF=30,
    DeepAR=50,
    AutoTabular=10,
)


def get_default_hps(key, prediction_length):
    DEFAULT_MODEL_HPS = {
        "toy": {
            "SFF": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
            "MQCNN": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
            "DeepAR": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
        },
        "toy_hpo": {
            "SFF": {
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
            "SFF": {},
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
            "SFF": {
                "context_length": ag.Int(
                    min(prediction_length, max(10, 2 * prediction_length), 250),
                    max(min(500, 12 * prediction_length), prediction_length),
                    default=prediction_length,
                ),
            },
        },
    }
    return DEFAULT_MODEL_HPS[key]


DEFAULT_CUSTOM_MODEL_PRIORITY = 0


def get_preset_models(
    path,
    prediction_length,
    freq,
    eval_metric,
    hyperparameters,
    hyperparameter_tune,
    use_feat_static_cat,
    use_feat_static_real,
    cardinality,
    **kwargs,
):
    """
    Create a list of models according to hyperparameters. If hyperparamaters=None, will create models according to presets.
    """
    models = []
    if isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(
            get_default_hps(hyperparameters, prediction_length)
        )
    else:
        if not hyperparameter_tune:
            hp_str = "default"
        else:
            hp_str = "default_hpo"
        default_hps = copy.deepcopy(get_default_hps(hp_str, prediction_length))
        if hyperparameters is not None:
            default_hps = {
                model: default_hps[model]
                for model in default_hps
                if model in hyperparameters
            }
            for model in hyperparameters:
                if model not in default_hps:
                    default_hps[model] = hyperparameters[model]
                else:
                    default_hps[model].update(hyperparameters[model])
        hyperparameters = copy.deepcopy(default_hps)
    if hyperparameter_tune:
        verify_contains_searchspace(hyperparameters)
    else:
        verify_no_searchspace(hyperparameters)
    for model, model_hps in hyperparameters.items():
        if "use_feat_static_cat" not in model_hps:
            model_hps["use_feat_static_cat"] = use_feat_static_cat
        if "use_feat_static_real" not in model_hps:
            model_hps["use_feat_static_real"] = use_feat_static_real
        if "cardinality" not in model_hps:
            model_hps["cardinality"] = cardinality
        if not inspect.isclass(model):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            model_type = MODEL_TYPES[model]
        elif not issubclass(model, AbstractForecastingModel):
            logger.warning(
                f"Customized model {model} does not inherit from {AbstractForecastingModel}"
            )
            model_type = model
        else:
            logger.log(20, f"Custom Model Type Detected: {model}")
            model_type = model

        models.append(
            model_type(
                path=path,
                freq=freq,
                prediction_length=prediction_length,
                eval_metric=eval_metric,
                hyperparameters=model_hps,
                **kwargs,
            )
        )
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
                f"Hyperparameter tuning specified, but no hyperparameter search space provided for {model}. Please convert one of the fixed hyperparameter values of this model to a search space and try again, or do not specify hyperparameter tuning."
            )


def verify_no_searchspace(hyperparameters):
    for model in hyperparameters:
        model_hps = hyperparameters[model]
        for hp in model_hps:
            hp_value = model_hps[hp]
            if isinstance(hp_value, ag.space.Space):
                raise ValueError(
                    f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. For {model}, hyperparameter {hp} currently given as search space: {hp_value}."
                )
