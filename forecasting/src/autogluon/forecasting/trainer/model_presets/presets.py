from ...models.gluonts_model.mqcnn.mqcnn_model import MQCNNModel
from ...models.gluonts_model.sff.simple_feedforward_model import SimpleFeedForwardModel
from ...models.gluonts_model.deepar.deepar_model import DeepARModel
from ...models.abstract.abstract_model import AbstractModel
import copy
from collections import defaultdict
from ...constants import AG_ARGS, AG_ARGS_FIT
import inspect
import logging

logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    SFF=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
)

DEFAULT_MODEL_NAMES = {
    MQCNNModel: "MQCNN",
    SimpleFeedForwardModel: "SFF",
    DeepARModel: "DeepAR"
}

DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=50,
    SFF=30,
    DeepAR=50,
)

DEFAULT_MODEL_HPS = {
    "default": {
        "MQCNN": {},
        "SFF": {},
        "DeepAR": {},
    }
}

DEFAULT_CUSTOM_MODEL_PRIORITY = 0


def get_preset_models(path, eval_metric, hyperparameters, freq, prediction_length, **kwargs):

    models = []
    if isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(DEFAULT_MODEL_HPS[hyperparameters])
    elif hyperparameters is None:
        hyperparameters = copy.deepcopy(DEFAULT_MODEL_HPS["default"])
    for model, model_hps in hyperparameters.items():
        if not inspect.isclass(model):
            if model not in MODEL_TYPES:
                raise ValueError(f"Model {model} is not supported yet.")
            model_type = MODEL_TYPES[model]
        elif not issubclass(model, AbstractModel):
            logger.warning(f"Customized model {model} does not inherit from {AbstractModel}")
            model_type = model
        else:
            logger.log(20, f'Custom Model Type Detected: {model}')
            model_type = model

        models.append(model_type(path=path, freq=freq, prediction_length=prediction_length, eval_metric=eval_metric,
                                 hyperparameters=model_hps, **kwargs))
    return models

