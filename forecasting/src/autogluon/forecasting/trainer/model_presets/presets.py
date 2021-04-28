from ...models.gluonts_model.mqcnn.mqcnn_model import MQCNNModel
from ...models.gluonts_model.sff.simple_feedforward_model import SimpleFeedForwardModel
from ...models.gluonts_model.deepar.deepar_model import DeepARModel
from ...models.gluonts_model.auto_tabular.auto_tabular_model import AutoTabularModel  # TODO: this module is not fully prepared.
from ...models.abstract.abstract_model import AbstractModel
import copy
import inspect
import logging
import autogluon.core as ag
logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel,
    SFF=SimpleFeedForwardModel,
    DeepAR=DeepARModel,
    AutoTabular=AutoTabularModel,
)

# disable AutoTabular until its saving issue is fixed.
DEFAULT_MODEL_NAMES = {
    MQCNNModel: "MQCNN",
    SimpleFeedForwardModel: "SFF",
    DeepARModel: "DeepAR",
    AutoTabularModel: "AutoTabular",
}

DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=50,
    SFF=30,
    DeepAR=50,
    AutoTabular=10,
)


def get_default_hps(key, prediction_length):
    DEFAULT_MODEL_HPS = {
        "default": {
            "MQCNN": {},
            "SFF": {},
            "DeepAR": {},
            "AutoTabular": {} # Predicting with AutoTabular model seems quite slow.
        },
        "default_hpo": {
            "MQCNN": {
                'context_length': ag.Int(min(prediction_length, max(10, 2 * prediction_length), 250), max(min(500,12*prediction_length), 4 * prediction_length),
                                         default=prediction_length * 4),
                "num_batches_per_epoch": 32,
                "epochs": 50},
            "DeepAR": {
                'context_length': ag.Int(min(prediction_length, max(10, 2 * prediction_length), 250), max(min(500,12*prediction_length), 4 * prediction_length),
                                         default=prediction_length),
                "num_batches_per_epoch": 32,
                "epochs": 50},
            "SFF": {
                'context_length': ag.Int(min(prediction_length, max(10, 2 * prediction_length), 250), max(min(500,12*prediction_length), 4 * prediction_length),
                                         default=prediction_length),
                "num_batches_per_epoch": 32,
                "epochs": 50},
        }
    }
    return DEFAULT_MODEL_HPS[key]


DEFAULT_CUSTOM_MODEL_PRIORITY = 0


def get_preset_models(path, prediction_length, freq, eval_metric, hyperparameters, hyperparameter_tune, use_feat_static_cat, use_feat_static_real, cardinality, **kwargs):

    models = []
    if isinstance(hyperparameters, str):
        hyperparameters = copy.deepcopy(get_default_hps(hyperparameters, prediction_length))
    elif hyperparameters is None:
        if not hyperparameter_tune:
            hyperparameters = copy.deepcopy(get_default_hps('default', prediction_length))
        else:
            hyperparameters = copy.deepcopy(get_default_hps('default_hpo', prediction_length))
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
        elif not issubclass(model, AbstractModel):
            logger.warning(f"Customized model {model} does not inherit from {AbstractModel}")
            model_type = model
        else:
            logger.log(20, f'Custom Model Type Detected: {model}')
            model_type = model

        models.append(model_type(path=path, freq=freq, prediction_length=prediction_length, eval_metric=eval_metric,
                                 hyperparameters=model_hps, **kwargs))
    return models

