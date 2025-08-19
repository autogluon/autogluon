import copy
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from autogluon.common import space
from autogluon.core import constants
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.utils.features import CovariateMetadata

from .abstract import AbstractTimeSeriesModel
from .multi_window.multi_window_model import MultiWindowBacktestingModel
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

ModelHyperparameters = Dict[str, Any]


VALID_AG_ARGS_KEYS = {
    "name",
    "name_prefix",
    "name_suffix",
}


def get_default_hps(key):
    default_model_hps = {
        "very_light": {
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {"max_num_samples": 100_000},
            "DirectTabular": {"max_num_samples": 100_000},
        },
        "light": {
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "Chronos": {"model_path": "bolt_small"},
        },
        "light_inference": {
            "SeasonalNaive": {},
            "DirectTabular": {},
            "RecursiveTabular": {},
            "TemporalFusionTransformer": {},
            "PatchTST": {},
        },
        "default": {
            "SeasonalNaive": {},
            "AutoETS": {},
            "NPTS": {},
            "DynamicOptimizedTheta": {},
            "RecursiveTabular": {},
            "DirectTabular": {},
            "TemporalFusionTransformer": {},
            "PatchTST": {},
            "DeepAR": {},
            "Chronos": [
                {
                    "ag_args": {"name_suffix": "ZeroShot"},
                    "model_path": "bolt_base",
                },
                {
                    "ag_args": {"name_suffix": "FineTuned"},
                    "model_path": "bolt_small",
                    "fine_tune": True,
                    "target_scaler": "standard",
                    "covariate_regressor": {"model_name": "CAT", "model_hyperparameters": {"iterations": 1_000}},
                },
            ],
            "TiDE": {
                "encoder_hidden_dim": 256,
                "decoder_hidden_dim": 256,
                "temporal_hidden_dim": 64,
                "num_batches_per_epoch": 100,
                "lr": 1e-4,
            },
        },
    }
    return default_model_hps[key]


def get_preset_models(
    freq: Optional[str],
    prediction_length: int,
    path: str,
    eval_metric: Union[str, TimeSeriesScorer],
    hyperparameters: Union[str, Dict, None],
    hyperparameter_tune: bool,
    covariate_metadata: CovariateMetadata,
    all_assigned_names: List[str],
    excluded_model_types: Optional[List[str]],
    multi_window: bool = False,
    **kwargs,
):
    """
    Create a list of models according to hyperparameters. If hyperparamaters=None,
    will create models according to presets.
    """
    models = []
    hyperparameter_dict = get_hyperparameter_dict(hyperparameters, hyperparameter_tune)

    model_priority_list = sorted(
        hyperparameter_dict.keys(), key=lambda x: ModelRegistry.get_model_priority(x), reverse=True
    )
    excluded_models = get_excluded_models(excluded_model_types)
    all_assigned_names = all_assigned_names.copy()

    for model in model_priority_list:
        if isinstance(model, str):
            if model in excluded_models:
                logger.info(
                    f"\tFound '{model}' model in `hyperparameters`, but '{model}' "
                    "is present in `excluded_model_types` and will be removed."
                )
                continue
            model_type: Type[AbstractTimeSeriesModel] = ModelRegistry.get_model_class(model)
        elif isinstance(model, type):
            if not issubclass(model, AbstractTimeSeriesModel):
                raise ValueError(f"Custom model type {model} must inherit from `AbstractTimeSeriesModel`.")
            model_type = model
        else:
            raise ValueError(
                f"Keys of the `hyperparameters` dictionary must be strings or types, received {type(model)}."
            )

        for model_hps in hyperparameter_dict[model]:
            ag_args = model_hps.pop(constants.AG_ARGS, {})
            for key in ag_args:
                if key not in VALID_AG_ARGS_KEYS:
                    raise ValueError(
                        f"Model {model_type} received unknown ag_args key: {key} (valid keys {VALID_AG_ARGS_KEYS})"
                    )
            model_name_base = get_model_name(ag_args, model_type)

            model_type_kwargs: Dict[str, Any] = dict(
                name=model_name_base,
                path=path,
                freq=freq,
                prediction_length=prediction_length,
                eval_metric=eval_metric,
                covariate_metadata=covariate_metadata,
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
                model = MultiWindowBacktestingModel(model_base=model, name=model.name, **model_type_kwargs)  # type: ignore

            all_assigned_names.append(model.name)
            models.append(model)

    return models


def get_excluded_models(excluded_model_types):
    excluded_models = set()
    if excluded_model_types is not None and len(excluded_model_types) > 0:
        if not isinstance(excluded_model_types, list):
            raise ValueError(f"`excluded_model_types` must be a list, received {type(excluded_model_types)}")
        logger.info(f"Excluded model types: {excluded_model_types}")
        for model in excluded_model_types:
            if not isinstance(model, str):
                raise ValueError(f"Each entry in `excluded_model_types` must be a string, received {type(model)}")
            excluded_models.add(normalize_model_type_name(model))
    return excluded_models


def get_hyperparameter_dict(
    hyperparameters: Optional[Union[str, Dict[str, Union[ModelHyperparameters, List[ModelHyperparameters]]]]],
    hyperparameter_tune: bool,
) -> Dict[str, List[ModelHyperparameters]]:
    hyperparameter_dict = {}

    if hyperparameters is None:
        hyperparameter_dict = copy.deepcopy(get_default_hps("default"))
    elif isinstance(hyperparameters, str):
        hyperparameter_dict = copy.deepcopy(get_default_hps(hyperparameters))
    elif isinstance(hyperparameters, dict):
        hyperparameter_dict = copy.deepcopy(hyperparameters)
    else:
        raise ValueError(
            f"hyperparameters must be a dict, a string or None (received {type(hyperparameters)}). "
            f"Please see the documentation for TimeSeriesPredictor.fit"
        )

    hyperparameter_dict = check_and_clean_hyperparameters(
        hyperparameter_dict, must_contain_searchspace=hyperparameter_tune
    )

    return hyperparameter_dict


def normalize_model_type_name(model_name: str) -> str:
    """Remove 'Model' suffix from the end of the string, if it's present."""
    if model_name.endswith("Model"):
        model_name = model_name[: -len("Model")]
    return model_name


def check_and_clean_hyperparameters(
    hyperparameters: Dict[str, Union[ModelHyperparameters, List[ModelHyperparameters]]],
    must_contain_searchspace: bool,
) -> Dict[str, List[ModelHyperparameters]]:
    """Convert the hyperparameters dictionary to a unified format:
    - Remove 'Model' suffix from model names, if present
    - Make sure that each value in the hyperparameters dict is a list with model configurations
    - Checks if hyperparameters contain searchspaces
    """
    hyperparameters_clean = defaultdict(list)
    for key, value in hyperparameters.items():
        # Handle model names ending with "Model", e.g., "DeepARModel" is mapped to "DeepAR"
        if isinstance(key, str):
            key = normalize_model_type_name(key)
        if not isinstance(value, list):
            value = [value]
        hyperparameters_clean[key].extend(value)

    if must_contain_searchspace:
        verify_contains_at_least_one_searchspace(hyperparameters_clean)
    else:
        verify_contains_no_searchspaces(hyperparameters_clean)

    return dict(hyperparameters_clean)


def get_model_name(ag_args: Dict[str, Any], model_type: Type[AbstractTimeSeriesModel]) -> str:
    name = ag_args.get("name")
    if name is None:
        name_stem = re.sub(r"Model$", "", model_type.__name__)
        name_prefix = ag_args.get("name_prefix", "")
        name_suffix = ag_args.get("name_suffix", "")
        name = name_prefix + name_stem + name_suffix
    return name


def contains_searchspace(model_hyperparameters: ModelHyperparameters) -> bool:
    for hp_value in model_hyperparameters.values():
        if isinstance(hp_value, space.Space):
            return True
    return False


def verify_contains_at_least_one_searchspace(hyperparameters: Dict[str, List[ModelHyperparameters]]):
    for model, model_hps_list in hyperparameters.items():
        for model_hps in model_hps_list:
            if contains_searchspace(model_hps):
                return

    raise ValueError(
        "Hyperparameter tuning specified, but no model contains a hyperparameter search space. "
        "Please disable hyperparameter tuning with `hyperparameter_tune_kwargs=None` or provide a search space "
        "for at least one model."
    )


def verify_contains_no_searchspaces(hyperparameters: Dict[str, List[ModelHyperparameters]]):
    for model, model_hps_list in hyperparameters.items():
        for model_hps in model_hps_list:
            if contains_searchspace(model_hps):
                raise ValueError(
                    f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. "
                    f"However, for model {model} hyperparameters {model_hps} contain a search space."
                )
