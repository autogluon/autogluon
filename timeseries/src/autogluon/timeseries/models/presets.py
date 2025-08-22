import copy
import logging
import re
from collections import defaultdict
from typing import Any, Optional, Type, Union

from autogluon.common import space
from autogluon.core import constants
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.models.abstract.abstract_timeseries_model import TimeSeriesModelBase
from autogluon.timeseries.utils.features import CovariateMetadata

from .abstract import AbstractTimeSeriesModel
from .multi_window.multi_window_model import MultiWindowBacktestingModel
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

ModelHyperparameters = dict[str, Any]


PRESETS: dict[str, dict[str, Union[ModelHyperparameters, list[ModelHyperparameters]]]] = {
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


class TrainableModelSetBuilder:
    """Responsible for building a list of model objects, in priority order, that will be trained by the
    Trainer."""

    VALID_AG_ARGS_KEYS = {
        "name",
        "name_prefix",
        "name_suffix",
    }

    def __init__(
        self,
        path: str,
        freq: Optional[str],
        prediction_length: int,
        eval_metric: TimeSeriesScorer,
        target: str,
        quantile_levels: list[float],
        covariate_metadata: CovariateMetadata,
        multi_window: bool,
    ):
        self.path = path
        self.freq = freq
        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.target = target
        self.quantile_levels = quantile_levels
        self.covariate_metadata = covariate_metadata
        self.multi_window = multi_window

    def get_model_set(
        self,
        hyperparameters: Union[str, dict, None],
        hyperparameter_tune: bool,
        excluded_model_types: Optional[list[str]],
        banned_model_names: Optional[list[str]] = None,
    ) -> list[TimeSeriesModelBase]:
        """Create a list of models according to given resolved and canonicalized dictionary of hyperparameters.
        Hyperparameters can be built using the `HyperparameterBuilder` class.
        """
        models = []
        banned_model_names = [] if banned_model_names is None else banned_model_names.copy()

        # resolve and normalize hyperparameters
        model_hp_map: dict[str, list[ModelHyperparameters]] = HyperparameterBuilder(
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            excluded_model_types=excluded_model_types,
        ).get_hyperparameters()

        model_priority_list = sorted(
            model_hp_map.keys(), key=lambda x: ModelRegistry.get_model_priority(x), reverse=True
        )

        for model_key in model_priority_list:
            model_type = self._get_model_type(model_key)

            for model_hps in model_hp_map[model_key]:
                ag_args = model_hps.pop(constants.AG_ARGS, {})

                for key in ag_args:
                    if key not in self.VALID_AG_ARGS_KEYS:
                        raise ValueError(
                            f"Model {model_type} received unknown ag_args key: {key} (valid keys {self.VALID_AG_ARGS_KEYS})"
                        )
                model_name_base = self._get_model_name(ag_args, model_type)

                model_type_kwargs: dict[str, Any] = dict(
                    name=model_name_base,
                    hyperparameters=model_hps,
                    **self._get_default_model_init_kwargs(),
                )

                # add models while preventing name collisions
                model = model_type(**model_type_kwargs)
                model_type_kwargs.pop("name", None)

                increment = 1
                while model.name in banned_model_names:
                    increment += 1
                    model = model_type(name=f"{model_name_base}_{increment}", **model_type_kwargs)

                if self.multi_window:
                    model = MultiWindowBacktestingModel(model_base=model, name=model.name, **model_type_kwargs)  # type: ignore

                banned_model_names.append(model.name)
                models.append(model)

        return models

    def _get_model_type(self, model: Union[str, Type[AbstractTimeSeriesModel]]) -> Type[AbstractTimeSeriesModel]:
        if isinstance(model, str):
            model_type: Type[AbstractTimeSeriesModel] = ModelRegistry.get_model_class(model)
        elif isinstance(model, type):
            model_type = model
        else:
            raise ValueError(
                f"Keys of the `hyperparameters` dictionary must be strings or types, received {type(model)}."
            )

        return model_type

    def _get_default_model_init_kwargs(self) -> dict[str, Any]:
        return dict(
            path=self.path,
            freq=self.freq,
            prediction_length=self.prediction_length,
            eval_metric=self.eval_metric,
            target=self.target,
            quantile_levels=self.quantile_levels,
            covariate_metadata=self.covariate_metadata,
        )

    def _get_model_name(self, ag_args: dict[str, Any], model_type: Type[AbstractTimeSeriesModel]) -> str:
        name = ag_args.get("name")
        if name is None:
            name_stem = re.sub(r"Model$", "", model_type.__name__)
            name_prefix = ag_args.get("name_prefix", "")
            name_suffix = ag_args.get("name_suffix", "")
            name = name_prefix + name_stem + name_suffix
        return name


class HyperparameterBuilder:
    """Given user hyperparameter specifications, this class resolves them against presets, removes
    excluded model types and canonicalizes the hyperparameter specification.
    """

    def __init__(
        self,
        hyperparameters: Union[str, dict, None],
        hyperparameter_tune: bool,
        excluded_model_types: Optional[list[str]],
    ):
        self.hyperparameters = hyperparameters
        self.hyperparameter_tune = hyperparameter_tune
        self.excluded_model_types = excluded_model_types

    def get_hyperparameters(self) -> dict[str, list[ModelHyperparameters]]:
        hyperparameter_dict = {}

        if self.hyperparameters is None:
            hyperparameter_dict = copy.deepcopy(PRESETS["default"])
        elif isinstance(self.hyperparameters, str):
            try:
                hyperparameter_dict = copy.deepcopy(PRESETS[self.hyperparameters])
            except KeyError:
                raise ValueError(f"{self.hyperparameters} is not a valid preset.")
        elif isinstance(self.hyperparameters, dict):
            hyperparameter_dict = copy.deepcopy(self.hyperparameters)
        else:
            raise ValueError(
                f"hyperparameters must be a dict, a string or None (received {type(self.hyperparameters)}). "
                f"Please see the documentation for TimeSeriesPredictor.fit"
            )

        return self._check_and_clean_hyperparameters(hyperparameter_dict)

    def _check_and_clean_hyperparameters(
        self,
        hyperparameters: dict[str, Union[ModelHyperparameters, list[ModelHyperparameters]]],
    ) -> dict[str, list[ModelHyperparameters]]:
        """Convert the hyperparameters dictionary to a unified format:
        - Remove 'Model' suffix from model names, if present
        - Make sure that each value in the hyperparameters dict is a list with model configurations
        - Checks if hyperparameters contain searchspaces
        """
        excluded_models = self._get_excluded_models()
        hyperparameters_clean = defaultdict(list)
        for model_name, model_hyperparameters in hyperparameters.items():
            # Handle model names ending with "Model", e.g., "DeepARModel" is mapped to "DeepAR"
            if isinstance(model_name, str):
                model_name = self._normalize_model_type_name(model_name)
                if model_name in excluded_models:
                    logger.info(
                        f"\tFound '{model_name}' model in `hyperparameters`, but '{model_name}' "
                        "is present in `excluded_model_types` and will be removed."
                    )
                    continue
            if not isinstance(model_hyperparameters, list):
                model_hyperparameters = [model_hyperparameters]
            hyperparameters_clean[model_name].extend(model_hyperparameters)

        self._verify_search_spaces(hyperparameters_clean)

        return dict(hyperparameters_clean)

    def _get_excluded_models(self) -> set[str]:
        excluded_models = set()
        if self.excluded_model_types is not None and len(self.excluded_model_types) > 0:
            if not isinstance(self.excluded_model_types, list):
                raise ValueError(f"`excluded_model_types` must be a list, received {type(self.excluded_model_types)}")
            logger.info(f"Excluded model types: {self.excluded_model_types}")
            for model in self.excluded_model_types:
                if not isinstance(model, str):
                    raise ValueError(f"Each entry in `excluded_model_types` must be a string, received {type(model)}")
                excluded_models.add(self._normalize_model_type_name(model))
        return excluded_models

    @staticmethod
    def _normalize_model_type_name(model_name: str) -> str:
        return model_name.removesuffix("Model")

    def _verify_search_spaces(self, hyperparameters: dict[str, list[ModelHyperparameters]]):
        if self.hyperparameter_tune:
            for model, model_hps_list in hyperparameters.items():
                for model_hps in model_hps_list:
                    if contains_searchspace(model_hps):
                        return

            raise ValueError(
                "Hyperparameter tuning specified, but no model contains a hyperparameter search space. "
                "Please disable hyperparameter tuning with `hyperparameter_tune_kwargs=None` or provide a search space "
                "for at least one model."
            )
        else:
            for model, model_hps_list in hyperparameters.items():
                for model_hps in model_hps_list:
                    if contains_searchspace(model_hps):
                        raise ValueError(
                            f"Hyperparameter tuning not specified, so hyperparameters must have fixed values. "
                            f"However, for model {model} hyperparameters {model_hps} contain a search space."
                        )


def contains_searchspace(model_hyperparameters: ModelHyperparameters) -> bool:
    for hp_value in model_hyperparameters.values():
        if isinstance(hp_value, space.Space):
            return True
    return False
