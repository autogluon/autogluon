from __future__ import annotations

import copy
from typing import Union

from Cython.Utils import OrderedSet

from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from autogluon.tabular.trainer.model_presets.presets import MODEL_TYPES


class ConfigBuilder:
    def __init__(self):
        self.config = {}

    def with_presets(self, presets: Union[str, list]) -> ConfigBuilder:
        valid_keys = list(tabular_presets_dict.keys())
        if not isinstance(presets, list):
            presets = [presets]

        unknown_keys = [k for k in presets if k not in valid_keys]
        assert len(unknown_keys) == 0, f'The following preset are not recognized: {unknown_keys} - use one of the valid presets: {valid_keys}'

        self.config['presets'] = presets
        return self

    def with_excluded_model_types(self, models: Union[str, list]) -> ConfigBuilder:
        valid_keys = [m for m in MODEL_TYPES.keys() if m not in ['ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED']]
        if not isinstance(models, list):
            models = [models]
        for model in models:
            assert model in valid_keys, f'{model} is not one of the valid models {valid_keys}'
        self.config['excluded_model_types'] = list(OrderedSet(models))
        return self

    def with_included_model_types(self, models: Union[str, list]) -> ConfigBuilder:
        valid_keys = [m for m in MODEL_TYPES.keys() if m not in ['ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED']]
        if not isinstance(models, list):
            models = [models]

        unknown_keys = [k for k in models if k not in valid_keys]
        assert len(unknown_keys) == 0, f'The following model types are not recognized: {unknown_keys} - use one of the valid models: {valid_keys}'

        models = [m for m in valid_keys if m not in models]
        self.config['excluded_model_types'] = models
        return self

    def with_time_limit(self, time_limit: int) -> ConfigBuilder:
        assert time_limit > 0, 'time_limit must be greater than zero'
        self.config['time_limit'] = time_limit
        return self

    def with_hyperparameters(self, hyperparameters: Union[str, dict]) -> ConfigBuilder:
        valid_keys = [m for m in MODEL_TYPES.keys() if m not in ['ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED']]
        valid_str_values = list(hyperparameter_config_dict.keys())
        if isinstance(hyperparameters, str):
            assert hyperparameters in hyperparameter_config_dict, f'{hyperparameters} is not one of the valid presets {valid_str_values}'
        elif isinstance(hyperparameters, dict):
            unknown_keys = [k for k in hyperparameters.keys() if k not in valid_keys]
            assert len(unknown_keys) == 0, f'The following model types are not recognized: {unknown_keys} - use one of the valid models: {valid_keys}'
        else:
            raise ValueError(f'hyperparameters must be either str: {valid_str_values} or dict with keys of {valid_keys}')
        self.config['hyperparameters'] = hyperparameters
        return self

    def with_auto_stack(self, auto_stack=True) -> ConfigBuilder:
        self.config['auto_stack'] = auto_stack
        return self

    def with_use_bag_holdout(self, use_bag_holdout: bool = True) -> ConfigBuilder:
        self.config['use_bag_holdout'] = use_bag_holdout
        return self

    def with_num_bag_folds(self, num_bag_folds: int) -> ConfigBuilder:
        self.config['num_bag_folds'] = num_bag_folds
        return self

    def with_num_bag_sets(self, num_bag_sets: int) -> ConfigBuilder:
        self.config['num_bag_sets'] = num_bag_sets
        return self

    def with_num_stack_levels(self, num_stack_levels: int) -> ConfigBuilder:
        self.config['num_stack_levels'] = num_stack_levels
        return self

    def with_holdout_frac(self, holdout_frac: float) -> ConfigBuilder:
        self.config['holdout_frac'] = holdout_frac
        return self

    def with_use_bag_holdout(self, use_bag_holdout: bool = True) -> ConfigBuilder:
        self.config['use_bag_holdout'] = use_bag_holdout
        return self

    def build(self) -> dict:
        return copy.deepcopy(self.config)
