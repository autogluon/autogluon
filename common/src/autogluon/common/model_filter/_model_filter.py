import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelFilter:
    """Class to filter models given user requirements"""

    @staticmethod
    def include_models(models: Union[Dict[str, Any], List[str]], included_model_types: List[str]) -> Union[Dict[str, Any], List[str]]:
        """
        Only include models specified in `included_model_types`, other models will be removed
        If model specified in `included_model_types` doesn't present in `models`, will warn users and ignore

        Parameters
        ----------
        models: Union[Dict[str, Any], List[str]]
            A dictionary containing models and their hyperparameters
        included_model_types: List[str]
            List of model types to be included

        Return
        ------
        Union[Dict[str, Any], List[str]]
            Updated dictionary or list with correct models
        """
        if isinstance(models, dict):
            included_models = {model: val for model, val in models.items() if model in included_model_types}
            missing_models = set(included_model_types) - set(included_models.keys())
        elif isinstance(models, list):
            included_models = [model for model in models if model in included_model_types]
            missing_models = set(included_model_types) - set(included_models)
        if included_model_types is not None:
            logger.log(20, f"Included models: {list(included_model_types)} (Specified by `included_model_types`, all other model types will be skipped)")
        if len(missing_models) > 0:
            logger.warning(f"\tThe models types {list(missing_models)} are not present in the model list specified by the user and will be ignored:")
        return included_models

    @staticmethod
    def exclude_models(models: Union[Dict[str, Any], List[str]], excluded_model_types: List[str]) -> Union[Dict[str, Any], List[str]]:
        """
        Exclude models from the current dictionary.
        All models specified in `excluded_model_types` will be removed

        Parameters
        ----------
        models: Union[Dict[str, Any], List[str]]
            A dictionary containing models and their hyperparameters
        excluded_model_types: List[str]
            List of model types to be excluded

        Return
        ------
        Union[Dict[str, Any], List[str]]
            Updated dictionary or list with correct models
        """
        excluded_models = None
        if isinstance(models, dict):
            models_after_exclusion = {model: val for model, val in models.items() if model not in excluded_model_types}
            excluded_models = set(models.keys()) - set(models_after_exclusion.keys())
        elif isinstance(models, list):
            models_after_exclusion = [model for model in models if model not in excluded_model_types]
            excluded_models = set(models) - set(models_after_exclusion)
        if excluded_models is not None:
            logger.log(20, f"Excluded models: {list(excluded_models)} (Specified by `excluded_model_types`)")
        return models_after_exclusion

    @staticmethod
    def filter_models(
        models: Union[Dict[str, Any], List[str]],
        included_model_types: Optional[List[str]] = None,
        excluded_model_types: Optional[List[str]] = None,
    ) -> Union[Dict[str, Any], List[str]]:
        """
        Filter models given `included_model_types` or `excluded_model_types`
        If both are provided, will only respect `included_model_types`

        Parameters
        ----------
        models: Union[Dict[str, Any], List[str]]
            A dictionary containing models and their hyperparameters
        included_model_types: List[str]
            List of model types to be included
        excluded_model_types: List[str]
            List of model types to be excluded

        Return
        ------
        Union[Dict[str, Any], List[str]]
            Updated dictionary or list with correct models
        """
        if included_model_types is not None and excluded_model_types is not None:
            logger.warning("Both `included_model_types` and `excluded_model_types` are specified. Will use `included_model_types` only")
            excluded_model_types = None
        if included_model_types is not None:
            return ModelFilter.include_models(models=models, included_model_types=included_model_types)
        if excluded_model_types is not None:
            return ModelFilter.exclude_models(models=models, excluded_model_types=excluded_model_types)
        return models
