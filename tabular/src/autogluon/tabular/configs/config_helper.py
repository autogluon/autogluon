from __future__ import annotations

import copy
from typing import Union

from sklearn.base import BaseEstimator

from autogluon.core.scheduler import scheduler_factory
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from autogluon.tabular.registry import ag_model_registry


class FeatureGeneratorBuilder:
    def __init__(self, parent=None):
        self.parent = parent
        self.config = {}

    def enable_numeric_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to keep features of 'int' and 'float' raw types.
        These features are passed without alteration to the models.
        Appends IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=['int', 'float']))) to the generator group.
        """
        self.config["enable_numeric_features"] = value
        return self

    def enable_categorical_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to keep features of 'object' and 'category' raw types.
        These features are processed into memory optimized 'category' features.
        Appends CategoryFeatureGenerator() to the generator group.
        """
        self.config["enable_categorical_features"] = value
        return self

    def enable_datetime_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to keep features of 'datetime' raw type and 'object' features identified as 'datetime_as_object' features.
        These features will be converted to 'int' features representing milliseconds since epoch.
        Appends DatetimeFeatureGenerator() to the generator group.
        """
        self.config["enable_datetime_features"] = value
        return self

    def enable_text_special_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to use 'object' features identified as 'text' features to generate 'text_special' features such as word count, capital letter ratio, and symbol counts.
        Appends TextSpecialFeatureGenerator() to the generator group.
        """
        self.config["enable_text_special_features"] = value
        return self

    def enable_text_ngram_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to use 'object' features identified as 'text' features to generate 'text_ngram' features.
        Appends TextNgramFeatureGenerator(vectorizer=vectorizer) to the generator group.
        """
        self.config["enable_text_ngram_features"] = value
        return self

    def enable_raw_text_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Whether to keep the raw text features.
        Appends IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['text'])) to the generator group.
        """
        self.config["enable_raw_text_features"] = value
        return self

    def enable_vision_features(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        [Experimental]
        Whether to keep 'object' features identified as 'image_path' special type. Features of this form should have a string path to an image file as their value.
        Only vision models can leverage these features, and these features will not be treated as categorical.
        Note: 'image_path' features will not be automatically inferred. These features must be explicitly specified as such in a custom FeatureMetadata object.
        Note: It is recommended that the string paths use absolute paths rather than relative, as it will likely be more stable.
        """
        self.config["enable_vision_features"] = value
        return self

    def vectorizer(self, value: BaseEstimator) -> FeatureGeneratorBuilder:
        """
        sklearn CountVectorizer object to use in TextNgramFeatureGenerator.
        Only used if `enable_text_ngram_features=True`.
        """
        self.config["vectorizer"] = value
        return self

    def text_ngram_params(self, value: bool = True) -> FeatureGeneratorBuilder:
        """
        Appends TextNgramFeatureGenerator(vectorizer=vectorizer, text_ngram_params) to the generator group. See text_ngram.py for valid parameters.
        """
        self.config["text_ngram_params"] = value
        return self

    def build(self) -> Union[ConfigBuilder, AutoMLPipelineFeatureGenerator]:
        generator = AutoMLPipelineFeatureGenerator(**self.config)
        if self.parent:
            self.parent.config["feature_generator"] = generator
            return self.parent
        else:
            return generator


class ConfigBuilder:
    def __init__(self):
        self.config = {}

    def _valid_keys(self):
        valid_keys = [m for m in ag_model_registry.keys if m not in ["ENS_WEIGHTED", "SIMPLE_ENS_WEIGHTED"]]
        return valid_keys

    def presets(self, presets: Union[str, list, dict]) -> ConfigBuilder:
        """
        List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned `predictor`.
        It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
        Available Presets: ['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'ignore_text']
        It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.
        If there is an overlap in presets keys, the latter presets will override the earlier ones.
        """
        valid_keys = list(tabular_presets_dict.keys())
        if isinstance(presets, str):
            presets = [presets]

        if isinstance(presets, list):
            unknown_keys = [k for k in presets if k not in valid_keys]
            assert len(unknown_keys) == 0, f"The following presets are not recognized: {unknown_keys} - use one of the valid presets: {valid_keys}"

        self.config["presets"] = presets
        return self

    def time_limit(self, time_limit: int) -> ConfigBuilder:
        """
        Approximately how long `fit()` should run for (wallclock time in seconds).
        If not specified, `fit()` will run until all models have completed training, but will not repeatedly bag models unless `num_bag_sets` is specified.
        """
        if time_limit is not None:
            assert time_limit > 0, "time_limit must be greater than zero"
        self.config["time_limit"] = time_limit
        return self

    def hyperparameters(self, hyperparameters: Union[str, dict]) -> ConfigBuilder:
        valid_keys = self._valid_keys()
        valid_str_values = list(hyperparameter_config_dict.keys())
        if isinstance(hyperparameters, str):
            assert hyperparameters in hyperparameter_config_dict, f"{hyperparameters} is not one of the valid presets {valid_str_values}"
        elif isinstance(hyperparameters, dict):
            unknown_keys = [k for k in hyperparameters.keys() if isinstance(k, str) and (k not in valid_keys)]
            assert len(unknown_keys) == 0, f"The following model types are not recognized: {unknown_keys} - use one of the valid models: {valid_keys}"
        else:
            raise ValueError(f"hyperparameters must be either str: {valid_str_values} or dict with keys of {valid_keys}")
        self.config["hyperparameters"] = hyperparameters
        return self

    def auto_stack(self, auto_stack: bool = True) -> ConfigBuilder:
        """
        Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy.
        Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy!
        Automatically sets `num_bag_folds` and `num_stack_levels` arguments based on dataset properties.
        Note: Setting `num_bag_folds` and `num_stack_levels` arguments will override `auto_stack`.
        Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.
        """
        self.config["auto_stack"] = auto_stack
        return self

    def num_bag_folds(self, num_bag_folds: int) -> ConfigBuilder:
        """
        Number of folds used for bagging of models. When `num_bag_folds = k`, training time is roughly increased by a factor of `k` (set = 0 to disable bagging).
        Disabled by default (0), but we recommend values between 5-10 to maximize predictive performance.
        Increasing num_bag_folds will result in models with lower bias but that are more prone to overfitting.
        `num_bag_folds = 1` is an invalid value, and will raise a ValueError.
        Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
        To further improve predictions, avoid increasing `num_bag_folds` much beyond 10 and instead increase `num_bag_sets`.
        """
        assert num_bag_folds >= 0, "num_bag_folds must be greater or equal than zero"
        self.config["num_bag_folds"] = num_bag_folds
        return self

    def num_bag_sets(self, num_bag_sets: int) -> ConfigBuilder:
        """
        Number of repeats of kfold bagging to perform (values must be >= 1). Total number of models trained during bagging = `num_bag_folds * num_bag_sets`.
        Defaults to 1 if `time_limit` is not specified, otherwise 20 (always disabled if `num_bag_folds` is not specified).
        Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
        """
        assert num_bag_sets > 0, "num_bag_sets must be greater than zero"
        self.config["num_bag_sets"] = num_bag_sets
        return self

    def num_stack_levels(self, num_stack_levels: int) -> ConfigBuilder:
        """
        Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `num_stack_levels+1` (set = 0 to disable stack ensembling).
        Disabled by default (0), but we recommend values between 1-3 to maximize predictive performance.
        To prevent overfitting, `num_bag_folds >= 2` must also be set or else a ValueError will be raised.
        """
        assert num_stack_levels >= 0, "num_stack_levels must be greater or equal than zero"
        self.config["num_stack_levels"] = num_stack_levels
        return self

    def holdout_frac(self, holdout_frac: float) -> ConfigBuilder:
        """
        Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bag_folds != 0` unless `use_bag_holdout == True`).
        Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
        Default value is doubled if `hyperparameter_tune_kwargs` is set, up to a maximum of 0.2.
        Disabled if `num_bag_folds >= 2` unless `use_bag_holdout == True`.
        """
        assert (holdout_frac >= 0) & (holdout_frac <= 1), "holdout_frac must be between 0 and 1"
        self.config["holdout_frac"] = holdout_frac
        return self

    def use_bag_holdout(self, use_bag_holdout: bool = True) -> ConfigBuilder:
        """
        If True, a `holdout_frac` portion of the data is held-out from model bagging.
        This held-out data is only used to score models and determine weighted ensemble weights.
        Enable this if there is a large gap between score_val and score_test in stack models.
        Note: If `tuning_data` was specified, `tuning_data` is used as the holdout data.
        Disabled if not bagging.
        """
        self.config["use_bag_holdout"] = use_bag_holdout
        return self

    def hyperparameter_tune_kwargs(self, hyperparameter_tune_kwargs: Union[str, dict]) -> ConfigBuilder:
        """
        Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
        If None, then hyperparameter tuning will not be performed.
        Valid preset values:
            'auto': Uses the 'random' preset.
            'random': Performs HPO via random search using local scheduler.
        The 'searcher' key is required when providing a dict.
        """
        valid_str_values = scheduler_factory._scheduler_presets.keys()
        if isinstance(hyperparameter_tune_kwargs, str):
            assert hyperparameter_tune_kwargs in valid_str_values, f"{hyperparameter_tune_kwargs} string must be one of {valid_str_values}"
        elif not isinstance(hyperparameter_tune_kwargs, dict):
            raise ValueError(f"hyperparameter_tune_kwargs must be either str: {valid_str_values} or dict")
        self.config["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

        return self

    def ag_args(self, ag_args: dict) -> ConfigBuilder:
        """
        Keyword arguments to pass to all models (i.e. common hyperparameters shared by all AutoGluon models).
        See the `ag_args` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
        Identical to specifying `ag_args` parameter for all models in `hyperparameters`.
        If a key in `ag_args` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
        """
        self.config["ag_args"] = ag_args
        return self

    def ag_args_fit(self, ag_args_fit: dict) -> ConfigBuilder:
        """
        Keyword arguments to pass to all models.
        See the `ag_args_fit` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
        Identical to specifying `ag_args_fit` parameter for all models in `hyperparameters`.
        If a key in `ag_args_fit` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
        """
        self.config["ag_args_fit"] = ag_args_fit
        return self

    def ag_args_ensemble(self, ag_args_ensemble: dict) -> ConfigBuilder:
        """
        Keyword arguments to pass to all models.
        See the `ag_args_ensemble` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
        Identical to specifying `ag_args_ensemble` parameter for all models in `hyperparameters`.
        If a key in `ag_args_ensemble` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
        """
        self.config["ag_args_ensemble"] = ag_args_ensemble
        return self

    def excluded_model_types(self, models: Union[str, list]) -> ConfigBuilder:
        """
        Banned subset of model types to avoid training during `fit()`, even if present in `hyperparameters`.
        Reference `hyperparameters` documentation for what models correspond to each value.
        Useful when a particular model type such as 'KNN' or 'custom' is not desired but altering the `hyperparameters` dictionary is difficult or time-consuming.
            Example: To exclude both 'KNN' and 'custom' models, specify `excluded_model_types=['KNN', 'custom']`.
        """
        valid_keys = self._valid_keys()
        if not isinstance(models, list):
            models = [models]
        for model in models:
            assert model in valid_keys, f"{model} is not one of the valid models {valid_keys}"
        self.config["excluded_model_types"] = sorted(list(set(models)))
        return self

    def included_model_types(self, models: Union[str, list]) -> ConfigBuilder:
        """
        Subset of model types to train during `fit()`.
        Reference `hyperparameters` documentation for what models correspond to each value.
        Useful when only the particular models should be trained such as 'KNN' or 'custom', but altering the `hyperparameters` dictionary is difficult or time-consuming.
            Example: To keep only 'KNN' and 'custom' models, specify `included_model_types=['KNN', 'custom']`.
        """
        valid_keys = self._valid_keys()
        if not isinstance(models, list):
            models = [models]

        unknown_keys = [k for k in models if isinstance(k, str) and (k not in valid_keys)]
        assert len(unknown_keys) == 0, f"The following model types are not recognized: {unknown_keys} - use one of the valid models: {valid_keys}"

        models = [m for m in valid_keys if m not in models]
        self.config["excluded_model_types"] = models
        return self

    def refit_full(self, refit_full: Union[bool, str] = True) -> ConfigBuilder:
        """
        Whether to retrain all models on all of the data (training + validation) after the normal training procedure.
        This is equivalent to calling `predictor.refit_full(model=refit_full)` after fit.
        If `refit_full=True`, it will be treated as `refit_full='all'`.
        If `refit_full=False`, refitting will not occur.
        Valid str values:
            `all`: refits all models.
            `best`: refits only the best model (and its ancestors if it is a stacker model).
            `{model_name}`: refits only the specified model (and its ancestors if it is a stacker model).
        """
        self.config["refit_full"] = refit_full
        return self

    def set_best_to_refit_full(self, set_best_to_refit_full=True) -> ConfigBuilder:
        """
        If True, will change the default model that Predictor uses for prediction when model is not specified to the refit_full version of the model that exhibited the highest validation score.
        Only valid if `refit_full` is set.
        """
        self.config["set_best_to_refit_full"] = set_best_to_refit_full
        return self

    def keep_only_best(self, keep_only_best=True) -> ConfigBuilder:
        """
        If True, only the best model and its ancestor models are saved in the outputted `predictor`. All other models are deleted.
            If you only care about deploying the most accurate predictor with the smallest file-size and no longer need any of the other trained models or functionality beyond prediction on new data, then set: `keep_only_best=True`, `save_space=True`.
            This is equivalent to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` directly after `fit()`.
        If used with `refit_full` and `set_best_to_refit_full`, the best model will be the refit_full model, and the original bagged best model will be deleted.
            `refit_full` will be automatically set to 'best' in this case to avoid training models which will be later deleted.
        """
        self.config["keep_only_best"] = keep_only_best
        return self

    def save_space(self, save_space=True) -> ConfigBuilder:
        """
        If True, reduces the memory and disk size of predictor by deleting auxiliary model files that aren't needed for prediction on new data.
            This is equivalent to calling `predictor.save_space()` directly after `fit()`.
        This has NO impact on inference accuracy.
        It is recommended if the only goal is to use the trained model for prediction.
        Certain advanced functionality may no longer be available if `save_space=True`. Refer to `predictor.save_space()` documentation for more details.
        """
        self.config["save_space"] = save_space
        return self

    def feature_generator(self) -> FeatureGeneratorBuilder:
        """
        The feature generator used by AutoGluon to process the input data to the form sent to the models. This often includes automated feature generation and data cleaning.
        It is generally recommended to keep the default feature generator unless handling an advanced use-case.
        """
        return FeatureGeneratorBuilder(self)

    def calibrate(self, calibrate=True) -> ConfigBuilder:
        """
        If True and the problem_type is classification, temperature scaling will be used to calibrate the Predictor's estimated class probabilities
        (which may improve metrics like log_loss) and will train a scalar parameter on the validation set.
        If True and the problem_type is quantile regression, conformalization will be used to calibrate the Predictor's estimated quantiles
        (which may improve the prediction interval coverage, and bagging could further improve it) and will compute a set of scalar parameters on the validation set.
        """
        self.config["calibrate"] = calibrate
        return self

    def build(self) -> dict:
        """
        Build the config.
        """
        return copy.deepcopy(self.config)
