import copy
import logging
import math
import pprint
import time

import numpy as np
import pandas as pd

from autogluon.core.dataset import TabularDataset
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import setup_outputdir, setup_compute, setup_trial_limits, default_holdout_frac

from ..configs.hyperparameter_configs import get_hyperparameter_config
from ..configs.presets_configs import set_presets, unpack
from ..configs.feature_generator_presets import get_default_feature_generator
from ..task.tabular_prediction.predictor_legacy import TabularPredictorV1
from ..learner import AbstractLearner, DefaultLearner
from ..trainer import AbstractTrainer

logger = logging.getLogger()  # return root logger

# TODO: Add generic documentation to hyperparameter_tune_kwargs
# TODO: num_cpus/num_gpus -> ag_args_fit
# TODO: num_bag_sets -> ag_args
# TODO: make core_kwargs a kwargs argument to predictor.fit,
# TODO: HPO in fit_extra, HPO via ag_args, per model.
# TODO: Document predictor attributes

# Extra TODOs (Stretch): Can occur post v0.1
# TODO: add aux_kwargs to predictor.fit
# TODO: add pip freeze + python version output after fit + log file, validate that same pip freeze on load as cached
# TODO: predictor.clone()
# TODO: Add logging comments that models are serialized on disk after fit
# TODO: consider adding kwarg option for data which has already been preprocessed by feature generator to skip feature generation.
# TODO: Resolve raw text feature usage in default feature generator

# Done for Tabular
# TODO: Remove all `time_limits` in project, replace with `time_limit`


class TabularPredictor(TabularPredictorV1):
    """
    AutoGluon Predictor predicts values in a column of a tabular dataset (classification or regression).

    Parameters
    ----------
    label : str
        Name of the column that contains the target variable to predict.
    problem_type : str, default = None
        Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
        If `problem_type = None`, the prediction problem type is inferred based on the label-values in provided dataset.
    eval_metric : function or str, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification and 'root_mean_squared_error' for regression.
        Otherwise, options for classification:
            ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro', 'precision_micro',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
        Options for regression:
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2']
        For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

        You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
    path : str, default = None
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
    **kwargs :
        learner_type : AbstractLearner, default = DefaultLearner
            A class which inherits from `AbstractLearner`. This dictates the inner logic of predictor.
            If you don't know what this is, keep it as the default.
        learner_kwargs : dict, default = None
            Kwargs to send to the learner. Options include:

            ignored_columns : list, default = None
                Banned subset of column names that predictor may not use as predictive features (e.g. unique identifier to a row or user-ID).
                These columns are ignored during `fit()`.
            label_count_threshold : int, default = 10
                For multi-class classification problems, this is the minimum number of times a label must appear in dataset in order to be considered an output class.
                AutoGluon will ignore any classes whose labels do not appear at least this many times in the dataset (i.e. will never predict them).
            cache_data : bool, default = True
                When enabled, the training and validation data are saved to disk for future reuse.
                Enables advanced functionality in predictor such as `fit_extra()` and feature importance calculation on the original data.
            trainer_type : AbstractTrainer, default = AutoTrainer
                A class inheriting from `AbstractTrainer` that controls training/ensembling of many models.
    """
    predictor_file_name = 'predictor.pkl'

    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=2,
            **kwargs
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path)

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())

        self._learner: AbstractLearner = learner_type(path_context=path, label=label, feature_generator=None,
                                                      eval_metric=eval_metric, problem_type=problem_type, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    @property
    def path(self):
        return self._learner.path

    @unpack(set_presets)
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            feature_metadata='infer',
            **kwargs):
        """
        Fit models to predict a column of a data table (label) based on the other columns (features).

        Parameters
        ----------
        train_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            Table of the training data, which is similar to a pandas DataFrame.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        tuning_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`, default = None
            Another dataset containing validation data reserved for tuning processes such as early stopping and hyperparameter tuning.
            This dataset should be in the same format as `train_data`.
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: final model returned may be fit on `tuning_data` as well as `train_data`. Do not provide your evaluation test data here!
            In particular, when `num_bag_folds` > 0 or `num_stack_levels` > 0, models will be trained on both `tuning_data` and `train_data`.
            If `tuning_data = None`, `fit()` will automatically hold out some random validation examples from `train_data`.
        time_limit : int, default = None
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until all models have completed training, but will not repeatedly bag models unless `num_bag_sets` is specified.
        presets : list or str or dict, default = ['medium_quality_faster_train']
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'`.
            To get good quality with minimal disk usage, set `presets=['good_quality_faster_inference_only_refit', 'optimize_for_deployment']`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/tabular/configs/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.

            Available Presets: ['best_quality', 'high_quality_fast_inference_only_refit', 'good_quality_faster_inference_only_refit', 'medium_quality_faster_train', 'optimize_for_deployment', 'ignore_text']
            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            In-depth Preset Info:
                best_quality={'auto_stack': True}
                    Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
                    Recommended for applications that benefit from the best possible model accuracy.

                high_quality_fast_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, '_save_bag_folds': False}
                    High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
                    Recommended for applications that require reasonable inference speed and/or model size.

                good_quality_faster_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, '_save_bag_folds': False, 'hyperparameters': 'light'}
                    Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality_fast_inference_only_refit`.
                    Recommended for applications that require fast inference speed.

                medium_quality_faster_train={'auto_stack': False}
                    Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality_faster_inference_only_refit`.
                    This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality_faster_inference_only_refit` results in significantly better predictive accuracy and faster inference time.

                optimize_for_deployment={'keep_only_best': True, 'save_space': True}
                    Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
                    Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
                    This will disable numerous advanced functionality, but has no impact on inference.
                    This will make certain functionality less informative, such as `predictor.leaderboard()` and `predictor.fit_summary()`.
                        Because unused models will be deleted under this preset, methods like `predictor.leaderboard()` and `predictor.fit_summary()` will no longer show the full set of models that were trained during `fit()`.
                    Recommended for applications where the inner details of AutoGluon's training is not important and there is no intention of manually choosing between the final models.
                    This preset pairs well with the other presets such as `good_quality_faster_inference_only_refit` to make a very compact final model.
                    Identical to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` and `predictor.save_space()` directly after `fit()`.

                ignore_text={'_feature_generator_kwargs': {'enable_text_ngram_features': False, 'enable_text_special_features': False, 'enable_raw_text_features': False}}
                    Disables automated feature generation when text features are detected.
                    This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.
                    Ignored if `feature_generator` was also specified.

        hyperparameters : str or dict, default = 'default'
            Determines the hyperparameters used by the models.
            If `str` is passed, will use a preset hyperparameter configuration.
                Valid `str` options: ['default', 'light', 'very_light', 'toy']
                    'default': Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
                    'light': Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
                    'very_light': Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
                    'toy': Results in extremely small models. Only use this when prototyping, as the model quality will be severely reduced.
                Reference `autogluon/tabular/configs/hyperparameter_configs.py` for information on the hyperparameters associated with each preset.
            Keys are strings that indicate which model types to train.
                Stable model options include:
                    'GBM' (LightGBM)
                    'CAT' (CatBoost)
                    'XGB' (XGBoost)
                    'RF' (random forest)
                    'XT' (extremely randomized trees)
                    'KNN' (k-nearest neighbors)
                    'LR' (linear regression)
                    'NN' (neural network with MXNet backend)
                    'FASTAI' (neural network with FastAI backend)
                Experimental model options include:
                    'FASTTEXT' (FastText)
                    'TEXT_NN_V1' (Multimodal Text+Tabular model, GPU is required)
                    'TRANSF' (Tabular Transformer, GPU is recommended)
                If a certain key is missing from hyperparameters, then `fit()` will not train any models of that type. Omitting a model key from hyperparameters is equivalent to including this model key in `excluded_model_types`.
                For example, set `hyperparameters = { 'NN':{...} }` if say you only want to train neural networks and no other types of models.
            Values = dict of hyperparameter settings for each model type, or list of dicts.
                Each hyperparameter can either be a single fixed value or a search space containing many possible values.
                Unspecified hyperparameters will be set to default values (or default search spaces if `hyperparameter_tune = True`).
                Caution: Any provided search spaces will be overridden by fixed defaults if `hyperparameter_tune = False`.
                To train multiple models of a given type, set the value to a list of hyperparameter dictionaries.
                    For example, `hyperparameters = {'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}]}` will result in 2 random forest models being trained with separate hyperparameters.
            Advanced functionality: Custom models
                `hyperparameters` can also take a special key 'custom', which maps to a list of model names (currently supported options = 'GBM').
                    If `hyperparameter_tune_kwargs = None`, then these additional models will also be trained using custom pre-specified hyperparameter settings that are known to work well.
            Advanced functionality: Custom stack levels
                By default, AutoGluon re-uses the same models and model hyperparameters at each level during stack ensembling.
                To customize this behaviour, create a hyperparameters dictionary separately for each stack level, and then add them as values to a new dictionary, with keys equal to the stack level.
                    Example: `hyperparameters = {0: {'RF': rf_params1}, 1: {'CAT': [cat_params1, cat_params2], 'NN': {}}}`
                    This will result in a stack ensemble that has one custom random forest in level 0 followed by two CatBoost models with custom hyperparameters and a default neural network in level 1, for a total of 4 models.
                If a level is not specified in `hyperparameters`, it will default to using the highest specified level to train models. This can also be explicitly controlled by adding a 'default' key.

            Default:
                hyperparameters = {
                    'NN': {},
                    'GBM': [
                        {},
                        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                    ],
                    'CAT': {},
                    'XGB': {},
                    'FASTAI': {},
                    'RF': [
                        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'XT': [
                        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'KNN': [
                        {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
                        {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
                    ],
                    'custom': ['GBM']
                }

            Details regarding the hyperparameters you can specify for each model are provided in the following files:
                NN: `autogluon.tabular.models.tabular_nn.hyperparameters.parameters`
                    Note: certain hyperparameter settings may cause these neural networks to train much slower.
                GBM: `autogluon.tabular.models.lgb.hyperparameters.parameters`
                     See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
                CAT: `autogluon.tabular.models.catboost.hyperparameters.parameters`
                     See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html
                XGB: `autogluon.tabular.models.xgboost.hyperparameters.parameters`
                     See also the XGBoost docs: https://xgboost.readthedocs.io/en/latest/parameter.html
                FASTAI: `autogluon.tabular.models.fastainn.hyperparameters.parameters`
                     See also the FastAI docs: https://docs.fast.ai/tabular.models.html
                RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                LR: `autogluon.tabular.models.lr.hyperparameters.parameters`
                    Note: Hyperparameter tuning is disabled for this model.
                    Note: 'penalty' parameter can be used for regression to specify regularization method: 'L1' and 'L2' values are supported.
                Advanced functionality: Custom AutoGluon model arguments
                    These arguments are optional and can be specified in any model's hyperparameters.
                        Example: `hyperparameters = {'RF': {..., 'ag_args': {'name_suffix': 'CustomModelSuffix', 'disable_in_hpo': True}}`
                    ag_args: Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, and the problem types it is valid for.
                        Valid keys:
                            name: (str) The name of the model. This overrides AutoGluon's naming logic and all other name arguments if present.
                            name_main: (str) The main name of the model. Example: 'RandomForest'.
                            name_prefix: (str) Add a custom prefix to the model name. Unused by default.
                            name_suffix: (str) Add a custom suffix to the model name. Unused by default.
                            priority: (int) Determines the order in which the model is trained. Larger values result in the model being trained earlier. Default values range from 100 (RF) to 0 (custom), dictated by model type. If you want this model to be trained first, set priority = 999.
                            problem_types: (list) List of valid problem types for the model. `problem_types=['binary']` will result in the model only being trained if `problem_type` is 'binary'.
                            disable_in_hpo: (bool) If True, the model will only be trained if `hyperparameter_tune=False`.
                            valid_stacker: (bool) If False, the model will not be trained as a level 1 or higher stacker model.
                            valid_base: (bool) If False, the model will not be trained as a level 0 (base) model.
                        Reference the default hyperparameters for example usage of these options.
                    ag_args_fit: Dictionary of model fit customization options related to how and with what constraints the model is trained. These parameters affect stacker fold models, but not stacker models themselves.
                        Clarification: `time_limit` is the internal time in seconds given to a particular model to train, which is dictated in part by the `time_limit` argument given during `fit()` but is not the same.
                        Valid keys:
                            max_memory_usage_ratio: (float, default=1.0) The ratio of memory usage relative to the default to allow before early stopping or killing the model. Values greater than 1.0 will be increasingly prone to out-of-memory errors.
                            max_time_limit_ratio: (float, default=1.0) The ratio of the provided time_limit to use during model `fit()`. If `time_limit=10` and `max_time_limit_ratio=0.3`, time_limit would be changed to 3. Does not alter max_time_limit or min_time_limit values.
                            max_time_limit: (float, default=None) Maximum amount of time to allow this model to train for (in sec). If the provided time_limit is greater than this value, it will be replaced by max_time_limit.
                            min_time_limit: (float, default=0) Allow this model to train for at least this long (in sec), regardless of the time limit it would otherwise be granted.
                                If `min_time_limit >= max_time_limit`, time_limit will be set to min_time_limit.
                                If `min_time_limit=None`, time_limit will be set to None and the model will have no training time restriction.
                    ag_args_ensemble: Dictionary of hyperparameters shared by all models that control how they are ensembled. Only models in stack levels >=1 are impacted by these parameters.
                        Valid keys:
                            use_orig_features: (bool) Whether a stack model will use the original features along with the stack features to train (akin to skip-connections). If the model has no stack features (no base models), this value is ignored and the stack model will use the original features.
                            max_base_models: (int, default=25) Maximum number of base models whose predictions form the features input to this stacker model. If more than `max_base_models` base models are available, only the top `max_base_models` models with highest validation score are used.
                            max_base_models_per_type: (int, default=5) Similar to `max_base_models`. If more than `max_base_models_per_type` of any particular model type are available, only the top `max_base_models_per_type` of that type are used. This occurs before the `max_base_models` filter.
                            save_bag_folds: (bool, default=True)
                                If True, bagged models will save their fold models (the models from each individual fold of bagging). This is required to use bagged models for prediction.
                                If False, bagged models will not save their fold models. This means that bagged models will not be valid models during inference.
                                    This should only be set to False when planning to call `predictor.refit_full()` or when `refit_full` is set and `set_best_to_refit_full=True`.
                                    Particularly useful if disk usage is a concern. By not saving the fold models, bagged models will use only very small amounts of disk space during training.
                                    In many training runs, this will reduce peak disk usage by >10x.

        feature_metadata : :class:`autogluon.tabular.FeatureMetadata` or str, default = 'infer'
            The feature metadata used in various inner logic in feature preprocessing.
            If 'infer', will automatically construct a FeatureMetadata object based on the properties of `train_data`.
            In this case, `train_data` is input into :meth:`autogluon.tabular.FeatureMetadata.from_df` to infer `feature_metadata`.
            If 'infer' incorrectly assumes the dtypes of features, consider explicitly specifying `feature_metadata`.
        **kwargs :
            auto_stack : bool, default = False
                Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy.
                Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy!
                Automatically sets `num_bag_folds` and `num_stack_levels` arguments based on dataset properties.
                Note: Setting `num_bag_folds` and `num_stack_levels` arguments will override `auto_stack`.
                Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.
            num_bag_folds : int, default = None
                Number of folds used for bagging of models. When `num_bag_folds = k`, training time is roughly increased by a factor of `k` (set = 0 to disable bagging).
                Disabled by default (0), but we recommend values between 5-10 to maximize predictive performance.
                Increasing num_bag_folds will result in models with lower bias but that are more prone to overfitting.
                `num_bag_folds = 1` is an invalid value, and will raise a ValueError.
                Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
                To further improve predictions, avoid increasing `num_bag_folds` much beyond 10 and instead increase `num_bag_sets`.
            num_bag_sets : int, default = None
                Number of repeats of kfold bagging to perform (values must be >= 1). Total number of models trained during bagging = `num_bag_folds * num_bag_sets`.
                Defaults to 1 if `time_limit` is not specified, otherwise 20 (always disabled if `num_bag_folds` is not specified).
                Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
            num_stack_levels : int, default = None
                Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `num_stack_levels+1` (set = 0 to disable stack ensembling).
                Disabled by default (0), but we recommend values between 1-3 to maximize predictive performance.
                To prevent overfitting, `num_bag_folds >= 2` must also be set or else a ValueError will be raised.
            holdout_frac : float, default = None
                Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bag_folds != 0`).
                Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
                Default value is doubled if `hyperparameter_tune_kwargs` is set, up to a maximum of 0.2.
                Disabled if `num_bag_folds >= 2`.
            hyperparameter_tune_kwargs : str or dict, default = None
                Hyperparameter tuning strategy and kwargs.
                If None, then hyperparameter tuning will not be performed.
                Valid preset values:
                    'auto': Uses the 'random' preset.
                    'random': Performs HPO via random search.
                    'bayesopt': Performs HPO via bayesian optimization.
                For valid dictionary keys, refer to :class:`autogluon.core.scheduler.FIFOScheduler` documentation.
                    The 'searcher' key is required when providing a dict.
            ag_args : dict, default = None
                Keyword arguments to pass to all models (i.e. common hyperparameters shared by all AutoGluon models).
                See the `ag_args` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
                Identical to specifying `ag_args` parameter for all models in `hyperparameters`.
                If a key in `ag_args` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
            ag_args_fit : dict, default = None
                Keyword arguments to pass to all models.
                See the `ag_args_fit` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
                Identical to specifying `ag_args_fit` parameter for all models in `hyperparameters`.
                If a key in `ag_args_fit` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
            ag_args_ensemble : dict, default = None
                Keyword arguments to pass to all models.
                See the `ag_args_ensemble` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
                Identical to specifying `ag_args_ensemble` parameter for all models in `hyperparameters`.
                If a key in `ag_args_ensemble` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
            excluded_model_types : list, default = None
                Banned subset of model types to avoid training during `fit()`, even if present in `hyperparameters`.
                Reference `hyperparameters` documentation for what models correspond to each value.
                Useful when a particular model type such as 'KNN' or 'custom' is not desired but altering the `hyperparameters` dictionary is difficult or time-consuming.
                    Example: To exclude both 'KNN' and 'custom' models, specify `excluded_model_types=['KNN', 'custom']`.
            refit_full : bool or str, default = False
                Whether to retrain all models on all of the data (training + validation) after the normal training procedure.
                This is equivalent to calling `predictor.refit_full(model=refit_full)` after fit.
                If `refit_full=True`, it will be treated as `refit_full='all'`.
                If `refit_full=False`, refitting will not occur.
                Valid str values:
                    `all`: refits all models.
                    `best`: refits only the best model (and its ancestors if it is a stacker model).
                    `{model_name}`: refits only the specified model (and its ancestors if it is a stacker model).
                For bagged models:
                    Reduces a model's inference time by collapsing bagged ensembles into a single model fit on all of the training data.
                    This process will typically result in a slight accuracy reduction and a large inference speedup.
                    The inference speedup will generally be between 10-200x faster than the original bagged ensemble model.
                        The inference speedup factor is equivalent to (k * n), where k is the number of folds (`num_bag_folds`) and n is the number of finished repeats (`num_bag_sets`) in the bagged ensemble.
                    The runtime is generally 10% or less of the original fit runtime.
                        The runtime can be roughly estimated as 1 / (k * n) of the original fit runtime, with k and n defined above.
                For non-bagged models:
                    Optimizes a model's accuracy by retraining on 100% of the data without using a validation set.
                    Will typically result in a slight accuracy increase and no change to inference time.
                    The runtime will be approximately equal to the original fit runtime.
                This process does not alter the original models, but instead adds additional models.
                If stacker models are refit by this process, they will use the refit_full versions of the ancestor models during inference.
                Models produced by this process will not have validation scores, as they use all of the data for training.
                    Therefore, it is up to the user to determine if the models are of sufficient quality by including test data in `predictor.leaderboard(test_data)`.
                    If the user does not have additional test data, they should reference the original model's score for an estimate of the performance of the refit_full model.
                        Warning: Be aware that utilizing refit_full models without separately verifying on test data means that the model is untested, and has no guarantee of being consistent with the original model.
                The time taken by this process is not enforced by `time_limit`.
            set_best_to_refit_full : bool, default = False
                If True, will change the default model that Predictor uses for prediction when model is not specified to the refit_full version of the model that exhibited the highest validation score.
                Only valid if `refit_full` is set.
            keep_only_best : bool, default = False
                If True, only the best model and its ancestor models are saved in the outputted `predictor`. All other models are deleted.
                    If you only care about deploying the most accurate predictor with the smallest file-size and no longer need any of the other trained models or functionality beyond prediction on new data, then set: `keep_only_best=True`, `save_space=True`.
                    This is equivalent to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` directly after `fit()`.
                If used with `refit_full` and `set_best_to_refit_full`, the best model will be the refit_full model, and the original bagged best model will be deleted.
                    `refit_full` will be automatically set to 'best' in this case to avoid training models which will be later deleted.
            save_space : bool, default = False
                If True, reduces the memory and disk size of predictor by deleting auxiliary model files that aren't needed for prediction on new data.
                    This is equivalent to calling `predictor.save_space()` directly after `fit()`.
                This has NO impact on inference accuracy.
                It is recommended if the only goal is to use the trained model for prediction.
                Certain advanced functionality may no longer be available if `save_space=True`. Refer to `predictor.save_space()` documentation for more details.
            num_cpus : int, default = 'auto'
                How many CPUs to use during fit.
                If 'auto', will use all available CPUs.
            num_gpus : int, default = 'auto'
                How many GPUs to use during fit.
                If 'auto', will use all available GPUs.
                Set to 0 to disable usage of GPUs.
            feature_generator : :class:`autogluon.tabular.features.generators.AbstractFeatureGenerator`, default = :class:`autogluon.tabular.features.generators.AutoMLPipelineFeatureGenerator`
                The feature generator used by AutoGluon to process the input data to the form sent to the models. This often includes automated feature generation and data cleaning.
                It is generally recommended to keep the default feature generator unless handling an advanced use-case.
                To control aspects of the default feature generation process, you can pass in an :class:`AutoMLPipelineFeatureGenerator` object constructed using some of these kwargs:
                    enable_numeric_features : bool, default True
                        Whether to keep features of 'int' and 'float' raw types.
                        These features are passed without alteration to the models.
                        Appends IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=['int', 'float']))) to the generator group.
                    enable_categorical_features : bool, default True
                        Whether to keep features of 'object' and 'category' raw types.
                        These features are processed into memory optimized 'category' features.
                        Appends CategoryFeatureGenerator() to the generator group.
                    enable_datetime_features : bool, default True
                        Whether to keep features of 'datetime' raw type and 'object' features identified as 'datetime_as_object' features.
                        These features will be converted to 'int' features representing milliseconds since epoch.
                        Appends DatetimeFeatureGenerator() to the generator group.
                    enable_text_special_features : bool, default True
                        Whether to use 'object' features identified as 'text' features to generate 'text_special' features such as word count, capital letter ratio, and symbol counts.
                        Appends TextSpecialFeatureGenerator() to the generator group.
                    enable_text_ngram_features : bool, default True
                        Whether to use 'object' features identified as 'text' features to generate 'text_ngram' features.
                        Appends TextNgramFeatureGenerator(vectorizer=vectorizer) to the generator group.
                    enable_raw_text_features : bool, default False
                        Whether to keep the raw text features.
                        Appends IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['text'])) to the generator group.
                    vectorizer : CountVectorizer, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
                        sklearn CountVectorizer object to use in TextNgramFeatureGenerator.
                        Only used if `enable_text_ngram_features=True`.
            unlabeled_data : pd.DataFrame, default = None
                [Experimental Parameter]
                Collection of data without labels that we can use to pretrain on. This is the same schema as train_data, except
                without the labels. Currently, unlabeled_data is only used for pretraining a TabTransformer model.
                If you do not specify 'TRANSF' with unlabeled_data, then no pretraining will occur and unlabeled_data will be ignored!
                After the pretraining step, we will finetune using the TabTransformer model as well. If TabTransformer is ensembled
                with other models, like in typical AutoGluon fashion, then the output of this "pretrain/finetune" will be ensembled
                with other models, which will not used the unlabeled_data. The "pretrain/finetune flow" is also known as semi-supervised learning.
                The typical use case for unlabeled_data is to add signal to your model where you may not have sufficient training
                data. e.g. 500 hand-labeled samples (perhaps a hard human task), whole data set (unlabeled) is thousands/millions.
                However, this isn't the only use case. Given enough unlabeled data(millions of rows), you may see improvements
                to any amount of labeled data.
            verbosity : int
                If specified, overrides the existing `predictor.verbosity` value.

        Returns
        -------
        :class:`TabularPredictor` object. Returns self.

        Examples
        --------
        >>> from autogluon.tabular import TabularDataset, TabularPredictor
        >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        >>> label = 'class'
        >>> predictor = TabularPredictor(label=label).fit(train_data)
        >>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
        >>> leaderboard = predictor.leaderboard(test_data)
        >>> y_test = test_data[label]
        >>> test_data = test_data.drop(columns=[label])
        >>> y_pred = predictor.predict(test_data)
        >>> perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)

        To maximize predictive performance, use the following:

        >>> eval_metric = 'roc_auc'  # set this to the metric you ultimately care about
        >>> time_limit = 3600  # set as long as you are willing to wait (in sec)
        >>> predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_data, presets=['best_quality'], time_limit=time_limit)
        """
        if self._learner.is_fit:
            raise AssertionError('Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new `Predictor`.')
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)

        if presets:
            if not isinstance(presets, list):
                presets = [presets]
            logger.log(20, f'Presets specified: {presets}')

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        holdout_frac = kwargs['holdout_frac']
        num_bag_folds = kwargs['num_bag_folds']
        num_bag_sets = kwargs['num_bag_sets']
        num_stack_levels = kwargs['num_stack_levels']
        auto_stack = kwargs['auto_stack']
        hyperparameter_tune_kwargs = kwargs['hyperparameter_tune_kwargs']
        num_cpus = kwargs['num_cpus']
        num_gpus = kwargs['num_gpus']
        feature_generator = kwargs['feature_generator']
        unlabeled_data = kwargs['unlabeled_data']

        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']

        feature_generator_init_kwargs = kwargs['_feature_generator_kwargs']
        if feature_generator_init_kwargs is None:
            feature_generator_init_kwargs = dict()

        train_data, tuning_data, unlabeled_data = self._validate_fit_data(train_data=train_data, tuning_data=tuning_data, unlabeled_data=unlabeled_data)

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        ###################################
        # FIXME: v0.1 This section is a hack
        if 'enable_raw_text_features' not in feature_generator_init_kwargs:
            if 'TEXT_NN_V1' in hyperparameters:
                feature_generator_init_kwargs['enable_raw_text_features'] = True
            else:
                for key in hyperparameters:
                    if isinstance(key, int) or key == 'default':
                        if 'TEXT_NN_V1' in hyperparameters[key]:
                            feature_generator_init_kwargs['enable_raw_text_features'] = True
                            break
        ###################################

        if feature_metadata is not None and isinstance(feature_metadata, str) and feature_metadata == 'infer':
            feature_metadata = None
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=feature_generator_init_kwargs)

        # Process kwargs to create trainer, schedulers, searchers:
        num_bag_folds, num_bag_sets, num_stack_levels = self._sanitize_stack_args(
            num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets, num_stack_levels=num_stack_levels,
            time_limit=time_limit, auto_stack=auto_stack, num_train_rows=len(train_data),
        )

        if hyperparameter_tune_kwargs is not None:
            scheduler_options = self._init_scheduler_tabular(hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, num_bag_folds, num_stack_levels)
        else:
            scheduler_options = None
        hyperparameter_tune = scheduler_options is not None
        if hyperparameter_tune:
            logger.log(30, 'Warning: hyperparameter tuning is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(len(train_data), hyperparameter_tune)

        if ag_args_fit is None:
            ag_args_fit = dict()
        if 'num_cpus' not in ag_args_fit and num_cpus != 'auto':
            ag_args_fit['num_cpus'] = num_cpus
        if 'num_gpus' not in ag_args_fit and num_gpus != 'auto':
            ag_args_fit['num_gpus'] = num_gpus

        if kwargs['_save_bag_folds'] is not None:
            if ag_args_ensemble is None:
                ag_args_ensemble = {}
            ag_args_ensemble['save_bag_folds'] = kwargs['_save_bag_folds']

        core_kwargs = {'ag_args': ag_args, 'ag_args_ensemble': ag_args_ensemble, 'ag_args_fit': ag_args_fit, 'excluded_model_types': excluded_model_types}
        self._learner.fit(X=train_data, X_val=tuning_data, X_unlabeled=unlabeled_data,
                          hyperparameter_tune_kwargs=scheduler_options,
                          holdout_frac=holdout_frac, num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets, num_stack_levels=num_stack_levels,
                          hyperparameters=hyperparameters, core_kwargs=core_kwargs,
                          time_limit=time_limit, verbosity=verbosity)
        self._set_post_fit_vars()

        self._post_fit(
            keep_only_best=kwargs['keep_only_best'],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs['set_best_to_refit_full'],
            save_space=kwargs['save_space'],
        )
        self.save()
        return self

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False, save_space=False):
        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = 'best'
                else:
                    logger.warning(f'refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted.')
                    refit_full = False
            else:
                refit_full = 'all'

        if refit_full is not False:
            trainer_model_best = self._trainer.get_model_best()
            self.refit_full(model=refit_full)
            if set_best_to_refit_full:
                if trainer_model_best in self._trainer.model_full_dict.keys():
                    self._trainer.model_best = self._trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    self._trainer.save()
                else:
                    logger.warning(f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

        if keep_only_best:
            self.delete_models(models_to_keep='best', dry_run=False)

        if save_space:
            self.save_space()

    def fit_extra(self, hyperparameters, time_limit=None, base_model_names=None, **kwargs):
        """
        Fits additional models after the original :meth:`TabularPredictor.fit` call.
        The original train_data and tuning_data will be used to train the models.

        Parameters
        ----------
        hyperparameters : str or dict
            Refer to argument documentation in :meth:`TabularPredictor.fit`.
            If `base_model_names` is specified and hyperparameters is using the level-based key notation,
            the key of the level which directly uses the base models should be 0. The level in the hyperparameters
            dictionary is relative, not absolute.
        time_limit : int, default = None
            Refer to argument documentation in :meth:`TabularPredictor.fit`.
        base_model_names : list, default = None
            The names of the models to use as base models for this fit call.
            Base models will provide their out-of-fold predictions as additional features to the models in `hyperparameters`.
            If specified, all models trained will be stack ensembles.
            If None, models will be trained as if they were specified in :meth:`TabularPredictor.fit`, without depending on existing models.
            Only valid if bagging is enabled.
        **kwargs :
            Refer to kwargs documentation in :meth:`TabularPredictor.fit`.
            Note that the following kwargs are not available in `fit_extra` as they cannot be changed from their values set in `fit()`:
                [`holdout_frac`, `num_bag_folds`, `auto_stack`, `feature_generator`, `unlabeled_data`]
        """
        time_start = time.time()

        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_extra_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        # TODO: Allow disable aux (default to disabled)
        # TODO: num_bag_sets
        # num_bag_sets = kwargs['num_bag_sets']
        num_stack_levels = kwargs['num_stack_levels']
        hyperparameter_tune_kwargs = kwargs['hyperparameter_tune_kwargs']
        num_cpus = kwargs['num_cpus']
        num_gpus = kwargs['num_gpus']
        # save_bag_folds = kwargs['save_bag_folds']  # TODO: Enable

        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']

        fit_new_weighted_ensemble = False  # TODO v0.1: Add as option
        aux_kwargs = None  # TODO v0.1: Add as option

        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        if num_stack_levels is None:
            hyperparameter_keys = list(hyperparameters.keys())
            highest_level = 0
            for key in hyperparameter_keys:
                if isinstance(key, int):
                    highest_level = max(key, highest_level)
            num_stack_levels = highest_level

        if hyperparameter_tune_kwargs is not None:
            scheduler_options = self._init_scheduler_tabular(hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, self._trainer.k_fold, num_stack_levels)
        else:
            scheduler_options = None
        hyperparameter_tune = scheduler_options is not None
        if hyperparameter_tune:
            raise ValueError('Hyperparameter Tuning is not allowed in `fit_extra`.')  # FIXME: Change this
            # logger.log(30, 'Warning: hyperparameter tuning is currently experimental and may cause the process to hang.')
        if ag_args_fit is None:
            ag_args_fit = dict()
        if 'num_cpus' not in ag_args_fit and num_cpus != 'auto':
            ag_args_fit['num_cpus'] = num_cpus
        if 'num_gpus' not in ag_args_fit and num_gpus != 'auto':
            ag_args_fit['num_gpus'] = num_gpus

        # TODO: v0.1: make core_kwargs a kwargs argument to predictor.fit, add aux_kwargs to predictor.fit
        core_kwargs = {'ag_args': ag_args, 'ag_args_ensemble': ag_args_ensemble, 'ag_args_fit': ag_args_fit, 'excluded_model_types': excluded_model_types}

        # TODO: Add special error message if called and training/val data was not cached.
        X_train, y_train, X_val, y_val = self._trainer.load_data()
        fit_models = self._trainer.train_multi_levels(
            X_train=X_train, y_train=y_train, hyperparameters=hyperparameters, X_val=X_val, y_val=y_val,
            base_model_names=base_model_names, time_limit=time_limit, relative_stack=True, level_end=num_stack_levels,
            core_kwargs=core_kwargs, aux_kwargs=aux_kwargs
        )

        if time_limit is not None:
            time_limit = time_limit - (time.time() - time_start)

        if fit_new_weighted_ensemble:
            if time_limit is not None:
                time_limit_weighted = max(time_limit, 60)
            else:
                time_limit_weighted = None
            fit_models += self.fit_weighted_ensemble(time_limit=time_limit_weighted)

        self._post_fit(
            keep_only_best=kwargs['keep_only_best'],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs['set_best_to_refit_full'],
            save_space=kwargs['save_space'],
        )
        self.save()
        return self

    def _init_scheduler_tabular(self, hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, num_bag_folds, num_stack_levels):
        num_cpus, num_gpus = setup_compute(num_cpus, num_gpus)  # TODO: use 'auto' downstream
        time_limit_hpo = time_limit
        if num_bag_folds >= 2 and (time_limit_hpo is not None):
            time_limit_hpo = time_limit_hpo / (1 + num_bag_folds * (1 + num_stack_levels))
        # FIXME: Incorrect if user specifies custom level-based hyperparameter config!
        time_limit_hpo, num_trials = setup_trial_limits(time_limit_hpo, None, hyperparameters)  # TODO: Move HPO time allocation to Trainer
        if time_limit is not None:
            time_limit_hpo = None
        if hyperparameter_tune_kwargs is None:
            return None
        scheduler_options = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, time_out=time_limit_hpo, num_trials=num_trials, nthreads_per_trial=num_cpus, ngpus_per_trial=num_gpus)

        assert scheduler_options[1]['searcher'] != 'bayesopt_hyperband', "searcher == 'bayesopt_hyperband' not yet supported"
        # TODO: Fix or remove in v0.1
        if scheduler_options[1].get('dist_ip_addrs', None):
            logger.log(30, 'Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.')

        if scheduler_options[1]['num_trials'] == 1:
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limit is too small for hyperparameter_tune, disabling HPO.')
            return None
        scheduler_ngpus = scheduler_options[1]['resource'].get('num_gpus', 0)
        if scheduler_ngpus is not None and isinstance(scheduler_ngpus, int) and scheduler_ngpus > 1:
            scheduler_options[1]['resource']['num_gpus'] = 1
            logger.warning("Warning: TabularPredictor currently doesn't use >1 GPU per training run. ngpus_per_trial set = 1")

        return scheduler_options

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._learner.persist_trainer(low_memory=True)
            self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object

    # TODO: Update and correct the logging message on loading directions
    def save(self):
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        super().save()
        self._learner = None
        self._trainer = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer

    @classmethod
    def load(cls, path, verbosity=2):
        set_logger_verbosity(verbosity, logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        predictor: TabularPredictor = load_pkl.load(path=path + cls.predictor_file_name)
        learner = predictor._learner_type.load(path)
        predictor._set_post_fit_vars(learner=learner)
        try:
            from ...version import __version__
            version_inference = __version__
        except:
            version_inference = None
        # TODO: v0.1 Move version var to predictor object in the case where learner does not exist
        try:
            version_fit = predictor._learner.version
        except:
            version_fit = None
        if version_fit is None:
            version_fit = 'Unknown (Likely <=0.0.11)'
        if version_inference != version_fit:
            logger.warning('')
            logger.warning('############################## WARNING ##############################')
            logger.warning('WARNING: AutoGluon version differs from the version used during the original model fit! This may lead to instability and it is highly recommended the model be loaded with the exact AutoGluon version it was fit with.')
            logger.warning(f'\tFit Version:     {version_fit}')
            logger.warning(f'\tCurrent Version: {version_inference}')
            logger.warning('############################## WARNING ##############################')
            logger.warning('')

        return predictor

    @classmethod
    def from_learner(cls, learner: AbstractLearner):
        predictor = cls(label=learner.label, path=learner.path)
        predictor._set_post_fit_vars(learner=learner)
        return predictor

    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'learner_type',
            'learner_kwargs',
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f'Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}')

    def _validate_fit_kwargs(self, kwargs):

        # TODO:
        #  Valid core_kwargs values:
        #  ag_args, ag_args_fit, ag_args_ensemble, stack_name, ensemble_type, name_suffix, time_limit
        #  Valid aux_kwargs values:
        #  name_suffix, time_limit, stack_name, aux_hyperparameters, ag_args, ag_args_ensemble

        # TODO: Remove features from models option for fit_extra
        # TODO: Constructor?
        fit_kwargs_default = dict(
            # data split / ensemble architecture kwargs -> Don't nest but have nested documentation -> Actually do nesting
            holdout_frac=None,  # TODO: Potentially error if num_bag_folds is also specified
            num_bag_folds=None,  # TODO: Potentially move to fit_extra, raise exception if value too large / invalid in fit_extra.
            auto_stack=False,

            # other
            feature_generator='auto',
            unlabeled_data=None,

            _feature_generator_kwargs=None,
        )

        kwargs = self._validate_fit_extra_kwargs(kwargs, extra_valid_keys=list(fit_kwargs_default.keys()))

        kwargs_sanitized = fit_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        return kwargs_sanitized

    def _validate_fit_extra_kwargs(self, kwargs, extra_valid_keys=None):
        fit_extra_kwargs_default = dict(
            # data split / ensemble architecture kwargs -> Don't nest but have nested documentation -> Actually do nesting
            num_bag_sets=None,
            num_stack_levels=None,

            hyperparameter_tune_kwargs=None,

            # core_kwargs -> +1 nest
            ag_args=None,
            ag_args_fit=None,
            ag_args_ensemble=None,
            excluded_model_types=None,

            # aux_kwargs -> +1 nest

            # post_fit_kwargs -> +1 nest
            set_best_to_refit_full=False,
            keep_only_best=False,
            save_space=False,
            refit_full=False,

            # move into ag_args_fit? +1
            num_cpus='auto',
            num_gpus='auto',

            # other
            verbosity=self.verbosity,

            # private
            _save_bag_folds=None,
        )

        allowed_kwarg_names = list(fit_extra_kwargs_default.keys())
        if extra_valid_keys is not None:
            allowed_kwarg_names += extra_valid_keys
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                public_kwarg_options = [kwarg for kwarg in allowed_kwarg_names if kwarg[0] != '_']
                public_kwarg_options.sort()
                raise ValueError(f"Unknown keyword argument specified: {kwarg_name}\nValid kwargs: {public_kwarg_options}")

        kwargs_sanitized = fit_extra_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        # Deepcopy args to avoid altering outer context
        deepcopy_args = ['ag_args', 'ag_args_fit', 'ag_args_ensemble', 'excluded_model_types']
        for deepcopy_arg in deepcopy_args:
            kwargs_sanitized[deepcopy_arg] = copy.deepcopy(kwargs_sanitized[deepcopy_arg])

        refit_full = kwargs_sanitized['refit_full']
        set_best_to_refit_full = kwargs_sanitized['set_best_to_refit_full']
        if refit_full and not self._learner.cache_data:
            raise ValueError('`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')
        if set_best_to_refit_full and not refit_full:
            raise ValueError('`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        return kwargs_sanitized

    def _validate_fit_data(self, train_data, tuning_data=None, unlabeled_data=None):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        if unlabeled_data is not None and isinstance(unlabeled_data, str):
            unlabeled_data = TabularDataset(unlabeled_data)

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None:
            train_features = np.array([column for column in train_data.columns if column != self.label_column])
            tuning_features = np.array([column for column in tuning_data.columns if column != self.label_column])
            if np.any(train_features != tuning_features):
                raise ValueError("Column names must match between training and tuning data")
        if unlabeled_data is not None:
            train_features = sorted(np.array([column for column in train_data.columns if column != self.label_column]))
            unlabeled_features = sorted(np.array([column for column in unlabeled_data.columns]))
            if np.any(train_features != unlabeled_features):
                raise ValueError("Column names must match between training and unlabeled data.\n"
                                 "Unlabeled data must have not the label column specified in it.\n")
        return train_data, tuning_data, unlabeled_data

    def _set_feature_generator(self, feature_generator='auto', feature_metadata=None, init_kwargs=None):
        if self._learner.feature_generator is not None:
            if isinstance(feature_generator, str) and feature_generator == 'auto':
                feature_generator = self._learner.feature_generator
            else:
                raise AssertionError('FeatureGenerator already exists!')
        self._learner.feature_generator = get_default_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=init_kwargs)

    def _sanitize_stack_args(self, num_bag_folds, num_bag_sets, num_stack_levels, time_limit, auto_stack, num_train_rows):
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limit? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            if num_bag_folds is None:
                num_bag_folds = min(10, max(5, math.floor(num_train_rows / 100)))
            if num_stack_levels is None:
                num_stack_levels = min(1, max(0, math.floor(num_train_rows / 750)))
        if num_bag_folds is None:
            num_bag_folds = 0
        if num_stack_levels is None:
            num_stack_levels = 0
        if not isinstance(num_bag_folds, int):
            raise ValueError(f'num_bag_folds must be an integer. (num_bag_folds={num_bag_folds})')
        if not isinstance(num_stack_levels, int):
            raise ValueError(f'num_stack_levels must be an integer. (num_stack_levels={num_stack_levels})')
        if num_bag_folds < 2 and num_bag_folds != 0:
            raise ValueError(f'num_bag_folds must be equal to 0 or >=2. (num_bag_folds={num_bag_folds})')
        if num_stack_levels != 0 and num_bag_folds == 0:
            raise ValueError(f'num_stack_levels must be 0 if num_bag_folds is 0. (num_stack_levels={num_stack_levels}, num_bag_folds={num_bag_folds})')
        if num_bag_sets is None:
            if num_bag_folds >= 2:
                if time_limit is not None:
                    num_bag_sets = 20  # TODO: v0.1 Reduce to 5 or 3 as 20 is unnecessarily extreme as a default.
                else:
                    num_bag_sets = 1
            else:
                num_bag_sets = 1
        if not isinstance(num_bag_sets, int):
            raise ValueError(f'num_bag_sets must be an integer. (num_bag_sets={num_bag_sets})')
        return num_bag_folds, num_bag_sets, num_stack_levels


# Location to store WIP functionality that will be later added to TabularPredictor
class _TabularPredictorExperimental(TabularPredictor):
    # TODO: Documentation, flesh out capabilities
    # TODO: Rename feature_generator -> feature_pipeline for users?
    # TODO: Return transformed data?
    # TODO: feature_generator_kwargs?
    def fit_feature_generator(self, data: pd.DataFrame, feature_generator='auto', feature_metadata=None):
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata)
        self._learner.fit_transform_features(data)

    # TODO: rename to `advice`
    # TODO: Add documentation
    def _advice(self):
        is_feature_generator_fit = self._learner.feature_generator.is_fit()
        is_learner_fit = self._learner.trainer_path is not None
        exists_trainer = self._trainer is not None

        advice_dict = dict(
            is_feature_generator_fit=is_feature_generator_fit,
            is_learner_fit=is_learner_fit,
            exists_trainer=exists_trainer,
            # TODO
        )

        advice_list = []

        if not advice_dict['is_feature_generator_fit']:
            advice_list.append('FeatureGenerator has not been fit, consider calling `predictor.fit_feature_generator(data)`.')
        if not advice_dict['is_learner_fit']:
            advice_list.append('Learner is not fit, consider calling `predictor.fit(...)`')
        if not advice_dict['exists_trainer']:
            advice_list.append('Trainer is not initialized, consider calling `predictor.fit(...)`')
        # TODO: Advice on unused features (if no model uses a feature)
        # TODO: Advice on fit_extra
        # TODO: Advice on distill
        # TODO: Advice on leaderboard
        # TODO: Advice on persist
        # TODO: Advice on refit_full
        # TODO: Advice on feature_importance
        # TODO: Advice on dropping poor models

        logger.log(20, '======================= AutoGluon Advice =======================')
        if advice_list:
            for advice in advice_list:
                logger.log(20, advice)
        else:
            logger.log(20, 'No further advice found.')
        logger.log(20, '================================================================')
