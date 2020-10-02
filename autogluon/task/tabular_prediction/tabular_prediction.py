import copy
import logging
import math

import numpy as np

from .dataset import TabularDataset
from .hyperparameter_configs import get_hyperparameter_config
from .predictor import TabularPredictor
from .presets_configs import set_presets, unpack
from ..base import BaseTask, compile_scheduler_options
from ..base.base_task import schedulers
from ...utils import verbosity2loglevel
from ...utils.tabular.features.generators import AutoMLPipelineFeatureGenerator
from ...utils.tabular.metrics import get_metric
from ...utils.tabular.ml.learner.default_learner import DefaultLearner as Learner
from ...utils.tabular.ml.trainer.auto_trainer import AutoTrainer
from ...utils.tabular.ml.utils import setup_outputdir, setup_compute, setup_trial_limits, default_holdout_frac

__all__ = ['TabularPrediction']

logger = logging.getLogger()  # return root logger


class TabularPrediction(BaseTask):
    """
    AutoGluon Task for predicting values in column of tabular dataset (classification or regression)
    """

    Dataset = TabularDataset
    Predictor = TabularPredictor

    @staticmethod
    def load(output_directory, verbosity=2):
        """
        Load a predictor object previously produced by `fit()` from file and returns this object.
        It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        Parameters
        ----------
        output_directory : str
            Path to directory where trained models are stored (i.e. the output_directory specified in previous call to `fit`).
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information will be printed by the loaded `Predictor`.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where L ranges from 0 to 50 (Note: higher values L correspond to fewer print statements, opposite of verbosity levels)

        Returns
        -------
        :class:`autogluon.task.tabular_prediction.TabularPredictor` object that can be used to make predictions.
        """
        return TabularPredictor.load(output_directory=output_directory, verbosity=verbosity)

    @staticmethod
    @unpack(set_presets)
    def fit(train_data,
            label,
            tuning_data=None,
            time_limits=None,
            output_directory=None,
            presets=None,
            problem_type=None,
            eval_metric=None,
            stopping_metric=None,
            auto_stack=False,
            hyperparameter_tune=False,
            hyperparameters=None,
            holdout_frac=None,
            num_bagging_folds=0,
            num_bagging_sets=None,
            stack_ensemble_levels=0,
            num_trials=None,
            search_strategy='random',
            verbosity=2,
            **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.

        Parameters
        ----------
        train_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`
            Table of the training data, which is similar to pandas DataFrame.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        label : str
            Name of the column that contains the target variable to predict.
        tuning_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            Another dataset containing validation data reserved for hyperparameter tuning (in same format as training data).
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: final model returned may be fit on this tuning_data as well as train_data. Do not provide your evaluation test data here!
            In particular, when `num_bagging_folds` > 0 or `stack_ensemble_levels` > 0, models will be trained on both `tuning_data` and `train_data`.
            If `tuning_data = None`, `fit()` will automatically hold out some random validation examples from `train_data`.
        time_limits : int, default = None
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until all models have completed training, but will not repeatedly bag models unless `num_bagging_sets` or `auto_stack` is specified.
        output_directory : str, default = None
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `output_directory` locations.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        presets : list or str or dict, default = 'medium_quality_faster_train'
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'`.
            To get good quality with minimal disk usage, set `presets=['good_quality_faster_inference_only_refit', 'optimize_for_deployment']`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/tasks/tabular_prediction/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.

            Available Presets: ['best_quality', 'best_quality_with_high_quality_refit', 'high_quality_fast_inference_only_refit', 'good_quality_faster_inference_only_refit', 'medium_quality_faster_train', 'optimize_for_deployment', 'ignore_text']
            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            In-depth Preset Info:
                best_quality={'auto_stack': True}
                    Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limits value.
                    Recommended for applications that benefit from the best possible model accuracy.

                best_quality_with_high_quality_refit={'auto_stack': True, 'refit_full': True}
                    Identical to best_quality but additionally trains refit_full models that have slightly lower predictive accuracy but are over 10x faster during inference and require 10x less disk space.

                high_quality_fast_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, 'save_bagged_folds': False}
                    High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
                    Recommended for applications that require reasonable inference speed and/or model size.

                good_quality_faster_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, 'save_bagged_folds': False, 'hyperparameters': 'light'}
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

                ignore_text={'feature_generator_kwargs': {'enable_text_ngram_features': False, 'enable_text_special_features': False}}
                    Disables automated feature generation when text features are detected.
                    This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.

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
                'roc_auc', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
                'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
            Options for regression:
                ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2']
            For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

            You can also pass your own evaluation function here as long as it follows formatting of the functions defined in `autogluon/utils/tabular/metrics/`.
        stopping_metric : function or str, default = None
            Metric which iteratively-trained models use to early stop to avoid overfitting.
            `stopping_metric` is not used by weighted ensembles, instead weighted ensembles maximize `eval_metric`.
            Defaults to `eval_metric` value except when `eval_metric='roc_auc'`, where it defaults to `log_loss`.
            Options are identical to options for `eval_metric`.
        auto_stack : bool, default = False
            Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy.
            Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy!
            Note: This overrides `num_bagging_folds` and `stack_ensemble_levels` arguments (selects optimal values for these parameters based on dataset properties).
            Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.
        hyperparameter_tune : bool, default = False
            Whether to tune hyperparameters or just use fixed hyperparameter values for each model. Setting as True will increase `fit()` runtimes.
            It is currently not recommended to use `hyperparameter_tune` with `auto_stack` due to potential overfitting.
            Use `auto_stack` to maximize predictive accuracy; use `hyperparameter_tune` if you prefer to deploy just a single model rather than an ensemble.
        hyperparameters : str or dict, default = 'default'
            Determines the hyperparameters used by the models.
            If `str` is passed, will use a preset hyperparameter configuration.
                Valid `str` options: ['default', 'light', 'very_light', 'toy']
                    'default': Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
                    'light': Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
                    'very_light': Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
                    'toy': Results in extremely small models. Only use this when prototyping, as the model quality will be severely reduced.
                Reference `autogluon/task/tabular_prediction/hyperparameter_configs.py` for information on the hyperparameters associated with each preset.
            Keys are strings that indicate which model types to train.
                Options include: 'NN' (neural network), 'GBM' (lightGBM boosted trees), 'CAT' (CatBoost boosted trees), 'RF' (random forest), 'XT' (extremely randomized trees), 'KNN' (k-nearest neighbors), 'LR' (linear regression)
                If certain key is missing from hyperparameters, then `fit()` will not train any models of that type. Omitting a model key from hyperparameters is equivalent to including this model key in `excluded_model_types`.
                For example, set `hyperparameters = { 'NN':{...} }` if say you only want to train neural networks and no other types of models.
            Values = dict of hyperparameter settings for each model type, or list of dicts.
                Each hyperparameter can either be a single fixed value or a search space containing many possible values.
                Unspecified hyperparameters will be set to default values (or default search spaces if `hyperparameter_tune = True`).
                Caution: Any provided search spaces will be overridden by fixed defaults if `hyperparameter_tune = False`.
                To train multiple models of a given type, set the value to a list of hyperparameter dictionaries.
                    For example, `hyperparameters = {'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}]}` will result in 2 random forest models being trained with separate hyperparameters.
            Advanced functionality: Custom models
                `hyperparameters` can also take a special key 'custom', which maps to a list of model names (currently supported options = 'GBM').
                    If `hyperparameter_tune = False`, then these additional models will also be trained using custom pre-specified hyperparameter settings that are known to work well.
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
                        {'extra_trees': True, 'AG_args': {'name_suffix': 'XT'}},
                    ],
                    'CAT': {},
                    'RF': [
                        {'criterion': 'gini', 'AG_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'AG_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'mse', 'AG_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'XT': [
                        {'criterion': 'gini', 'AG_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'AG_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'mse', 'AG_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'KNN': [
                        {'weights': 'uniform', 'AG_args': {'name_suffix': 'Unif'}},
                        {'weights': 'distance', 'AG_args': {'name_suffix': 'Dist'}},
                    ],
                    'custom': ['GBM']
                }

            Details regarding the hyperparameters you can specify for each model are provided in the following files:
                NN: `autogluon/utils/tabular/ml/models/tabular_nn/hyperparameters/parameters.py`
                    Note: certain hyperparameter settings may cause these neural networks to train much slower.
                GBM: `autogluon/utils/tabular/ml/models/lgb/hyperparameters/parameters.py`
                     See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
                CAT: `autogluon/utils/tabular/ml/models/catboost/hyperparameters/parameters.py`
                     See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html
                RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                LR: `autogluon/utils/tabular/ml/models/lr/hyperparameters/parameters.py`
                    Note: Hyperparameter tuning is disabled for this model.
                    Note: 'penalty' parameter can be used for regression to specify regularization method: 'L1' and 'L2' values are supported.
                Advanced functionality: Custom AutoGluon model arguments
                    These arguments are optional and can be specified in any model's hyperparameters.
                        Example: `hyperparameters = {'RF': {..., 'AG_args': {'name_suffix': 'CustomModelSuffix', 'disable_in_hpo': True}}`
                    AG_args: Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, and the problem types it is valid for.
                        Valid keys:
                            name: (str) The name of the model. This overrides AutoGluon's naming logic and all other name arguments if present.
                            name_main: (str) The main name of the model. In 'RandomForestClassifier', this is 'RandomForest'.
                            name_prefix: (str) Add a custom prefix to the model name. Unused by default.
                            name_type_suffix: (str) Override the type suffix of the model name. In 'RandomForestClassifier', this is 'Classifier'. This comes before 'name_suffix'.
                            name_suffix: (str) Add a custom suffix to the model name. Unused by default.
                            priority: (int) Determines the order in which the model is trained. Larger values result in the model being trained earlier. Default values range from 100 (RF) to 0 (custom), dictated by model type. If you want this model to be trained first, set priority = 999.
                            problem_types: (list) List of valid problem types for the model. `problem_types=['binary']` will result in the model only being trained if `problem_type` is 'binary'.
                            disable_in_hpo: (bool) If True, the model will only be trained if `hyperparameter_tune=False`.
                        Reference the default hyperparameters for example usage of these options.
                    AG_args_fit: Dictionary of model fit customization options related to how and with what constraints the model is trained. These parameters affect stacker fold models, but not stacker models themselves.
                        Clarification: `time_limit` is the internal time in seconds given to a particular model to train, which is dictated in part by the `time_limits` argument given during `fit()` but is not the same.
                        Valid keys:
                            max_memory_usage_ratio: (float, default=1.0) The ratio of memory usage relative to the default to allow before early stopping or killing the model. Values greater than 1.0 will be increasingly prone to out-of-memory errors.
                            max_time_limit_ratio: (float, default=1.0) The ratio of the provided time_limit to use during model `fit()`. If `time_limit=10` and `max_time_limit_ratio=0.3`, time_limit would be changed to 3. Does not alter max_time_limit or min_time_limit values.
                            max_time_limit: (float, default=None) Maximum amount of time to allow this model to train for (in sec). If the provided time_limit is greater than this value, it will be replaced by max_time_limit.
                            min_time_limit: (float, default=0) Allow this model to train for at least this long (in sec), regardless of the time limit it would otherwise be granted.
                                If `min_time_limit >= max_time_limit`, time_limit will be set to min_time_limit.
                                If `min_time_limit=None`, time_limit will be set to None and the model will have no training time restriction.

        holdout_frac : float, default = None
            Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bagging_folds != 0`).
            Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
            Default value is doubled if `hyperparameter_tune = True`, up to a maximum of 0.2.
            Disabled if `num_bagging_folds >= 2`.
        num_bagging_folds : int, default = 0
            Number of folds used for bagging of models. When `num_bagging_folds = k`, training time is roughly increased by a factor of `k` (set = 0 to disable bagging).
            Disabled by default, but we recommend values between 5-10 to maximize predictive performance.
            Increasing num_bagging_folds will result in models with lower bias but that are more prone to overfitting.
            Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
            To further improve predictions, avoid increasing num_bagging_folds much beyond 10 and instead increase num_bagging_sets.
        num_bagging_sets : int, default = None
            Number of repeats of kfold bagging to perform (values must be >= 1). Total number of models trained during bagging = num_bagging_folds * num_bagging_sets.
            Defaults to 1 if time_limits is not specified, otherwise 20 (always disabled if num_bagging_folds is not specified).
            Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
        stack_ensemble_levels : int, default = 0
            Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `stack_ensemble_levels+1` (set = 0 to disable stack ensembling).
            Disabled by default, but we recommend values between 1-3 to maximize predictive performance.
            To prevent overfitting, this argument is ignored unless you have also set `num_bagging_folds >= 2`.
        num_trials : int, default = None
            Maximal number of different hyperparameter settings of each model type to evaluate during HPO (only matters if `hyperparameter_tune = True`).
            If both `time_limits` and `num_trials` are specified, `time_limits` takes precedent.
        search_strategy : str, default = 'random'
            Which hyperparameter search algorithm to use (only matters if `hyperparameter_tune=True`).
            Options include: 'random' (random search), 'bayesopt' (Gaussian process Bayesian optimization), 'skopt' (SKopt Bayesian optimization), 'grid' (grid search).
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed during fit().
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)

        Kwargs can include additional arguments for advanced users:
            AG_args_fit : dict, default={}
                Keyword arguments to pass to all models. See the `AG_args_fit` argument from "Advanced functionality: Custom AutoGluon model arguments" in the `hyperparameters` argument documentation for valid values.
                Identical to specifying `AG_args_fit` parameter for all models in `hyperparameters`.
                If a key in `AG_args_fit` is already specified for a model in `hyperparameters`, it will not be altered through this argument.
            excluded_model_types : list, default = []
                Banned subset of model types to avoid training during `fit()`, even if present in `hyperparameters`.
                Valid values: ['RF', 'XT', 'KNN', 'GBM', 'CAT', 'NN', 'LR', 'custom']. Reference `hyperparameters` documentation for what models correspond to each value.
                Useful when a particular model type such as 'KNN' or 'custom' is not desired but altering the `hyperparameters` dictionary is difficult or time-consuming.
                    Example: To exclude both 'KNN' and 'custom' models, specify `excluded_model_types=['KNN', 'custom']`.
            id_columns : list, default = []
                Banned subset of column names that model may not use as predictive features (e.g. contains label, user-ID, etc).
                These columns are ignored during `fit()`, but DataFrame of just these columns with appended predictions may be produced, for example to submit in a ML competition.
            label_count_threshold : int, default = 10
                For multi-class classification problems, this is the minimum number of times a label must appear in dataset in order to be considered an output class.
                AutoGluon will ignore any classes whose labels do not appear at least this many times in the dataset (i.e. will never predict them).
            save_bagged_folds : bool, default = True
                If True, bagged models will save their fold models (the models from each individual fold of bagging). This is required to use bagged models for prediction after `fit()`.
                If False, bagged models will not save their fold models. This means that bagged models will not be valid models during inference.
                    This should only be set to False when planning to call `predictor.refit_full()` or when `refit_full` is set and `set_best_to_refit_full=True`.
                    Particularly useful if disk usage is a concern. By not saving the fold models, bagged models will use only very small amounts of disk space during training.
                    In many training runs, this will reduce peak disk usage by >10x.
                This parameter has no effect if bagging is disabled.
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
            cache_data : bool, default = True
                When enabled, the training and validation data are saved to disk for future reuse.
                Enables advanced functionality in the resulting Predictor object such as feature importance calculation on the original data.
            refit_full : bool or str, default = False
                Whether to retrain all models on all of the data (training + validation) after the normal training procedure.
                This is equivalent to calling `predictor.refit_full(model=refit_full)` after training.
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
                        The inference speedup factor is equivalent to (k * n), where k is the number of folds (`num_bagging_folds`) and n is the number of finished repeats (`num_bagging_sets`) in the bagged ensemble.
                    The runtime is generally 10% or less of the original fit runtime.
                        The runtime can be roughly estimated as 1 / (k * n) of the original fit runtime, with k and n defined above.
                For non-bagged models:
                    Optimizes a model's accuracy by retraining on 100% of the data without using a validation set.
                    Will typically result in a slight accuracy increase and no change to inference time.
                    The runtime will be approximately equal to the original fit runtime.
                This process does not alter the original models, but instead adds additional models.
                If stacker models are refit by this process, they will use the refit_full versions of the ancestor models during inference.
                Models produced by this process will not have validation scores, as they use all of the data for training.
                    Therefore, it is up to the user to determine if the models are of sufficient quality by including test data in `predictor.leaderboard(dataset=test_data)`.
                    If the user does not have additional test data, they should reference the original model's score for an estimate of the performance of the refit_full model.
                        Warning: Be aware that utilizing refit_full models without separately verifying on test data means that the model is untested, and has no guarantee of being consistent with the original model.
                The time taken by this process is not enforced by `time_limits`.
                `cache_data` must be set to `True` to enable this functionality.
            set_best_to_refit_full : bool, default = False
                If True, will set Trainer.best_model = Trainer.full_model_dict[Trainer.best_model]
                This will change the default model that Predictor uses for prediction when model is not specified to the refit_full version of the model that previously exhibited the highest validation score.
                Only valid if `refit_full` is set.
            feature_generator : `autogluon.utils.tabular.features.generators.abstract.AbstractFeatureGenerator`, default = `autogluon.utils.tabular.features.generators.auto_ml_pipeline.AutoMLPipelineFeatureGenerator()`
                The feature generator used by AutoGluon to process the input data to the form sent to the models. This often includes automated feature generation and data cleaning.
                It is generally recommended to keep the default feature generator unless handling an advanced use-case.
                To control aspects of the default feature generation process, you can pass in an AutoMLPipelineFeatureGenerator object constructed using some of these kwargs:
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
                    vectorizer : CountVectorizer, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
                        sklearn CountVectorizer object to use in TextNgramFeatureGenerator.
                        Only used if `enable_text_ngram_features=True`.
            trainer_type : `Trainer` class, default=`AutoTrainer`
                A class inheriting from `autogluon.utils.tabular.ml.trainer.abstract_trainer.AbstractTrainer` that controls training/ensembling of many models.
                Note: In order to use a custom `Trainer` class, you must import the class file that defines it into the current Python session.
            random_seed : int, default = 0
                Seed to use when generating data split indices such as kfold splits and train/validation splits.
                Caution: This seed only enables reproducible data splits (and the ability to randomize splits in each run by changing seed values).
                This seed is NOT used in the training of individual models, for that you need to explicitly set the corresponding seed hyperparameter (usually called 'seed_value') of each individual model.
                If stacking is enabled:
                    The seed used for stack level L is equal to `seed+L`.
                    This means `random_seed=1` will have the same split indices at L=0 as `random_seed=0` will have at L=1.
                If `random_seed=None`, a random integer is used.
            feature_prune : bool, default = False
                Whether or not to perform feature selection.
            scheduler_options : dict, default = None
                Extra arguments passed to __init__ of scheduler, to configure the orchestration of training jobs during hyperparameter-tuning.
                Ignored if `hyperparameter_tune=False`.
            search_options : dict, default = None
                Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization.
            nthreads_per_trial : int, default = None
                How many CPUs to use in each training run of an individual model.
                This is automatically determined by AutoGluon when left as None (based on available compute).
            ngpus_per_trial : int, default = None
                How many GPUs to use in each trial (ie. single training run of a model).
                This is automatically determined by AutoGluon when left as None.
            dist_ip_addrs : list, default = None
                List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
            visualizer : str, default = None
                How to visualize the neural network training progress during `fit()`. Options: ['mxboard', 'tensorboard', None].

        Returns
        -------
        :class:`autogluon.task.tabular_prediction.TabularPredictor` object which can make predictions on new data and summarize what happened during `fit()`.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        >>> label_column = 'class'
        >>> predictor = task.fit(train_data=train_data, label=label_column)
        >>> test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
        >>> y_test = test_data[label_column]
        >>> test_data = test_data.drop(labels=[label_column], axis=1)
        >>> y_pred = predictor.predict(test_data)
        >>> perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)
        >>> results = predictor.fit_summary()

        To maximize predictive performance, use the following:

        >>> eval_metric = 'roc_auc'  # set this to the metric you ultimately care about
        >>> time_limits = 360  # set as long as you are willing to wait (in sec)
        >>> predictor = task.fit(train_data=train_data, label=label_column, eval_metric=eval_metric, auto_stack=True, time_limits=time_limits)
        """
        assert search_strategy != 'bayesopt_hyperband', \
            "search_strategy == 'bayesopt_hyperband' not yet supported"
        if verbosity < 0:
            verbosity = 0
        elif verbosity > 4:
            verbosity = 4

        logger.setLevel(verbosity2loglevel(verbosity))
        allowed_kwarg_names = {
            'feature_generator',
            'feature_generator_type',  # TODO: Remove on v0.1 release
            'feature_generator_kwargs',  # TODO: Remove on v0.1 release
            'trainer_type',
            'AG_args_fit',
            'excluded_model_types',
            'label_count_threshold',
            'id_columns',
            'set_best_to_refit_full',
            'save_bagged_folds',
            'keep_only_best',
            'save_space',
            'cache_data',
            'refit_full',
            'random_seed',
            'enable_fit_continuation',  # TODO: Remove on v0.1 release
            'feature_prune',
            'scheduler_options',
            'search_options',
            'nthreads_per_trial',
            'ngpus_per_trial',
            'dist_ip_addrs',
            'visualizer',
        }
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                raise ValueError("Unknown keyword argument specified: %s" % kwarg_name)

        # TODO: v0.1 - time_limits -> time_limit?
        # TODO: v0.1 - stack_ensemble_levels -> num_stack_levels / num_stack_layers?
        # TODO: v0.1 - id_columns -> ignored_columns?
        # TODO: v0.1 - nthreads_per_trial/ngpus_per_trial -> rename/rework
        # TODO: v0.1 - visualizer -> consider reworking/removing

        feature_prune = kwargs.get('feature_prune', False)
        scheduler_options = kwargs.get('scheduler_options', None)
        search_options = kwargs.get('search_options', None)
        nthreads_per_trial = kwargs.get('nthreads_per_trial', None)
        ngpus_per_trial = kwargs.get('ngpus_per_trial', None)
        dist_ip_addrs = kwargs.get('dist_ip_addrs', None)
        visualizer = kwargs.get('visualizer', None)

        if isinstance(train_data, str):
            train_data = TabularDataset(file_path=train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(file_path=tuning_data)

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None:
            train_features = np.array([column for column in train_data.columns if column != label])
            tuning_features = np.array([column for column in tuning_data.columns if column != label])
            if np.any(train_features != tuning_features):
                raise ValueError("Column names must match between training and tuning data")

        if feature_prune:
            feature_prune = False  # TODO: Fix feature pruning to add back as an option
            # Currently disabled, needs to be updated to align with new model class functionality
            logger.log(30, 'Warning: feature_prune does not currently work, setting to False.')
        # TODO: Fix or remove in v0.1
        if dist_ip_addrs is not None:
            logger.log(30, 'Warning: dist_ip_addrs does not currently work. Distributed instances will not be utilized.')

        cache_data = kwargs.get('cache_data', True)
        refit_full = kwargs.get('refit_full', False)
        # TODO: Remove on 0.1.0 release
        if 'enable_fit_continuation' in kwargs.keys():
            logger.log(30, 'Warning: `enable_fit_continuation` is a deprecated parameter. It has been renamed to `cache_data`. Starting from AutoGluon 0.1.0, specifying `enable_fit_continuation` as a parameter will cause an exception.')
            logger.log(30, 'Setting `cache_data` value equal to `enable_fit_continuation` value.')
            cache_data = kwargs['enable_fit_continuation']
        if not cache_data:
            logger.log(30, 'Warning: `cache_data=False` will disable or limit advanced functionality after training such as feature importance calculations. It is recommended to set `cache_data=True` unless you explicitly wish to not have the data saved to disk.')
            if refit_full:
                raise ValueError('`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')

        set_best_to_refit_full = kwargs.get('set_best_to_refit_full', False)
        if set_best_to_refit_full and not refit_full:
            raise ValueError('`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        save_bagged_folds = kwargs.get('save_bagged_folds', True)

        if hyperparameter_tune:
            logger.log(30, 'Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')

        if dist_ip_addrs is None:
            dist_ip_addrs = []

        if search_options is None:
            search_options = dict()

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        # Process kwargs to create feature generator, trainer, schedulers, searchers for each model:
        output_directory = setup_outputdir(output_directory)  # Format directory name
        if 'feature_generator_type' in kwargs or 'feature_generator_kwargs' in kwargs:
            logger.log(30, 'Warning: `feature_generator_type` and `feature_generator_kwargs` are deprecated arguments. Use `feature_generator` instead. Starting from AutoGluon 0.1.0, specifying these arguments will cause an exception.')
            feature_generator_type = kwargs.get('feature_generator_type', AutoMLPipelineFeatureGenerator)
            feature_generator_kwargs = kwargs.get('feature_generator_kwargs', {})
            feature_generator = feature_generator_type(**feature_generator_kwargs)  # instantiate FeatureGenerator object
        else:
            feature_generator = kwargs.get('feature_generator', AutoMLPipelineFeatureGenerator())

        id_columns = kwargs.get('id_columns', [])
        trainer_type = kwargs.get('trainer_type', AutoTrainer)
        ag_args_fit = kwargs.get('AG_args_fit', {})
        excluded_model_types = kwargs.get('excluded_model_types', [])
        random_seed = kwargs.get('random_seed', 0)
        nthreads_per_trial, ngpus_per_trial = setup_compute(nthreads_per_trial, ngpus_per_trial)
        num_train_rows = len(train_data)
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limits? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            num_bagging_folds = min(10, max(5, math.floor(num_train_rows / 100)))
            stack_ensemble_levels = min(1, max(0, math.floor(num_train_rows / 750)))

        if num_bagging_sets is None:
            if num_bagging_folds >= 2:
                if time_limits is not None:
                    num_bagging_sets = 20
                else:
                    num_bagging_sets = 1
            else:
                num_bagging_sets = 1

        label_count_threshold = kwargs.get('label_count_threshold', 10)
        if num_bagging_folds is not None:  # Ensure there exist sufficient labels for stratified splits across all bags
            label_count_threshold = max(label_count_threshold, num_bagging_folds)

        time_limits_orig = copy.deepcopy(time_limits)
        time_limits_hpo = copy.deepcopy(time_limits)

        if num_bagging_folds >= 2 and (time_limits_hpo is not None):
            time_limits_hpo = time_limits_hpo / (1 + num_bagging_folds * (1 + stack_ensemble_levels))
        # FIXME: Incorrect if user specifies custom level-based hyperparameter config!
        time_limits_hpo, num_trials = setup_trial_limits(time_limits_hpo, num_trials, hyperparameters)  # TODO: Move HPO time allocation to Trainer

        if (num_trials is not None) and hyperparameter_tune and (num_trials == 1):
            hyperparameter_tune = False
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limits is too small for hyperparameter_tune, setting to False.')

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(num_train_rows, hyperparameter_tune)

        # Add visualizer to NN hyperparameters:
        if (visualizer is not None) and (visualizer != 'none') and ('NN' in hyperparameters):
            hyperparameters['NN']['visualizer'] = visualizer

        eval_metric = get_metric(eval_metric, problem_type, 'eval_metric')
        stopping_metric = get_metric(stopping_metric, problem_type, 'stopping_metric')

        # All models use the same scheduler:
        scheduler_options = compile_scheduler_options(
            scheduler_options=scheduler_options,
            search_strategy=search_strategy,
            search_options=search_options,
            nthreads_per_trial=nthreads_per_trial,
            ngpus_per_trial=ngpus_per_trial,
            checkpoint=None,
            num_trials=num_trials,
            time_out=time_limits_hpo,
            resume=False,
            visualizer=visualizer,
            time_attr='epoch',
            reward_attr='validation_performance',
            dist_ip_addrs=dist_ip_addrs)
        scheduler_cls = schedulers[search_strategy.lower()]
        scheduler_options = (scheduler_cls, scheduler_options)  # wrap into tuple
        learner = Learner(path_context=output_directory, label=label, problem_type=problem_type, eval_metric=eval_metric, stopping_metric=stopping_metric,
                          id_columns=id_columns, feature_generator=feature_generator, trainer_type=trainer_type,
                          label_count_threshold=label_count_threshold, random_seed=random_seed)
        learner.fit(X=train_data, X_val=tuning_data, scheduler_options=scheduler_options,
                    hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                    holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
                    hyperparameters=hyperparameters, ag_args_fit=ag_args_fit, excluded_model_types=excluded_model_types, time_limit=time_limits_orig, save_data=cache_data, save_bagged_folds=save_bagged_folds, verbosity=verbosity)

        predictor = TabularPredictor(learner=learner)

        keep_only_best = kwargs.get('keep_only_best', False)
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
            trainer = predictor._trainer
            trainer_model_best = trainer.get_model_best()
            predictor.refit_full(model=refit_full)
            if set_best_to_refit_full:
                if trainer_model_best in trainer.model_full_dict.keys():
                    trainer.model_best = trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    trainer.save()
                else:
                    logger.warning(f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

        if keep_only_best:
            predictor.delete_models(models_to_keep='best', dry_run=False)

        save_space = kwargs.get('save_space', False)
        if save_space:
            predictor.save_space()

        return predictor
