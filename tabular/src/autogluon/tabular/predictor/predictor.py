import copy
import inspect
import logging
import math
import os
import pprint
import time
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.calibrate.temperature_scaling import tune_temperature_scaling
from autogluon.core.calibrate.conformity_score import compute_conformity_score
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, AUTO_WEIGHT, BALANCE_WEIGHT, PSEUDO_MODEL_SUFFIX, PROBLEM_TYPES_CLASSIFICATION
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.dataset import TabularDataset
from autogluon.core.pseudolabeling.pseudolabeling import filter_pseudo, filter_ensemble_pseudo
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.trainer import AbstractTrainer
from autogluon.core.utils import get_pred_from_proba_df, try_import_torch
from autogluon.core.utils import plot_performance_vs_trials, plot_summary_of_models, plot_tabular_models
from autogluon.core.utils.decorators import apply_presets
from autogluon.tabular.models import _IModelsModel

from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.core.utils.utils import default_holdout_frac

from ..configs.feature_generator_presets import get_default_feature_generator
from ..configs.hyperparameter_configs import get_hyperparameter_config
from ..configs.presets_configs import tabular_presets_dict, tabular_presets_alias
from ..learner import AbstractTabularLearner, DefaultLearner

logger = logging.getLogger(__name__)  # return autogluon root logger


# TODO: num_bag_sets -> ag_args

# Extra TODOs (Stretch): Can occur post v0.1
# TODO: make core_kwargs a kwargs argument to predictor.fit
# TODO: add aux_kwargs to predictor.fit
# TODO: add pip freeze + python version output after fit + log file, validate that same pip freeze on load as cached
# TODO: predictor.clone()
# TODO: Add logging comments that models are serialized on disk after fit
# TODO: consider adding kwarg option for data which has already been preprocessed by feature generator to skip feature generation.
# TODO: Resolve raw text feature usage in default feature generator

# Done for Tabular
# TODO: Remove all `time_limits` in project, replace with `time_limit`

class TabularPredictor:
    """
    AutoGluon TabularPredictor predicts values in a column of a tabular dataset (classification or regression).

    Parameters
    ----------
    label : str
        Name of the column that contains the target variable to predict.
    problem_type : str, default = None
        Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression', 'quantile').
        If `problem_type = None`, the prediction problem type is inferred based on the label-values in provided dataset.
    eval_metric : function or str, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression, and 'pinball_loss' for quantile.

        Otherwise, options for classification:
            ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro', 'precision_micro',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
        Options for regression:
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2']
        For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

        You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
        For detailed instructions on creating and using a custom metric, refer to https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-custom-metric.html
    path : str, default = None
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    sample_weight : str, default = None
        If specified, this column-name indicates which column of the data should be treated as sample weights. This column will NOT be considered as a predictive feature.
        Sample weights should be non-negative (and cannot be nan), with larger values indicating which rows are more important than others.
        If you want your usage of sample weights to match results obtained outside of this Predictor, then ensure sample weights for your training (or tuning) data sum to the number of rows in the training (or tuning) data.
        You may also specify two special strings: 'auto_weight' (automatically choose a weighting strategy based on the data) or 'balance_weight' (equally weight classes in classification, no effect in regression). If specifying your own sample_weight column, make sure its name does not match these special strings.
    weight_evaluation : bool, default = False
        Only considered when `sample_weight` column is not None. Determines whether sample weights should be taken into account when computing evaluation metrics on validation/test data.
        If True, then weighted metrics will be reported based on the sample weights provided in the specified `sample_weight` (in which case `sample_weight` column must also be present in test data).
        In this case, the 'best' model used by default for prediction will also be decided based on a weighted version of evaluation metric.
        Note: we do not recommend specifying `weight_evaluation` when `sample_weight` is 'auto_weight' or 'balance_weight', instead specify appropriate `eval_metric`.
    groups : str, default = None
        [Experimental] If specified, AutoGluon will use the column named the value of groups in `train_data` during `.fit` as the data splitting indices for the purposes of bagging.
        This column will not be used as a feature during model training.
        This parameter is ignored if bagging is not enabled. To instead specify a custom validation set with bagging disabled, specify `tuning_data` in `.fit`.
        The data will be split via `sklearn.model_selection.LeaveOneGroupOut`.
        Use this option to control the exact split indices AutoGluon uses.
        It is not recommended to use this option unless it is required for very specific situations.
        Bugs may arise from edge cases if the provided groups are not valid to properly train models, such as if not all classes are present during training in multiclass classification. It is up to the user to sanitize their groups.

        As an example, if you want your data folds to preserve adjacent rows in the table without shuffling, then for 3 fold bagging with 6 rows of data, the groups column values should be [0, 0, 1, 1, 2, 2].
    **kwargs :
        learner_type : AbstractLearner, default = DefaultLearner
            A class which inherits from `AbstractLearner`. This dictates the inner logic of predictor.
            If you don't know what this is, keep it as the default.
        learner_kwargs : dict, default = None
            Kwargs to send to the learner. Options include:

            positive_class : str or int, default = None
                Used to determine the positive class in binary classification.
                This is used for certain metrics such as 'f1' which produce different scores depending on which class is considered the positive class.
                If not set, will be inferred as the second element of the existing unique classes after sorting them.
                    If classes are [0, 1], then 1 will be selected as the positive class.
                    If classes are ['def', 'abc'], then 'def' will be selected as the positive class.
                    If classes are [True, False], then True will be selected as the positive class.
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
                If you don't know what this is, keep it as the default.

    Attributes
    ----------
    path : str
        Path to directory where all models used by this Predictor are stored.
    problem_type : str
        What type of prediction problem this Predictor has been trained for.
    eval_metric : function or str
        What metric is used to evaluate predictive performance.
    label : str
        Name of table column that contains data from the variable to predict (often referred to as: labels, response variable, target variable, dependent variable, Y, etc).
    feature_metadata : :class:`autogluon.common.features.feature_metadata.FeatureMetadata`
        Inferred data type of each predictive variable after preprocessing transformation (i.e. column of training data table used to predict `label`).
        Contains both raw dtype and special dtype information. Each feature has exactly 1 raw dtype (such as 'int', 'float', 'category') and zero to many special dtypes (such as 'datetime_as_int', 'text', 'text_ngram').
        Special dtypes are AutoGluon specific feature types that are used to identify features with meaning beyond what the raw dtype can convey.
            `feature_metadata.type_map_raw`: Dictionary of feature name -> raw dtype mappings.
            `feature_metadata.type_group_map_special`: Dictionary of lists of special feature names, grouped by special feature dtype.
    positive_class : str or int
        Returns the positive class name in binary classification. Useful for computing metrics such as F1 which require a positive and negative class.
        In binary classification, :meth:`TabularPredictor.predict_proba` returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.
    class_labels : list
        For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output.
        For binary problems, this list contains the class labels in sorted order of `predict_proba(as_multiclass=True)` output.
            `class_labels[0]` corresponds to internal label = 0 (negative class), `class_labels[1]` corresponds to internal label = 1 (positive class).
            This is relevant for certain metrics such as F1 where True and False labels impact the metric score differently.
        For other problem types, will equal None.
        For example if `pred = predict_proba(x, as_multiclass=True)`, then ith index of `pred` provides predicted probability that `x` belongs to class given by `class_labels[i]`.
    class_labels_internal : list
        For multiclass problems, this list contains the internal class labels in sorted order of internal `predict_proba()` output.
        For binary problems, this list contains the internal class labels in sorted order of internal `predict_proba(as_multiclass=True)` output.
            The value will always be `class_labels_internal=[0, 1]` for binary problems, with 0 as the negative class, and 1 as the positive class.
        For other problem types, will equal None.
    class_labels_internal_map : dict
        For binary and multiclass classification problems, this dictionary contains the mapping of the original labels to the internal labels.
        For example, in binary classification, label values of 'True' and 'False' will be mapped to the internal representation `1` and `0`.
            Therefore, class_labels_internal_map would equal {'True': 1, 'False': 0}
        For other problem types, will equal None.
        For multiclass, it is possible for not all of the label values to have a mapping.
            This indicates that the internal models will never predict those missing labels, and training rows associated with the missing labels were dropped.
    """

    Dataset = TabularDataset
    predictor_file_name = 'predictor.pkl'
    _predictor_version_file_name = '__version__'

    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=2,
            sample_weight=None,
            weight_evaluation=False,
            groups=None,
            **kwargs
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity)
        if sample_weight == AUTO_WEIGHT:  # TODO: update auto_weight strategy and make it the default
            sample_weight = None
            logger.log(15, f"{AUTO_WEIGHT} currently does not use any sample weights.")
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation  # TODO: sample_weight and weight_evaluation can both be properties that link to self._learner.sample_weight, self._learner.weight_evaluation
        if self.sample_weight in [AUTO_WEIGHT, BALANCE_WEIGHT] and self.weight_evaluation:
            logger.warning(
                f"We do not recommend specifying weight_evaluation when sample_weight='{self.sample_weight}', instead specify appropriate eval_metric.")
        self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path)

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())
        quantile_levels = kwargs.get('quantile_levels', None)

        self._learner: AbstractTabularLearner = learner_type(path_context=path, label=label, feature_generator=None,
                                                             eval_metric=eval_metric, problem_type=problem_type,
                                                             quantile_levels=quantile_levels,
                                                             sample_weight=self.sample_weight,
                                                             weight_evaluation=self.weight_evaluation, groups=groups,
                                                             **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    @property
    def class_labels(self):
        return self._learner.class_labels

    @property
    def class_labels_internal(self):
        return self._learner.label_cleaner.ordered_class_labels_transformed

    @property
    def class_labels_internal_map(self):
        return self._learner.label_cleaner.inv_map

    @property
    def quantile_levels(self):
        return self._learner.quantile_levels

    @property
    def eval_metric(self):
        return self._learner.eval_metric

    @property
    def problem_type(self):
        return self._learner.problem_type

    def features(self, feature_stage: str = 'original'):
        """
        Returns a list of feature names dependent on the value of feature_stage.

        Parameters
        ----------
        feature_stage : str, default = 'original'
            If 'original', returns the list of features specified in the original training data. This feature set is required in input data when making predictions.
            If 'transformed', returns the list of features after pre-processing by the feature generator.

        Returns
        -------
        Returns a list of feature names
        """
        if feature_stage == 'original':
            return self.feature_metadata_in.get_features()
        elif feature_stage == 'transformed':
            return self.feature_metadata.get_features()
        else:
            raise ValueError(f"Unknown feature_stage: '{feature_stage}'. Must be one of {['original', 'transformed']}")

    @property
    def feature_metadata(self):
        return self._trainer.feature_metadata

    @property
    def feature_metadata_in(self):
        return self._learner.feature_generator.feature_metadata_in

    @property
    def label(self):
        return self._learner.label

    @property
    def path(self):
        return self._learner.path

    @apply_presets(tabular_presets_dict, tabular_presets_alias)
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            feature_metadata='infer',
            infer_limit=None,
            infer_limit_batch_size=None,
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
        presets : list or str or dict, default = ['medium_quality']
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'`.
            To get good quality with minimal disk usage, set `presets=['good_quality', 'optimize_for_deployment']`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/tabular/configs/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.

            Available Presets: ['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'interpretable', 'ignore_text']

            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            In-depth Preset Info:
                best_quality={'auto_stack': True}
                    Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
                    Recommended for applications that benefit from the best possible model accuracy.

                high_quality={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, '_save_bag_folds': False}
                    High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
                    Recommended for applications that require reasonable inference speed and/or model size.

                good_quality={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, '_save_bag_folds': False, 'hyperparameters': 'light'}
                    Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality`.
                    Recommended for applications that require fast inference speed.

                medium_quality={'auto_stack': False}
                    Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality`.
                    This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality` results in significantly better predictive accuracy and faster inference time.

                optimize_for_deployment={'keep_only_best': True, 'save_space': True}
                    Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
                    Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
                    This will disable numerous advanced functionality, but has no impact on inference.
                    This will make certain functionality less informative, such as `predictor.leaderboard()` and `predictor.fit_summary()`.
                        Because unused models will be deleted under this preset, methods like `predictor.leaderboard()` and `predictor.fit_summary()` will no longer show the full set of models that were trained during `fit()`.
                    Recommended for applications where the inner details of AutoGluon's training is not important and there is no intention of manually choosing between the final models.
                    This preset pairs well with the other presets such as `good_quality` to make a very compact final model.
                    Identical to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` and `predictor.save_space()` directly after `fit()`.

                interpretable={'auto_stack': False, 'hyperparameters': 'interpretable'}
                    Fits only interpretable rule-based models from the imodels package.
                    Trades off predictive accuracy for conciseness.

                ignore_text={'_feature_generator_kwargs': {'enable_text_ngram_features': False, 'enable_text_special_features': False, 'enable_raw_text_features': False}}
                    Disables automated feature generation when text features are detected.
                    This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.
                    Ignored if `feature_generator` was also specified.

        hyperparameters : str or dict, default = 'default'
            Determines the hyperparameters used by the models.
            If `str` is passed, will use a preset hyperparameter configuration.
                Valid `str` options: ['default', 'light', 'very_light', 'toy', 'multimodal']
                    'default': Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
                    'light': Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
                    'very_light': Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
                    'toy': Results in extremely small models. Only use this when prototyping, as the model quality will be severely reduced.
                    'multimodal': [EXPERIMENTAL] Trains a multimodal transformer model alongside tabular models. Requires that some text columns appear in the data, a GPU, and CUDA-enabled MXNet.
                        When combined with 'best_quality' `presets` option, this can achieve extremely strong results in multimodal data tables that contain columns with text in addition to numeric/categorical columns.
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
                    'NN_MXNET' (neural network implemented in MXNet)
                    'NN_TORCH' (neural network implemented in Pytorch)
                    'FASTAI' (neural network with FastAI backend)
                Experimental model options include:
                    'FASTTEXT' (FastText)
                    'AG_TEXT_NN' (Multimodal Text+Tabular model, GPU is required)
                    'TRANSF' (Tabular Transformer, GPU is recommended)
                If a certain key is missing from hyperparameters, then `fit()` will not train any models of that type. Omitting a model key from hyperparameters is equivalent to including this model key in `excluded_model_types`.
                For example, set `hyperparameters = { 'NN_TORCH':{...} }` if say you only want to train (PyTorch) neural networks and no other types of models.
            Values = dict of hyperparameter settings for each model type, or list of dicts.
                Each hyperparameter can either be a single fixed value or a search space containing many possible values.
                Unspecified hyperparameters will be set to default values (or default search spaces if `hyperparameter_tune = True`).
                Caution: Any provided search spaces will error if `hyperparameter_tune = False`.
                To train multiple models of a given type, set the value to a list of hyperparameter dictionaries.
                    For example, `hyperparameters = {'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}]}` will result in 2 random forest models being trained with separate hyperparameters.
                Some model types have preset hyperparameter configs keyed under strings as shorthand for a complex model hyperparameter configuration known to work well:
                    'GBM': ['GBMLarge']
            Advanced functionality: Bring your own model / Custom model support
                AutoGluon fully supports custom models. For a detailed tutorial on creating and using custom models with AutoGluon, refer to https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-custom-model.html
            Advanced functionality: Custom stack levels
                By default, AutoGluon re-uses the same models and model hyperparameters at each level during stack ensembling.
                To customize this behaviour, create a hyperparameters dictionary separately for each stack level, and then add them as values to a new dictionary, with keys equal to the stack level.
                    Example: `hyperparameters = {1: {'RF': rf_params1}, 2: {'CAT': [cat_params1, cat_params2], 'NN_TORCH': {}}}`
                    This will result in a stack ensemble that has one custom random forest in level 1 followed by two CatBoost models with custom hyperparameters and a default neural network in level 2, for a total of 4 models.
                If a level is not specified in `hyperparameters`, it will default to using the highest specified level to train models. This can also be explicitly controlled by adding a 'default' key.

            Default:
                hyperparameters = {
                    'NN_TORCH': {},
                    'GBM': [
                        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                        {},
                        'GBMLarge',
                    ],
                    'CAT': {},
                    'XGB': {},
                    'FASTAI': {},
                    'RF': [
                        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'XT': [
                        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                        {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
                    ],
                    'KNN': [
                        {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
                        {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
                    ],
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
                    ag_args: Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, the problem types it is valid for, and the type of HPO it utilizes.
                        Valid keys:
                            name: (str) The name of the model. This overrides AutoGluon's naming logic and all other name arguments if present.
                            name_main: (str) The main name of the model. Example: 'RandomForest'.
                            name_prefix: (str) Add a custom prefix to the model name. Unused by default.
                            name_suffix: (str) Add a custom suffix to the model name. Unused by default.
                            priority: (int) Determines the order in which the model is trained. Larger values result in the model being trained earlier. Default values range from 100 (KNN) to 0 (custom), dictated by model type. If you want this model to be trained first, set priority = 999.
                            problem_types: (list) List of valid problem types for the model. `problem_types=['binary']` will result in the model only being trained if `problem_type` is 'binary'.
                            disable_in_hpo: (bool) If True, the model will only be trained if `hyperparameter_tune_kwargs=None`.
                            valid_stacker: (bool) If False, the model will not be trained as a level 2 or higher stacker model.
                            valid_base: (bool) If False, the model will not be trained as a level 1 (base) model.
                            hyperparameter_tune_kwargs: (dict) Refer to :meth:`TabularPredictor.fit` hyperparameter_tune_kwargs argument. If specified here, will override global HPO settings for this model.
                        Reference the default hyperparameters for example usage of these options.
                    ag_args_fit: Dictionary of model fit customization options related to how and with what constraints the model is trained. These parameters affect stacker fold models, but not stacker models themselves.
                        Clarification: `time_limit` is the internal time in seconds given to a particular model to train, which is dictated in part by the `time_limit` argument given during `predictor.fit()` but is not the same.
                        Valid keys:
                            stopping_metric: (str or :class:`autogluon.core.metrics.Scorer`, default=None) The metric to use for early stopping of the model. If None, model will decide.
                            max_memory_usage_ratio: (float, default=1.0) The ratio of memory usage relative to the default to allow before early stopping or killing the model. Values greater than 1.0 will be increasingly prone to out-of-memory errors.
                            max_time_limit_ratio: (float, default=1.0) The ratio of the provided time_limit to use during model `fit()`. If `time_limit=10` and `max_time_limit_ratio=0.3`, time_limit would be changed to 3. Does not alter max_time_limit or min_time_limit values.
                            max_time_limit: (float, default=None) Maximum amount of time to allow this model to train for (in sec). If the provided time_limit is greater than this value, it will be replaced by max_time_limit.
                            min_time_limit: (float, default=0) Allow this model to train for at least this long (in sec), regardless of the time limit it would otherwise be granted.
                                If `min_time_limit >= max_time_limit`, time_limit will be set to min_time_limit.
                                If `min_time_limit=None`, time_limit will be set to None and the model will have no training time restriction.
                            num_cpus : (int or str, default='auto')
                                How many CPUs to use during model fit.
                                If 'auto', model will decide.
                            num_gpus : (int or str, default='auto')
                                How many GPUs to use during model fit.
                                If 'auto', model will decide. Some models can use GPUs but don't by default due to differences in model quality.
                                Set to 0 to disable usage of GPUs.
                    ag_args_ensemble: Dictionary of hyperparameters shared by all models that control how they are ensembled, if bag mode is enabled.
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
                            fold_fitting_strategy: (AbstractFoldFittingStrategy default=auto) Whether to fit folds in parallel or in sequential order.
                                If parallel_local, folds will be trained in parallel with evenly distributed computing resources. This could bring 2-4x speedup compared to SequentialLocalFoldFittingStrategy, but could consume much more memory.
                                If sequential_local, folds will be trained in sequential.
                                If auto, strategy will be determined by OS and whether ray is installed or not. MacOS support for parallel_local is unstable, and may crash if enabled.
                            num_folds_parallel: (int or str, default='auto') Number of folds to be trained in parallel if using ParallelLocalFoldFittingStrategy. Consider lowering this value if you encounter either out of memory issue or CUDA out of memory issue(when trained on gpu).
                                if 'auto', will try to train all folds in parallel.

        feature_metadata : :class:`autogluon.tabular.FeatureMetadata` or str, default = 'infer'
            The feature metadata used in various inner logic in feature preprocessing.
            If 'infer', will automatically construct a FeatureMetadata object based on the properties of `train_data`.
            In this case, `train_data` is input into :meth:`autogluon.tabular.FeatureMetadata.from_df` to infer `feature_metadata`.
            If 'infer' incorrectly assumes the dtypes of features, consider explicitly specifying `feature_metadata`.
        infer_limit : float, default = None
            The inference time limit in seconds per row to adhere to during fit.
            If infer_limit=0.05 and infer_limit_batch_size=1000, AutoGluon will avoid training models that take longer than 50 ms/row to predict when given a batch of 1000 rows to predict (must predict 1000 rows in no more than 50 seconds).
            If bagging is enabled, the inference time limit will be respected based on estimated inference speed of `_FULL` models after refit_full is called, NOT on the inference speed of the bagged ensembles.
            The inference times calculated for models are assuming `predictor.persist_models('all')` is called after fit.
            If None, no limit is enforced.
            If it is impossible to satisfy the constraint, an exception will be raised.
        infer_limit_batch_size : int, default = None
            The batch size to use when predicting in bulk to estimate per-row inference time.
            Must be an integer greater than 0.
            If None and `infer_limit` is specified, will default to 10000.
            It is recommended to set to 10000 unless you must satisfy an online-inference scenario.
            Small values, especially `infer_limit_batch_size=1`, will result in much larger per-row inference times and should be avoided if possible.
            Refer to `infer_limit` for more details on how this is used.
            If specified when `infer_limit=None`, the inference time will be logged during training but will not be limited.
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
                Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bag_folds != 0` unless `use_bag_holdout == True`).
                Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
                Default value is doubled if `hyperparameter_tune_kwargs` is set, up to a maximum of 0.2.
                Disabled if `num_bag_folds >= 2` unless `use_bag_holdout == True`.
            use_bag_holdout : bool, default = False
                If True, a `holdout_frac` portion of the data is held-out from model bagging.
                This held-out data is only used to score models and determine weighted ensemble weights.
                Enable this if there is a large gap between score_val and score_test in stack models.
                Note: If `tuning_data` was specified, `tuning_data` is used as the holdout data.
                Disabled if not bagging.
            hyperparameter_tune_kwargs : str or dict, default = None
                Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
                If None, then hyperparameter tuning will not be performed.
                Valid preset values:
                    'auto': Uses the 'random' preset.
                    'random': Performs HPO via random search using local scheduler.
                The 'searcher' key is required when providing a dict.
            feature_prune_kwargs: dict, default = None
                Performs layer-wise feature pruning via recursive feature elimination with permutation feature importance.
                This fits all models in a stack layer once, discovers a pruned set of features, fits all models in the stack layer
                again with the pruned set of features, and updates input feature lists for models whose validation score improved.
                If None, do not perform feature pruning. If empty dictionary, perform feature pruning with default configurations.
                For valid dictionary keys, refer to :class:`autogluon.core.utils.feature_selection.FeatureSelector` and
                `autogluon.core.trainer.abstract_trainer.AbstractTrainer._proxy_model_feature_prune` documentation.
                To force all models to work with the pruned set of features, set force_prune=True in the dictionary.
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
            feature_generator : :class:`autogluon.features.generators.AbstractFeatureGenerator`, default = :class:`autogluon.features.generators.AutoMLPipelineFeatureGenerator`
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
            calibrate: bool or str, default = 'auto'
                Note: It is recommended to use ['auto', False] as the values and avoid True.
                If 'auto' will automatically set to True if the problem_type and eval_metric are suitable for calibration.
                If True and the problem_type is classification, temperature scaling will be used to calibrate the Predictor's estimated class probabilities
                (which may improve metrics like log_loss) and will train a scalar parameter on the validation set.
                If True and the problem_type is quantile regression, conformalization will be used to calibrate the Predictor's estimated quantiles
                (which may improve the prediction interval coverage, and bagging could futher improve it) and will compute a set of scalar parameters on the validation set.

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
            raise AssertionError(
                'Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new `Predictor`.')
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity)

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
        feature_generator = kwargs['feature_generator']
        unlabeled_data = kwargs['unlabeled_data']
        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']
        use_bag_holdout = kwargs['use_bag_holdout']

        if ag_args is None:
            ag_args = {}
        ag_args = self._set_hyperparameter_tune_kwargs_in_ag_args(kwargs['hyperparameter_tune_kwargs'], ag_args,
                                                                  time_limit=time_limit)

        feature_generator_init_kwargs = kwargs['_feature_generator_kwargs']
        if feature_generator_init_kwargs is None:
            feature_generator_init_kwargs = dict()

        train_data, tuning_data, unlabeled_data = self._validate_fit_data(train_data=train_data,
                                                                          tuning_data=tuning_data,
                                                                          unlabeled_data=unlabeled_data)
        infer_limit, infer_limit_batch_size = self._validate_infer_limit(infer_limit=infer_limit, infer_limit_batch_size=infer_limit_batch_size)

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        # TODO: Hyperparam could have non-serializble objects. Save as pkl and loaded on demand
        # in case the hyperprams are large in memory
        self.fit_hyperparameters_ = hyperparameters

        ###################################
        # FIXME: v0.1 This section is a hack
        if 'enable_raw_text_features' not in feature_generator_init_kwargs:
            if 'AG_TEXT_NN' in hyperparameters or 'VW' in hyperparameters:
                feature_generator_init_kwargs['enable_raw_text_features'] = True
            else:
                for key in hyperparameters:
                    if isinstance(key, int) or key == 'default':
                        if 'AG_TEXT_NN' in hyperparameters[key] or 'VW' in hyperparameters[key]:
                            feature_generator_init_kwargs['enable_raw_text_features'] = True
                            break
        ###################################

        if feature_metadata is not None and isinstance(feature_metadata, str) and feature_metadata == 'infer':
            feature_metadata = None
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata,
                                    init_kwargs=feature_generator_init_kwargs)

        num_bag_folds, num_bag_sets, num_stack_levels = self._sanitize_stack_args(
            num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets, num_stack_levels=num_stack_levels,
            time_limit=time_limit, auto_stack=auto_stack, num_train_rows=len(train_data),
        )

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(len(train_data),
                                                ag_args.get('hyperparameter_tune_kwargs', None) is not None)

        if kwargs['_save_bag_folds'] is not None:
            if use_bag_holdout and not kwargs['_save_bag_folds']:
                logger.log(30,
                           f'WARNING: Attempted to disable saving of bagged fold models when `use_bag_holdout=True`. Forcing `save_bag_folds=True` to avoid errors.')
            else:
                if ag_args_ensemble is None:
                    ag_args_ensemble = {}
                ag_args_ensemble['save_bag_folds'] = kwargs['_save_bag_folds']

        if time_limit is None:
            mb_mem_usage_train_data = get_approximate_df_mem_usage(train_data, sample_ratio=0.2).sum() / 1e6
            num_rows_train = len(train_data)
            if mb_mem_usage_train_data >= 50 or num_rows_train >= 100000:
                logger.log(20,
                           f'Warning: Training may take a very long time because `time_limit` was not specified and `train_data` is large ({num_rows_train} samples, {round(mb_mem_usage_train_data, 2)} MB).')
                logger.log(20,
                           f'\tConsider setting `time_limit` to ensure training finishes within an expected duration or experiment with a small portion of `train_data` to identify an ideal `presets` and `hyperparameters` configuration.')

        core_kwargs = {
            'ag_args': ag_args,
            'ag_args_ensemble': ag_args_ensemble,
            'ag_args_fit': ag_args_fit,
            'excluded_model_types': excluded_model_types,
            'feature_prune_kwargs': kwargs.get('feature_prune_kwargs', None)
        }
        self.save(silent=True)  # Save predictor to disk to enable prediction and training after interrupt
        self._learner.fit(X=train_data, X_val=tuning_data, X_unlabeled=unlabeled_data,
                          holdout_frac=holdout_frac, num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets,
                          num_stack_levels=num_stack_levels,
                          hyperparameters=hyperparameters, core_kwargs=core_kwargs,
                          time_limit=time_limit, infer_limit=infer_limit, infer_limit_batch_size=infer_limit_batch_size,
                          verbosity=verbosity, use_bag_holdout=use_bag_holdout)
        self._set_post_fit_vars()

        self._post_fit(
            keep_only_best=kwargs['keep_only_best'],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs['set_best_to_refit_full'],
            save_space=kwargs['save_space'],
            calibrate=kwargs['calibrate'],
            infer_limit=infer_limit,
        )
        self.save()
        return self

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False, save_space=False,
                  calibrate=False, infer_limit=None):
        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = 'best'
                else:
                    logger.warning(
                        f'refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted.')
                    refit_full = False
            else:
                refit_full = 'all'

        if refit_full is not False:
            if infer_limit is not None:
                infer_limit = infer_limit - self._learner.preprocess_1_time
            trainer_model_best = self._trainer.get_model_best(infer_limit=infer_limit)
            self.refit_full(model=refit_full, set_best_to_refit_full=False)
            if set_best_to_refit_full:
                if trainer_model_best in self._trainer.model_full_dict.keys():
                    self._trainer.model_best = self._trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    self._trainer.save()
                else:
                    logger.warning(
                        f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

        if calibrate == 'auto':
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION and self.eval_metric.needs_proba:
                calibrate = True
            elif self.problem_type == QUANTILE:
                calibrate = True
            else:
                calibrate = False

        if calibrate:
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                self._calibrate_model()
            elif self.problem_type == QUANTILE:
                self._calibrate_model()
            else:
                logger.log(30, 'WARNING: `calibrate=True` is only applicable to classification or quantile regression problems. Skipping calibration...')

        if keep_only_best:
            self.delete_models(models_to_keep='best', dry_run=False)

        if save_space:
            self.save_space()

    def _calibrate_model(self, model_name: str = None, lr: float = 0.01, max_iter: int = 1000, init_val: float = 1.0):
        """
        Applies temperature scaling to the AutoGluon model. Applies
        inverse softmax to predicted probs then trains temperature scalar
        on validation data to maximize negative log likelihood. Inversed
        softmaxes are divided by temperature scalar then softmaxed to return
        predicted probs.

        Parameters:
        -----------
        model_name: str: default=None
            model name to tune temperature scaling on. If set to None
            then will tune best model only. Best model chosen by validation score
        lr: float: default=0.01
            The learning rate for temperature scaling algorithm
        max_iter: int: default=1000
            Number of iterations optimizer should take for
            tuning temperature scaler
        init_val: float: default=1.0
            The initial value for temperature scalar term
        """
        # TODO: Note that temperature scaling is known to worsen calibration in the face of shifted test data.
        try:
            # FIXME: Avoid depending on torch for temp scaling
            try_import_torch
        except ImportError:
            logger.log(30, 'Warning: Torch is not installed, skipping calibration step...')
            return

        if model_name is None:
            model_name = self.get_model_best()

        model_full_dict = self._trainer.model_full_dict
        model_name_og = model_name
        for m, m_full in model_full_dict.items():
            if m_full == model_name:
                model_name_og = m
                break
        if self._trainer.bagged_mode:
            y_val_probs = self.get_oof_pred_proba(model_name_og, transformed=True, internal_oof=True).to_numpy()
            y_val = self._trainer.load_y().to_numpy()
        else:
            X_val = self._trainer.load_X_val()
            y_val_probs = self._trainer.predict_proba(X_val, model_name_og)
            y_val = self._trainer.load_y_val().to_numpy()

            if self.problem_type == BINARY:
                y_val_probs = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_val_probs)

        model = self._trainer.load_model(model_name=model_name)
        if self.problem_type == QUANTILE:
            logger.log(15, f'Conformity scores being computed to calibrate model: {model_name}')
            conformalize = compute_conformity_score(y_val_pred=y_val_probs, y_val=y_val,
                                                    quantile_levels=self.quantile_levels)
            model.conformalize = conformalize
            model.save()
        else:
            logger.log(15, f'Temperature scaling term being tuned for model: {model_name}')
            temp_scalar = tune_temperature_scaling(y_val_probs=y_val_probs, y_val=y_val,
                                                   init_val=init_val, max_iter=max_iter, lr=lr)
            if temp_scalar is None:
                logger.log(15, f'Warning: Infinity found during calibration, skipping calibration on {model.name}! '
                               f'This can occur when the model is absolutely certain of a validation prediction (1.0 pred_proba).')
            else:
                logger.log(15, f'Temperature term found is: {temp_scalar}')
                model.temperature_scalar = temp_scalar
                model.save()

    # TODO: Consider adding infer_limit to fit_extra
    def fit_extra(self, hyperparameters, time_limit=None, base_model_names=None, **kwargs):
        """
        Fits additional models after the original :meth:`TabularPredictor.fit` call.
        The original train_data and tuning_data will be used to train the models.

        Parameters
        ----------
        hyperparameters : str or dict
            Refer to argument documentation in :meth:`TabularPredictor.fit`.
            If `base_model_names` is specified and hyperparameters is using the level-based key notation,
            the key of the level which directly uses the base models should be 1. The level in the hyperparameters
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
            pseudo_data : pd.DataFrame, default = None
                Data that has been self labeled by Autogluon model and will be incorporated into training during 'fit_extra'
        """
        self._assert_is_fit('fit_extra')
        time_start = time.time()

        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_extra_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity)

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
        # save_bag_folds = kwargs['save_bag_folds']  # TODO: Enable

        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']
        pseudo_data = kwargs.get('pseudo_data', None)

        # TODO: Since data preprocessor is fitted on original train_data it cannot account for if
        # labeled pseudo data has new labels unseen in the original train. Probably need to refit
        # data preprocessor if this is the case.
        if pseudo_data is not None:
            if self.label not in pseudo_data.columns:
                raise ValueError('\'pseudo_data\' does not contain the labeled column.')

            if self.sample_weight is not None:
                raise ValueError('Applying \'sample_weight\' while calling \'fit_pseudolabel\' is not supported')

            X_pseudo = pseudo_data.drop(columns=[self.label])
            y_pseudo_og = pseudo_data[self.label]
            X_pseudo = self._learner.transform_features(X_pseudo)
            y_pseudo = self._learner.label_cleaner.transform(y_pseudo_og)

            if np.isnan(y_pseudo.unique()).any():
                raise Exception('NaN was found in the label column for pseudo labeled data.'
                                'Please ensure no NaN values in target column')
        else:
            X_pseudo = None
            y_pseudo = None

        if ag_args is None:
            ag_args = {}
        ag_args = self._set_hyperparameter_tune_kwargs_in_ag_args(kwargs['hyperparameter_tune_kwargs'], ag_args,
                                                                  time_limit=time_limit)

        fit_new_weighted_ensemble = False  # TODO: Add as option
        aux_kwargs = None  # TODO: Add as option

        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        if num_stack_levels is None:
            hyperparameter_keys = list(hyperparameters.keys())
            highest_level = 1
            for key in hyperparameter_keys:
                if isinstance(key, int):
                    highest_level = max(key, highest_level)
            num_stack_levels = highest_level

        # TODO: make core_kwargs a kwargs argument to predictor.fit, add aux_kwargs to predictor.fit
        core_kwargs = {'ag_args': ag_args, 'ag_args_ensemble': ag_args_ensemble, 'ag_args_fit': ag_args_fit,
                       'excluded_model_types': excluded_model_types}

        if X_pseudo is not None and y_pseudo is not None:
            core_kwargs['X_pseudo'] = X_pseudo
            core_kwargs['y_pseudo'] = y_pseudo

        # TODO: Add special error message if called and training/val data was not cached.
        X, y, X_val, y_val = self._trainer.load_data()

        if y_pseudo is not None and self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            y_og = self._learner.label_cleaner.inverse_transform(y)
            y_og_classes = y_og.unique()
            y_pseudo_classes = y_pseudo_og.unique()
            matching_classes = np.in1d(y_pseudo_classes, y_og_classes)

            if not matching_classes.all():
                raise Exception(f'Pseudo training data contains classes not in original train data: {y_pseudo_classes[~matching_classes]}')

        name_suffix = kwargs.get('name_suffix', '')

        fit_models = self._trainer.train_multi_levels(
            X=X, y=y, hyperparameters=hyperparameters, X_val=X_val, y_val=y_val,
            base_model_names=base_model_names, time_limit=time_limit, relative_stack=True, level_end=num_stack_levels,
            core_kwargs=core_kwargs, aux_kwargs=aux_kwargs, name_suffix=name_suffix
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
            calibrate=kwargs['calibrate']
        )
        self.save()
        return self

    def _get_all_fit_extra_args(self):
        ret = list(self._fit_extra_kwargs_dict().keys()) + list(inspect.signature(self.fit_extra).parameters.keys())
        ret.remove('kwargs')

        return ret

    def _fit_weighted_ensemble_pseudo(self):
        """
        Fits weighted ensemble on top models trained with pseudo labeling, then if new
        weighted ensemble model is best model then sets `model_best` in trainer to
        weighted ensemble model.
        """
        logger.log(15, 'Fitting weighted ensemble using top models')
        weighted_ensemble_model_name = self.fit_weighted_ensemble()[0]

        # TODO: This is a hack! self.predict_prob does not update to use weighted ensemble
        # if it's the best model.
        # TODO: There should also be PL added to weighted ensemble model name to notify
        # users it is a model trained with PL models if they are indeed ensembled
        model_best_name = self._trainer.leaderboard().iloc[0]['model']
        if model_best_name == weighted_ensemble_model_name:
            self._trainer.model_best = model_best_name
            self._trainer.save()
            logger.log(15, 'Weighted ensemble was the best model for current iteration of pseudo labeling')
        else:
            logger.log(15, 'Weighted ensemble was not the best model for current iteration of pseudo labeling')

    def _run_pseudolabeling(self, unlabeled_data: pd.DataFrame, max_iter: int,
                            return_pred_prob: bool = False, use_ensemble: bool = False,
                            fit_ensemble: bool = False, fit_ensemble_every_iter: bool = False,
                            **kwargs):
        """
        Runs pseudolabeling algorithm using the same hyperparameters and model and fit settings
        used in original model unless specified by the user. This is an internal function that iteratively
        self labels unlabeled test data then incorporates all self labeled data above a threshold into training.
        Will keep incorporating self labeled data into training until validation score does not improve

        Parameters:
        -----------
        unlabeled_data: Extra unlabeled data (could be the test data) to assign pseudolabels to
            and incorporate as extra training data.
        max_iter: int, default = 5
            Maximum allowed number of iterations, where in each iteration, the data are pseudolabeled
            by the current predictor and the predictor is refit including the pseudolabled data in its training set.
        return_pred_proba: bool, default = False
            Transductive learning setting, will return predictive probabiliteis of unlabeled_data
        use_ensemble: bool, default = False
            If True will use ensemble pseudo labeling algorithm if False will use best model
            pseudo labeling method
        fit_ensemble: bool, default = False
            If True will fit weighted ensemble on final best models. Fitting weighted ensemble will be done after fitting
            of models is completed unless otherwise specified. If False will not fit weighted ensemble on final best
            models.
        fit_ensemble_every_iter: bool, default = False
            If True will fit weighted ensemble model using combination of best models
            for every iteration of pseudo label algorithm. If False and fit_ensemble
            is True, will just do it at the very end of training pseudo labeled models.

        Returns:
        --------
        self: TabularPredictor
        """
        previous_score = self.info()['best_model_score_val']
        y_pseudo_og = pd.Series()
        if return_pred_prob:
            if self.problem_type is REGRESSION:
                y_pred_proba_og = pd.Series()
            else:
                y_pred_proba_og = pd.DataFrame()
        X_test = unlabeled_data.copy()

        for i in range(max_iter):
            if len(X_test) == 0:
                logger.log(20, f'No more unlabeled data to pseudolabel. Done with pseudolabeling...')
                break

            iter_print = str(i + 1)
            logger.log(20, f'Beginning iteration {iter_print} of pseudolabeling out of max: {max_iter}')

            if use_ensemble:
                if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                    test_pseudo_idxes_true, y_pred_proba, y_pred = filter_ensemble_pseudo(predictor=self,
                                                                                          unlabeled_data=X_test)
                else:
                    test_pseudo_idxes_true, y_pred = filter_ensemble_pseudo(predictor=self, unlabeled_data=X_test)
                    y_pred_proba = y_pred.copy()
            else:
                y_pred_proba = self.predict_proba(data=X_test, as_multiclass=True)
                y_pred = get_pred_from_proba_df(y_pred_proba, problem_type=self.problem_type)
                test_pseudo_idxes_true = filter_pseudo(y_pred_proba_og=y_pred_proba, problem_type=self.problem_type)

            if return_pred_prob:
                if i == 0:
                    y_pred_proba_og = y_pred_proba
                else:
                    y_pred_proba_og.loc[test_pseudo_idxes_true.index] = y_pred_proba.loc[test_pseudo_idxes_true.index]

            if len(test_pseudo_idxes_true) < 1:
                logger.log(20,
                           f'Could not confidently assign pseudolabels for any of the provided rows in iteration: {iter_print}. Done with pseudolabeling...')
                break
            else:
                logger.log(20,
                           f'Pseudolabeling algorithm confidently assigned pseudolabels to: {len(test_pseudo_idxes_true)} rows of data'
                           f'on iteration: {iter_print}. Adding to train data')

            test_pseudo_idxes = pd.Series(data=False, index=y_pred_proba.index)
            test_pseudo_idxes[test_pseudo_idxes_true.index] = True

            y_pseudo_og = y_pseudo_og.append(y_pred.loc[test_pseudo_idxes_true.index], verify_integrity=True)

            pseudo_data = unlabeled_data.loc[y_pseudo_og.index]
            pseudo_data[self.label] = y_pseudo_og
            self.fit_extra(pseudo_data=pseudo_data, name_suffix=PSEUDO_MODEL_SUFFIX.format(iter=(i + 1)),
                           **kwargs)

            if fit_ensemble and fit_ensemble_every_iter:
                self._fit_weighted_ensemble_pseudo()

            current_score = self.info()['best_model_score_val']
            logger.log(20,
                       f'Pseudolabeling algorithm changed validation score from: {previous_score}, to: {current_score}'
                       f' using evaluation metric: {self.eval_metric.name}')

            if previous_score >= current_score:
                break
            else:
                # Cut down X_test to not include pseudo labeled data
                X_test = X_test.loc[test_pseudo_idxes[~test_pseudo_idxes].index]
                previous_score = current_score

        if fit_ensemble and not fit_ensemble_every_iter:
            self._fit_weighted_ensemble_pseudo()
            y_pred_proba_og = self.predict_proba(unlabeled_data)

        if return_pred_prob:
            return self, y_pred_proba_og
        else:
            return self

    def fit_pseudolabel(self, pseudo_data: pd.DataFrame, max_iter: int = 5, return_pred_prob: bool = False,
                        use_ensemble: bool = False, fit_ensemble: bool = False, fit_ensemble_every_iter: bool = False,
                        **kwargs):
        """
        If 'pseudo_data' is labeled then incorporates all test_data into train_data for
        newly fit models. If 'pseudo_data' is unlabeled then 'fit_pseudolabel' will self label the
        data and will augment the original training data by adding all the self labeled
        data that meets a criteria (For example all rows with predictive prob above 95%). If
        predictor is fit then will call fit_extra with added training data, if predictor
        is not fit then will fit model on train_data then run.

        Parameters
        ----------
        pseudo_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            Extra data to incorporate into training. Pre-labeled test data allowed. If no labels
            then pseudolabeling algorithm will predict and filter out which rows to incorporate into
            training
        max_iter: int, default = 5
            Maximum iterations of pseudolabeling allowed
        return_pred_prob: bool, default = False
            Returns held-out predictive probabilities from pseudo-labeling. If test_data is labeled then
            returns model's predictive probabilities.
        use_ensemble: bool, default = False
            If True will use ensemble pseudo labeling algorithm. If False will just use best model
            for pseudo labeling algorithm.
        fit_ensemble: bool, default = False
            If True with fit weighted ensemble model using combination of best models.
            Fitting weighted ensemble will be done after fitting has
            being completed unless otherwise specified. If False will not fit weighted ensemble
            over models trained with pseudo labeling and models trained without it.
        fit_ensemble_every_iter: bool, default = False
            If True fits weighted ensemble model for every iteration of pseudo labeling algorithm. If False
            and fit_ensemble is True will fit after all pseudo labeling training is done.
        kwargs: dict
            If predictor is not already fit, then kwargs are for the functions 'fit' and 'fit_extra':
            Refer to parameters documentation in :meth:`TabularPredictor.fit`.
            Refer to parameters documentation in :meth:`TabularPredictor.fit_extra`.
            If predictor is fit kwargs are for 'fit_extra':
            Refer to parameters documentation in :meth:`TabularPredictor.fit_extra`.

        Returns
        -------
        self : TabularPredictor
            Returns self, which is a Python class of TabularPredictor
        """
        if len(pseudo_data) < 1:
            raise Exception('No pseudo data given')

        self._validate_unique_indices(pseudo_data, 'pseudo_data')

        if not self._learner.is_fit:
            if 'train_data' not in kwargs.keys():
                Exception('Autogluon is required to be fit or given \'train_data\' in order to run \'fit_pseudolabel\'.'
                          ' Autogluon is not fit and \'train_data\' was not given')

            logger.log(20,
                       f'Predictor not fit prior to pseudolabeling. Fitting now...')
            self.fit(**kwargs)

        if self.problem_type is MULTICLASS and self.eval_metric.name != 'accuracy':
            logger.warning('AutoGluon has detected the problem type as \'multiclass\' and '
                           f'eval_metric is {self.eval_metric.name}, we recommend using'
                           f'fit_pseudolabeling when eval metric is \'accuracy\'')

        is_labeled = self.label in pseudo_data.columns

        hyperparameters = kwargs.get('hyperparameters', None)
        if hyperparameters is None:
            if self._learner.is_fit:
                hyperparameters = self.fit_hyperparameters_
        elif isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        kwargs['hyperparameters'] = hyperparameters
        fit_extra_args = self._get_all_fit_extra_args()
        fit_extra_kwargs = {key: value for key, value in kwargs.items() if key in fit_extra_args}
        if is_labeled:
            logger.log(20, "Fitting predictor using the provided pseudolabeled examples as extra training data...")
            self.fit_extra(pseudo_data=pseudo_data, name_suffix=PSEUDO_MODEL_SUFFIX.format(iter='')[:-1],
                           **fit_extra_kwargs)

            if fit_ensemble:
                logger.log(15, 'Fitting weighted ensemble model using best models')
                self.fit_weighted_ensemble()

            if return_pred_prob:
                y_pred_proba = self.predict_proba(pseudo_data)
                return self, y_pred_proba
            else:
                return self
        else:
            logger.log(20, 'Given test_data for pseudo labeling did not contain labels. '
                           'AutoGluon will assign pseudo labels to data and use it for extra training data...')
            return self._run_pseudolabeling(unlabeled_data=pseudo_data, max_iter=max_iter,
                                            return_pred_prob=return_pred_prob, use_ensemble=use_ensemble,
                                            fit_ensemble=fit_ensemble, fit_ensemble_every_iter=fit_ensemble_every_iter,
                                            **fit_extra_kwargs)

    def predict(self, data, model=None, as_pandas=True, transform_features=True):
        """
        Use trained models to produce predictions of `label` column values for new data.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training Dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        as_pandas : bool, default = True
            Whether to return the output as a :class:`pd.Series` (True) or :class:`np.ndarray` (False).
        transform_features : bool, default = True
            If True, preprocesses data before predicting with models.
            If False, skips global feature preprocessing.
                This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset. Either :class:`np.ndarray` or :class:`pd.Series` depending on `as_pandas` argument.
        """
        self._assert_is_fit('predict')
        data = self.__get_dataset(data)
        return self._learner.predict(X=data, model=model, as_pandas=as_pandas, transform_features=transform_features)

    def predict_proba(self, data, model=None, as_pandas=True, as_multiclass=True, transform_features=True):
        """
        Use trained models to produce predicted class probabilities rather than class-labels (if task is classification).
        If `predictor.problem_type` is regression, this functions identically to `predict`, returning the same output.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or numpy array (False).
            Pandas object is a DataFrame if this is a multiclass problem or `as_multiclass=True`, otherwise it is a Series.
            If the output is a DataFrame, the column order will be equivalent to `predictor.class_labels`.
        as_multiclass : bool, default = True
            Whether to return binary classification probabilities as if they were for multiclass classification.
                Output will contain two columns, and if `as_pandas=True`, the column names will correspond to the binary class labels.
                The columns will be the same order as `predictor.class_labels`.
            If False, output will contain only 1 column for the positive class (get positive_class name via `predictor.positive_class`).
            Only impacts output for binary classification problems.
        transform_features : bool, default = True
            If True, preprocesses data before predicting with models.
            If False, skips global feature preprocessing.
                This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        May be a :class:`np.ndarray` or :class:`pd.DataFrame` / :class:`pd.Series` depending on `as_pandas` and `as_multiclass` arguments and the type of prediction problem.
        For binary classification problems, the output contains for each datapoint the predicted probabilities of the negative and positive classes, unless you specify `as_multiclass=False`.
        """
        self._assert_is_fit('predict_proba')
        data = self.__get_dataset(data)
        return self._learner.predict_proba(X=data, model=model, as_pandas=as_pandas, as_multiclass=as_multiclass, transform_features=transform_features)

    def evaluate(self, data, model=None, silent=False, auxiliary_metrics=True, detailed_report=False) -> dict:
        """
        Report the predictive performance evaluated over a given dataset.
        This is basically a shortcut for: `pred_proba = predict_proba(data); evaluate_predictions(data[label], pred_proba)`.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            This dataset must also contain the `label` with the same column-name as previously specified.
            If str is passed, `data` will be loaded using the str value as the file path.
            If `self.sample_weight` is set and `self.weight_evaluation==True`, then a column with the sample weight name is checked and used for weighted metric evaluation if it exists.
        model : str (optional)
            The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
        silent : bool, default = False
            If False, performance results are printed.
        auxiliary_metrics: bool, default = True
            Should we compute other (`problem_type` specific) metrics in addition to the default metric?
        detailed_report : bool, default = False
            Should we computed more detailed versions of the `auxiliary_metrics`? (requires `auxiliary_metrics = True`)

        Returns
        -------
        Returns dict where keys = metrics, values = performance along each metric. To get the `eval_metric` score, do `output[predictor.eval_metric.name]`
        NOTE: Metrics scores always show in higher is better form.
        This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
        """
        self._assert_is_fit('evaluate')
        data = self.__get_dataset(data)
        y_pred_proba = self.predict_proba(data=data, model=model)
        if self.sample_weight is not None and self.weight_evaluation and self.sample_weight in data:
            sample_weight = data[self.sample_weight]
        else:
            sample_weight = None
        return self.evaluate_predictions(y_true=data[self.label], y_pred=y_pred_proba, sample_weight=sample_weight, silent=silent,
                                         auxiliary_metrics=auxiliary_metrics, detailed_report=detailed_report)

    def evaluate_predictions(self, y_true, y_pred, sample_weight=None, silent=False, auxiliary_metrics=True, detailed_report=False) -> dict:
        """
        Evaluate the provided prediction probabilities against ground truth labels.
        Evaluation is based on the `eval_metric` previously specified in init, or default metrics if none was specified.

        Parameters
        ----------
        y_true : :class:`np.array` or :class:`pd.Series`
            The ordered collection of ground-truth labels.
        y_pred : :class:`pd.Series` or :class:`pd.DataFrame`
            The ordered collection of prediction probabilities or predictions.
            Obtainable via the output of `predictor.predict_proba`.
            Caution: For certain types of `eval_metric` (such as 'roc_auc'), `y_pred` must be predicted-probabilities rather than predicted labels.
        sample_weight : :class:`pd.Series`, default = None
            Sample weight for each row of data. If None, uniform sample weights are used.
        silent : bool, default = False
            If False, performance results are printed.
        auxiliary_metrics: bool, default = True
            Should we compute other (`problem_type` specific) metrics in addition to the default metric?
        detailed_report : bool, default = False
            Should we computed more detailed versions of the `auxiliary_metrics`? (requires `auxiliary_metrics = True`)

        Returns
        -------
        Returns dict where keys = metrics, values = performance along each metric.
        NOTE: Metrics scores always show in higher is better form.
        This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
        """
        return self._learner.evaluate_predictions(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, silent=silent,
                                                  auxiliary_metrics=auxiliary_metrics, detailed_report=detailed_report)

    def leaderboard(self, data=None, extra_info=False, extra_metrics=None, only_pareto_frontier=False, silent=False):
        """
        Output summary of information about models produced during `fit()` as a :class:`pd.DataFrame`.
        Includes information on test and validation scores for all models, model training times, inference times, and stack levels.
        Output DataFrame columns include:
            'model': The name of the model.

            'score_val': The validation score of the model on the 'eval_metric'.
                NOTE: Metrics scores always show in higher is better form.
                This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
                This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.
            'pred_time_val': The inference time required to compute predictions on the validation data end-to-end.
                Equivalent to the sum of all 'pred_time_val_marginal' values for the model and all of its base models.
            'fit_time': The fit time required to train the model end-to-end (Including base models if the model is a stack ensemble).
                Equivalent to the sum of all 'fit_time_marginal' values for the model and all of its base models.
            'pred_time_val_marginal': The inference time required to compute predictions on the validation data (Ignoring inference times for base models).
                Note that this ignores the time required to load the model into memory when bagging is disabled.
            'fit_time_marginal': The fit time required to train the model (Ignoring base models).
            'stack_level': The stack level of the model.
                A model with stack level N can take any set of models with stack level less than N as input, with stack level 1 models having no model inputs.
            'can_infer': If model is able to perform inference on new data. If False, then the model either was not saved, was deleted, or an ancestor of the model cannot infer.
                `can_infer` is often False when `save_bag_folds=False` was specified in initial `fit()`.
            'fit_order': The order in which models were fit. The first model fit has `fit_order=1`, and the Nth model fit has `fit_order=N`. The order corresponds to the first child model fit in the case of bagged ensembles.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame` (optional)
            This Dataset must also contain the label-column with the same column-name as specified during fit().
            If specified, then the leaderboard returned will contain additional columns 'score_test', 'pred_time_test', and 'pred_time_test_marginal'.
                'score_test': The score of the model on the 'eval_metric' for the data provided.
                    NOTE: Metrics scores always show in higher is better form.
                    This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
                    This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.
                'pred_time_test': The true end-to-end wall-clock inference time of the model for the data provided.
                    Equivalent to the sum of all 'pred_time_test_marginal' values for the model and all of its base models.
                'pred_time_test_marginal': The inference time of the model for the data provided, minus the inference time for the model's base models, if it has any.
                    Note that this ignores the time required to load the model into memory when bagging is disabled.
            If str is passed, `data` will be loaded using the str value as the file path.
        extra_info : bool, default = False
            If `True`, will return extra columns with advanced info.
            This requires additional computation as advanced info data is calculated on demand.
            Additional output columns when `extra_info=True` include:
                'num_features': Number of input features used by the model.
                    Some models may ignore certain features in the preprocessed data.
                'num_models': Number of models that actually make up this "model" object.
                    For non-bagged models, this is 1. For bagged models, this is equal to the number of child models (models trained on bagged folds) the bagged ensemble contains.
                'num_models_w_ancestors': Equivalent to the sum of 'num_models' values for the model and its' ancestors (see below).
                'memory_size': The amount of memory in bytes the model requires when persisted in memory. This is not equivalent to the amount of memory the model may use during inference.
                    For bagged models, this is the sum of the 'memory_size' of all child models.
                'memory_size_w_ancestors': Equivalent to the sum of 'memory_size' values for the model and its' ancestors.
                    This is the amount of memory required to avoid loading any models in-between inference calls to get predictions from this model.
                    For online-inference, this is critical. It is important that the machine performing online inference has memory more than twice this value to avoid loading models for every call to inference by persisting models in memory.
                'memory_size_min': The amount of memory in bytes the model minimally requires to perform inference.
                    For non-bagged models, this is equivalent to 'memory_size'.
                    For bagged models, this is equivalent to the largest child model's 'memory_size_min'.
                    To minimize memory usage, child models can be loaded and un-persisted one by one to infer. This is the default behavior if a bagged model was not already persisted in memory prior to inference.
                'memory_size_min_w_ancestors': Equivalent to the max of the 'memory_size_min' values for the model and its' ancestors.
                    This is the minimum required memory to infer with the model by only loading one model at a time, as each of its ancestors will also have to be loaded into memory.
                    For offline-inference where latency is not a concern, this should be used to determine the required memory for a machine if 'memory_size_w_ancestors' is too large.
                'num_ancestors': Number of ancestor models for the given model.

                'num_descendants': Number of descendant models for the given model.

                'model_type': The type of the given model.
                    If the model is an ensemble type, 'child_model_type' will indicate the inner model type. A stack ensemble of bagged LightGBM models would have 'StackerEnsembleModel' as its model type.
                'child_model_type': The child model type. None if the model is not an ensemble. A stack ensemble of bagged LightGBM models would have 'LGBModel' as its child type.
                    child models are models which are used as a group to generate a given bagged ensemble model's predictions. These are the models trained on each fold of a bagged ensemble.
                    For 10-fold bagging, the bagged ensemble model would have 10 child models.
                    For 10-fold bagging with 3 repeats, the bagged ensemble model would have 30 child models.
                    Note that child models are distinct from ancestors and descendants.
                'hyperparameters': The hyperparameter values specified for the model.
                    All hyperparameters that do not appear in this dict remained at their default values.
                'hyperparameters_fit': The hyperparameters set by the model during fit.
                    This overrides the 'hyperparameters' value for a particular key if present in 'hyperparameters_fit' to determine the fit model's final hyperparameters.
                    This is most commonly set for hyperparameters that indicate model training iterations or epochs, as early stopping can find a different value from what 'hyperparameters' indicated.
                    In these cases, the provided hyperparameter in 'hyperparameters' is used as a maximum for the model, but the model is still able to early stop at a smaller value during training to achieve a better validation score or to satisfy time constraints.
                    For example, if a NN model was given `epochs=500` as a hyperparameter, but found during training that `epochs=60` resulted in optimal validation score, it would use `epoch=60` and `hyperparameters_fit={'epoch': 60}` would be set.
                'ag_args_fit': Special AutoGluon arguments that influence model fit.
                    See the documentation of the `hyperparameters` argument in `TabularPredictor.fit()` for more information.
                'features': List of feature names used by the model.

                'child_hyperparameters': Equivalent to 'hyperparameters', but for the model's children.

                'child_hyperparameters_fit': Equivalent to 'hyperparameters_fit', but for the model's children.

                'child_ag_args_fit': Equivalent to 'ag_args_fit', but for the model's children.

                'ancestors': The model's ancestors. Ancestor models are the models which are required to make predictions during the construction of the model's input features.
                    If A is an ancestor of B, then B is a descendant of A.
                    If a model's ancestor is deleted, the model is no longer able to infer on new data, and its 'can_infer' value will be False.
                    A model can only have ancestor models whose 'stack_level' are lower than itself.
                    'stack_level'=1 models have no ancestors.
                'descendants': The model's descendants. Descendant models are the models which require this model to make predictions during the construction of their input features.
                    If A is a descendant of B, then B is an ancestor of A.
                    If this model is deleted, then all descendant models will no longer be able to infer on new data, and their 'can_infer' values will be False.
                    A model can only have descendant models whose 'stack_level' are higher than itself.
        extra_metrics : list, default = None
            A list of metrics to calculate scores for and include in the output DataFrame.
            Only valid when `data` is specified. The scores refer to the scores on `data` (same data as used to calculate the `score_test` column).
            This list can contain any values which would also be valid for `eval_metric` in predictor init.
            For example, `extra_metrics=['accuracy', 'roc_auc', 'log_loss']` would be valid in binary classification.
            This example would return 3 additional columns in the output DataFrame, whose column names match the names of the metrics.
            Passing `extra_metrics=[predictor.eval_metric]` would return an extra column in the name of the eval metric that has identical values to `score_test`.
            This also works with custom metrics. If passing an object instead of a string, the column name will be equal to the `.name` attribute of the object.
            NOTE: Metrics scores always show in higher is better form.
            This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.
        only_pareto_frontier : bool, default = False
            If `True`, only return model information of models in the Pareto frontier of the accuracy/latency trade-off (models which achieve the highest score within their end-to-end inference time).
            At minimum this will include the model with the highest score and the model with the lowest inference time.
            This is useful when deciding which model to use during inference if inference time is a consideration.
            Models filtered out by this process would never be optimal choices for a user that only cares about model inference time and score.
        silent : bool, default = False
            Should leaderboard DataFrame be printed?

        Returns
        -------
        :class:`pd.DataFrame` of model performance summary information.
        """
        self._assert_is_fit('leaderboard')
        data = self.__get_dataset(data) if data is not None else data
        return self._learner.leaderboard(X=data, extra_info=extra_info, extra_metrics=extra_metrics,
                                         only_pareto_frontier=only_pareto_frontier, silent=silent)

    def fit_summary(self, verbosity=3, show_plot=False):
        """
        Output summary of information about models produced during `fit()`.
        May create various generated summary plots and store them in folder: `predictor.path`.

        Parameters
        ----------
        verbosity : int, default = 3
            Controls how detailed of a summary to output.
            Set <= 0 for no output printing, 1 to print just high-level summary,
            2 to print summary and create plots, >= 3 to print all information produced during `fit()`.
        show_plot : bool, default = False
            If True, shows the model summary plot in browser when verbosity > 1.

        Returns
        -------
        Dict containing various detailed information. We do not recommend directly printing this dict as it may be very large.
        """
        self._assert_is_fit('fit_summary')
        # hpo_used = len(self._trainer.hpo_results) > 0
        hpo_used = False  # Disabled until a more memory efficient hpo_results object is implemented.
        model_types = self._trainer.get_models_attribute_dict(attribute='type')
        model_inner_types = self._trainer.get_models_attribute_dict(attribute='type_inner')
        model_typenames = {key: model_types[key].__name__ for key in model_types}
        model_innertypenames = {key: model_inner_types[key].__name__ for key in model_types if key in model_inner_types}
        MODEL_STR = 'Model'
        ENSEMBLE_STR = 'Ensemble'
        for model in model_typenames:
            if (model in model_innertypenames) and (ENSEMBLE_STR not in model_innertypenames[model]) and (
                    ENSEMBLE_STR in model_typenames[model]):
                new_model_typename = model_typenames[model] + "_" + model_innertypenames[model]
                if new_model_typename.endswith(MODEL_STR):
                    new_model_typename = new_model_typename[:-len(MODEL_STR)]
                model_typenames[model] = new_model_typename

        unique_model_types = set(model_typenames.values())  # no more class info
        # all fit() information that is returned:
        results = {
            'model_types': model_typenames,  # dict with key = model-name, value = type of model (class-name)
            'model_performance': self._trainer.get_models_attribute_dict('val_score'),
            # dict with key = model-name, value = validation performance
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.get_models_attribute_dict('path'),
            # dict with key = model-name, value = path to model file
            'model_fit_times': self._trainer.get_models_attribute_dict('fit_time'),
            'model_pred_times': self._trainer.get_models_attribute_dict('predict_time'),
            'num_bag_folds': self._trainer.k_fold,
            'max_stack_level': self._trainer.get_max_level(),
        }
        if self.problem_type == QUANTILE:
            results['num_quantiles'] = len(self.quantile_levels)
        elif self.problem_type != REGRESSION:
            results['num_classes'] = self._trainer.num_classes
        # if hpo_used:
        #     results['hpo_results'] = self._trainer.hpo_results
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in self._trainer.get_model_names():
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params
        results['model_hyperparams'] = model_hyperparams

        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            results['leaderboard'] = self._learner.leaderboard(silent=False)
            # self._summarize('model_performance', 'Validation performance of individual models', results)
            #  self._summarize('model_best', 'Best model (based on validation performance)', results)
            # self._summarize('hyperparameter_tune', 'Hyperparameter-tuning used', results)
            print("Number of models trained: %s" % len(results['model_performance']))
            print("Types of models trained:")
            print(unique_model_types)
            num_fold_str = ""
            bagging_used = results['num_bag_folds'] > 0
            if bagging_used:
                num_fold_str = f" (with {results['num_bag_folds']} folds)"
            print("Bagging used: %s %s" % (bagging_used, num_fold_str))
            num_stack_str = ""
            stacking_used = results['max_stack_level'] > 2
            if stacking_used:
                num_stack_str = f" (with {results['max_stack_level']} levels)"
            print("Multi-layer stack-ensembling used: %s %s" % (stacking_used, num_stack_str))
            hpo_str = ""
            # if hpo_used and verbosity <= 2:
            #     hpo_str = " (call fit_summary() with verbosity >= 3 to see detailed HPO info)"
            # print("Hyperparameter-tuning used: %s %s" % (hpo_used, hpo_str))
            # TODO: uncomment once feature_prune is functional:  self._summarize('feature_prune', 'feature-selection used', results)
            print("Feature Metadata (Processed):")
            print("(raw dtype, special dtypes):")
            print(self.feature_metadata)
        if verbosity > 1:  # create plots
            plot_tabular_models(results, output_directory=self.path,
                                save_file="SummaryOfModels.html",
                                plot_title="Models produced during fit()",
                                show_plot=show_plot)
            if hpo_used:
                for model_type in results['hpo_results']:
                    if 'trial_info' in results['hpo_results'][model_type]:
                        plot_summary_of_models(
                            results['hpo_results'][model_type],
                            output_directory=self.path, save_file=model_type + "_HPOmodelsummary.html",
                            plot_title=f"Models produced during {model_type} HPO", show_plot=show_plot)
                        plot_performance_vs_trials(
                            results['hpo_results'][model_type],
                            output_directory=self.path, save_file=model_type + "_HPOperformanceVStrials.png",
                            plot_title=f"HPO trials for {model_type} models", show_plot=show_plot)
        if verbosity > 2:  # print detailed information
            if hpo_used:
                hpo_results = results['hpo_results']
                print("*** Details of Hyperparameter optimization ***")
                for model_type in hpo_results:
                    hpo_model = hpo_results[model_type]
                    if 'trial_info' in hpo_model:
                        print(
                            f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s, Search strategy = {hpo_model['search_strategy']}")
                        print(
                            f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):")
                        print(hpo_model['best_config'])
            """
            if bagging_used:
                pass # TODO: print detailed bagging info
            if stacking_used:
                pass # TODO: print detailed stacking info, like how much it improves validation performance
            if results['feature_prune']:
                pass # TODO: print detailed feature-selection info once feature-selection is functional.
            """
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results

    def transform_features(self, data=None, model=None, base_models=None, return_original_features=True):
        """
        Transforms data features through the AutoGluon feature generator.
        This is useful to gain an understanding of how AutoGluon interprets the data features.
        The output of this function can be used to train further models, even outside of AutoGluon.
        This can be useful for training your own models on the same data representation as AutoGluon.
        Individual AutoGluon models like the neural network may apply additional feature transformations that are not reflected in this method.
        This method only applies universal transforms employed by all AutoGluon models.
        When `data=None`, `base_models=[{best_model}], and bagging was enabled during fit():
            This returns the out-of-fold predictions of the best model, which can be used as training input to a custom user stacker model.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame` (optional)
            The data to apply feature transformation to.
            This data does not require the label column.
            If str is passed, `data` will be loaded using the str value as the file path.
            If not specified, the original data used during fit() will be used if fit() was previously called with `cache_data=True`. Otherwise, an exception will be raised.
                For non-bagged mode predictors:
                    The data used when not specified is the validation set.
                    This can either be an automatically generated validation set or the user-defined `tuning_data` if passed during fit().
                    If all parameters are unspecified, then the output is equivalent to `predictor.load_data_internal(data='val', return_X=True, return_y=False)[0]`.
                    To get the label values of the output, call `predictor.load_data_internal(data='val', return_X=False, return_y=True)[1]`.
                    If the original training set is desired, it can be passed in through `data`.
                        Warning: Do not pass the original training set if `model` or `base_models` are set. This will result in overfit feature transformation.
                For bagged mode predictors:
                    The data used when not specified is the full training set.
                    If all parameters are unspecified, then the output is equivalent to `predictor.load_data_internal(data='train', return_X=True, return_y=False)[0]`.
                    To get the label values of the output, call `predictor.load_data_internal(data='train', return_X=False, return_y=True)[1]`.
                    `base_model` features generated in this instance will be from out-of-fold predictions.
                    Note that the training set may differ from the training set originally passed during fit(), as AutoGluon may choose to drop or duplicate rows during training.
                    Warning: Do not pass the original training set through `data` if `model` or `base_models` are set. This will result in overfit feature transformation. Instead set `data=None`.
        model : str, default = None
            Model to generate input features for.
            The output data will be equivalent to the input data that would be sent into `model.predict_proba(data)`.
                Note: This only applies to cases where `data` is not the training data.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
            Specifying a `refit_full` model will cause an exception if `data=None`.
            `base_models=None` is a requirement when specifying `model`.
        base_models : list, default = None
            List of model names to use as base_models for a hypothetical stacker model when generating input features.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
            If a stacker model S exists with `base_models=M`, then setting `base_models=M` is equivalent to setting `model=S`.
            `model=None` is a requirement when specifying `base_models`.
        return_original_features : bool, default = True
            Whether to return the original features.
            If False, only returns the additional output columns from specifying `model` or `base_models`.
                This is useful to set to False if the intent is to use the output as input to further stacker models without the original features.

        Returns
        -------
        :class:`pd.DataFrame` of the provided `data` after feature transformation has been applied.
        This output does not include the label column, and will remove it if present in the supplied `data`.
        If a transformed label column is desired, use `predictor.transform_labels`.

        Examples
        --------
        >>> from autogluon.tabular import TabularPredictor
        >>> predictor = TabularPredictor(label='class').fit('train.csv', label='class', auto_stack=True)  # predictor is in bagged mode.
        >>> model = 'WeightedEnsemble_L2'
        >>> train_data_transformed = predictor.transform_features(model=model)  # Internal training DataFrame used as input to `model.fit()` for each model trained in predictor.fit()`
        >>> test_data_transformed = predictor.transform_features('test.csv', model=model)  # Internal test DataFrame used as input to `model.predict_proba()` during `predictor.predict_proba(test_data, model=model)`

        """
        self._assert_is_fit('transform_features')
        data = self.__get_dataset(data) if data is not None else data
        return self._learner.get_inputs_to_stacker(dataset=data, model=model, base_models=base_models,
                                                   use_orig_features=return_original_features)

    def transform_labels(self, labels, inverse=False, proba=False):
        """
        Transforms data labels to the internal label representation.
        This can be useful for training your own models on the same data label representation as AutoGluon.
        Regression problems do not differ between original and internal representation, and thus this method will return the provided labels.
        Warning: When `inverse=False`, it is possible for the output to contain NaN label values in multiclass problems if the provided label was dropped during training.

        Parameters
        ----------
        labels : :class:`np.ndarray` or :class:`pd.Series`
            Labels to transform.
            If `proba=False`, an example input would be the output of `predictor.predict(test_data)`.
            If `proba=True`, an example input would be the output of `predictor.predict_proba(test_data, as_multiclass=False)`.
        inverse : boolean, default = False
            When `True`, the input labels are treated as being in the internal representation and the original representation is outputted.
        proba : boolean, default = False
            When `True`, the input labels are treated as probabilities and the output will be the internal representation of probabilities.
                In this case, it is expected that `labels` be a :class:`pd.DataFrame` or :class:`np.ndarray`.
                If the `problem_type` is multiclass:
                    The input column order must be equal to `predictor.class_labels`.
                    The output column order will be equal to `predictor.class_labels_internal`.
                    if `inverse=True`, the same logic applies, but with input and output columns interchanged.
            When `False`, the input labels are treated as actual labels and the output will be the internal representation of the labels.
                In this case, it is expected that `labels` be a :class:`pd.Series` or :class:`np.ndarray`.

        Returns
        -------
        :class:`pd.Series` of labels if `proba=False` or :class:`pd.DataFrame` of label probabilities if `proba=True`.

        """
        self._assert_is_fit('transform_labels')
        if inverse:
            if proba:
                labels_transformed = self._learner.label_cleaner.inverse_transform_proba(y=labels, as_pandas=True)
            else:
                labels_transformed = self._learner.label_cleaner.inverse_transform(y=labels)
        else:
            if proba:
                labels_transformed = self._learner.label_cleaner.transform_proba(y=labels, as_pandas=True)
            else:
                labels_transformed = self._learner.label_cleaner.transform(y=labels)
        return labels_transformed

    def feature_importance(self, data=None, model=None, features=None, feature_stage='original', subsample_size=5000,
                           time_limit=None, num_shuffle_sets=None, include_confidence_band=True, confidence_level=0.99,
                           silent=False):
        """
        Calculates feature importance scores for the given model via permutation importance. Refer to https://explained.ai/rf-importance/ for an explanation of permutation importance.
        A feature's importance score represents the performance drop that results when the model makes predictions on a perturbed copy of the data where this feature's values have been randomly shuffled across rows.
        A feature score of 0.01 would indicate that the predictive performance dropped by 0.01 when the feature was randomly shuffled.
        The higher the score a feature has, the more important it is to the model's performance.
        If a feature has a negative score, this means that the feature is likely harmful to the final model, and a model trained with the feature removed would be expected to achieve a better predictive performance.
        Note that calculating feature importance can be a very computationally expensive process, particularly if the model uses hundreds or thousands of features. In many cases, this can take longer than the original model training.
        To estimate how long `feature_importance(model, data, features)` will take, it is roughly the time taken by `predict_proba(data, model)` multiplied by the number of features.

        Note: For highly accurate importance and p_value estimates, it is recommended to set `subsample_size` to at least 5000 if possible and `num_shuffle_sets` to at least 10.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame` (optional)
            This data must also contain the label-column with the same column-name as specified during `fit()`.
            If specified, then the data is used to calculate the feature importance scores.
            If str is passed, `data` will be loaded using the str value as the file path.
            If not specified, the original data used during `fit()` will be used if `cache_data=True`. Otherwise, an exception will be raised.
            Do not pass the training data through this argument, as the feature importance scores calculated will be biased due to overfitting.
                More accurate feature importances will be obtained from new data that was held-out during `fit()`.
        model : str, default = None
            Model to get feature importances for, if None the best model is chosen.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        features : list, default = None
            List of str feature names that feature importances are calculated for and returned, specify None to get all feature importances.
            If you only want to compute feature importances for some of the features, you can pass their names in as a list of str.
            Valid feature names change depending on the `feature_stage`.
                To get the list of feature names for `feature_stage='original'`, call `predictor.feature_metadata_in.get_features()`.
                To get the list of feature names for `feature_stage='transformed'`, call `list(predictor.transform_features().columns)`.
                To get the list of feature names for `feature_stage=`transformed_model`, call `list(predictor.transform_features(model={model_name}).columns)`.
            [Advanced] Can also contain tuples as elements of (feature_name, feature_list) form.
                feature_name can be any string so long as it is unique with all other feature names / features in the list.
                feature_list can be any list of valid features in the data.
                This will compute importance of the combination of features in feature_list, naming the set of features in the returned DataFrame feature_name.
                This importance will differ from adding the individual importances of each feature in feature_list, and will be more accurate to the overall group importance.
                Example: ['featA', 'featB', 'featC', ('featBC', ['featB', 'featC'])]
                In this example, the importance of 'featBC' will be calculated by jointly permuting 'featB' and 'featC' together as if they were a single two-dimensional feature.
        feature_stage : str, default = 'original'
            What stage of feature-processing should importances be computed for.
            Options:
                'original':
                    Compute importances of the original features.
                    Warning: `data` must be specified with this option, otherwise an exception will be raised.
                'transformed':
                    Compute importances of the post-internal-transformation features (after automated feature engineering). These features may be missing some original features, or add new features entirely.
                    An example of new features would be ngram features generated from a text column.
                    Warning: For bagged models, feature importance calculation is not yet supported with this option when `data=None`. Doing so will raise an exception.
                'transformed_model':
                    Compute importances of the post-model-transformation features. These features are the internal features used by the requested model. They may differ greatly from the original features.
                    If the model is a stack ensemble, this will include stack ensemble features such as the prediction probability features of the stack ensemble's base (ancestor) models.
        subsample_size : int, default = 5000
            The number of rows to sample from `data` when computing feature importance.
            If `subsample_size=None` or `data` contains fewer than `subsample_size` rows, all rows will be used during computation.
            Larger values increase the accuracy of the feature importance scores.
            Runtime linearly scales with `subsample_size`.
        time_limit : float, default = None
            Time in seconds to limit the calculation of feature importance.
            If None, feature importance will calculate without early stopping.
            A minimum of 1 full shuffle set will always be evaluated. If a shuffle set evaluation takes longer than `time_limit`, the method will take the length of a shuffle set evaluation to return regardless of the `time_limit`.
        num_shuffle_sets : int, default = None
            The number of different permutation shuffles of the data that are evaluated.
            Larger values will increase the quality of the importance evaluation.
            It is generally recommended to increase `subsample_size` before increasing `num_shuffle_sets`.
            Defaults to 5 if `time_limit` is None or 10 if `time_limit` is specified.
            Runtime linearly scales with `num_shuffle_sets`.
        include_confidence_band: bool, default = True
            If True, returned DataFrame will include two additional columns specifying confidence interval for the true underlying importance value of each feature.
            Increasing `subsample_size` and `num_shuffle_sets` will tighten the confidence interval.
        confidence_level: float, default = 0.99
            This argument is only considered when `include_confidence_band` is True, and can be used to specify the confidence level used for constructing confidence intervals.  
            For example, if `confidence_level` is set to 0.99, then the returned DataFrame will include columns 'p99_high' and 'p99_low' which indicates that the true feature importance will be between 'p99_high' and 'p99_low' 99% of the time (99% confidence interval).
            More generally, if `confidence_level` = 0.XX, then the columns containing the XX% confidence interval will be named 'pXX_high' and 'pXX_low'. 
        silent : bool, default = False
            Whether to suppress logging output.

        Returns
        -------
        :class:`pd.DataFrame` of feature importance scores with 6 columns:
            index: The feature name.
            'importance': The estimated feature importance score.
            'stddev': The standard deviation of the feature importance score. If NaN, then not enough num_shuffle_sets were used to calculate a variance.
            'p_value': P-value for a statistical t-test of the null hypothesis: importance = 0, vs the (one-sided) alternative: importance > 0.
                Features with low p-value appear confidently useful to the predictor, while the other features may be useless to the predictor (or even harmful to include in its training data).
                A p-value of 0.01 indicates that there is a 1% chance that the feature is useless or harmful, and a 99% chance that the feature is useful.
                A p-value of 0.99 indicates that there is a 99% chance that the feature is useless or harmful, and a 1% chance that the feature is useful.
            'n': The number of shuffles performed to estimate importance score (corresponds to sample-size used to determine confidence interval for true score).
            'pXX_high': Upper end of XX% confidence interval for true feature importance score (where XX=99 by default).
            'pXX_low': Lower end of XX% confidence interval for true feature importance score.
        """
        self._assert_is_fit('feature_importance')
        data = self.__get_dataset(data) if data is not None else data
        if (data is None) and (not self._trainer.is_data_saved):
            raise AssertionError(
                'No data was provided and there is no cached data to load for feature importance calculation. `cache_data=True` must be set in the `TabularPredictor` init `learner_kwargs` argument call to enable this functionality when data is not specified.')
        if data is not None:
            self._validate_unique_indices(data, 'data')

        if num_shuffle_sets is None:
            num_shuffle_sets = 10 if time_limit else 5

        fi_df = self._learner.get_feature_importance(model=model, X=data, features=features,
                                                     feature_stage=feature_stage,
                                                     subsample_size=subsample_size, time_limit=time_limit,
                                                     num_shuffle_sets=num_shuffle_sets, silent=silent)

        if include_confidence_band:
            if confidence_level <= 0.5 or confidence_level >= 1.0:
                raise ValueError("confidence_level must lie between 0.5 and 1.0")
            ci_str = "{:0.0f}".format(confidence_level * 100)
            import scipy.stats
            num_features = len(fi_df)
            ci_low_dict = dict()
            ci_high_dict = dict()
            for i in range(num_features):
                fi = fi_df.iloc[i]
                mean = fi['importance']
                stddev = fi['stddev']
                n = fi['n']
                if stddev == np.nan or n == np.nan or mean == np.nan or n == 1:
                    ci_high = np.nan
                    ci_low = np.nan
                else:
                    t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                    ci_high = mean + t_val * stddev / math.sqrt(n)
                    ci_low = mean - t_val * stddev / math.sqrt(n)
                ci_high_dict[fi.name] = ci_high
                ci_low_dict[fi.name] = ci_low
            high_str = 'p' + ci_str + '_high'
            low_str = 'p' + ci_str + '_low'
            fi_df[high_str] = pd.Series(ci_high_dict)
            fi_df[low_str] = pd.Series(ci_low_dict)
        return fi_df

    def persist_models(self, models='best', with_ancestors=True, max_memory=0.1) -> list:
        """
        Persist models in memory for reduced inference latency. This is particularly important if the models are being used for online-inference where low latency is critical.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.

        Parameters
        ----------
        models : list of str or str, default = 'best'
            Model names of models to persist.
            If 'best' then the model with the highest validation score is persisted (this is the model used for prediction by default).
            If 'all' then all models are persisted.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
        with_ancestors : bool, default = True
            If True, all ancestor models of the provided models will also be persisted.
            If False, stacker models will not have the models they depend on persisted unless those models were specified in `models`. This will slow down inference as the ancestor models will still need to be loaded from disk for each predict call.
            Only relevant for stacker models.
        max_memory : float, default = 0.1
            Proportion of total available memory to allow for the persisted models to use.
            If the models' summed memory usage requires a larger proportion of memory than max_memory, they are not persisted. In this case, the output will be an empty list.
            If None, then models are persisted regardless of estimated memory usage. This can cause out-of-memory errors.

        Returns
        -------
        List of persisted model names.
        """
        self._assert_is_fit('persist_models')
        return self._learner.persist_trainer(low_memory=False, models=models, with_ancestors=with_ancestors,
                                             max_memory=max_memory)

    def unpersist_models(self, models='all') -> list:
        """
        Unpersist models in memory for reduced memory usage.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.
        Note: Another way to reset the predictor and unpersist models is to reload the predictor from disk via `predictor = TabularPredictor.load(predictor.path)`.

        Parameters
        ----------
        models : list of str or str, default = 'all'
            Model names of models to unpersist.
            If 'all' then all models are unpersisted.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names_persisted()`.

        Returns
        -------
        List of unpersisted model names.
        """
        self._assert_is_fit('unpersist_models')
        return self._learner.load_trainer().unpersist_models(model_names=models)

    def refit_full(self, model='all', set_best_to_refit_full=True):
        """
        Retrain model on all of the data (training + validation).
        For bagged models:
            Optimizes a model's inference time by collapsing bagged ensembles into a single model fit on all of the training data.
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
        `cache_data` must have been set to `True` during the original training to enable this functionality.

        Parameters
        ----------
        model : str, default = 'all'
            Model name of model to refit.
                If 'all' then all models are refitted.
                If 'best' then the model with the highest validation score is refit.
            All ancestor models will also be refit in the case that the selected model is a weighted or stacker ensemble.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
        set_best_to_refit_full : bool, default = True
            If True, sets best model to the refit_full version of the prior best model.
            This means the model used when `predictor.predict(data)` is called will be the refit_full version instead of the original version of the model.
            Ignored if `model` is not the best model.

        Returns
        -------
        Dictionary of original model names -> refit_full model names.
        """
        self._assert_is_fit('refit_full')
        model_best = self._get_model_best(can_infer=None)
        refit_full_dict = self._learner.refit_ensemble_full(model=model)

        if set_best_to_refit_full:
            if model_best in self._trainer.model_full_dict.keys():
                self._trainer.model_best = self._trainer.model_full_dict[model_best]
                # Note: model_best will be overwritten if additional training is done with new models,
                # since model_best will have validation score of None and any new model will have a better validation score.
                # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                self._trainer.save()
                logger.log(20, f'Updated best model to "{self._trainer.model_best}" (Previously "{model_best}"). '
                               f'AutoGluon will default to using "{self._trainer.model_best}" for predict() and predict_proba().')
            else:
                logger.warning(
                    f'Best model ("{model_best}") is not present in refit_full dictionary. '
                    f'Training may have failed on the refit model. AutoGluon will default to using "{model_best}" for predict() and predict_proba().')

        return refit_full_dict

    def get_model_best(self):
        """
        Returns the string model name of the best model by validation score that can infer.
        This is the same model used during inference when `predictor.predict` is called without specifying a model.
        This can be updated to be a model other than the model with best validation score by methods such as refit_full and set_model_best.

        Returns
        -------
        String model name of the best model
        """
        return self._get_model_best(can_infer=True)

    def _get_model_best(self, can_infer=None):
        self._assert_is_fit('get_model_best')
        # TODO: Set self._trainer.model_best to the best model at end of fit instead of best WeightedEnsemble.
        if self._trainer.model_best is not None:
            models = self._trainer.get_model_names(can_infer=can_infer)
            if self._trainer.model_best in models:
                return self._trainer.model_best
        return self._trainer.get_model_best(can_infer=can_infer)

    def set_model_best(self, model: str):
        """
        Sets the model to be used by default when calling `predictor.predict(data)`.
        By default, this is the model with the best validation score, but this is not always the case.
        If manually set, this can be overwritten internally if further training occurs, such as through fit_extra, refit_full, or distill.

        Parameters
        ----------
        model : str
            Name of model to set to best. If model does not exist or cannot infer, raises an AssertionError.
        """
        self._assert_is_fit('set_model_best')
        models = self._trainer.get_model_names(can_infer=True)
        if model in models:
            self._trainer.model_best = model
        else:
            raise AssertionError(f'Model "{model}" is not a valid model to specify as best! Valid models: {models}')

    def get_model_full_dict(self):
        """
        Returns a dictionary of original model name -> refit full model name.
        Empty unless `refit_full=True` was set during fit or `predictor.refit_full()` was called.
        This can be useful when determining the best model based off of `predictor.leaderboard()`, then getting the _FULL version of the model by passing its name as the key to this dictionary.

        Returns
        -------
        Dictionary of original model name -> refit full model name.
        """
        self._assert_is_fit('get_model_full_dict')
        return copy.deepcopy(self._trainer.model_full_dict)

    def info(self):
        """
        [EXPERIMENTAL] Returns a dictionary of `predictor` metadata.
        Warning: This functionality is currently in preview mode.
            The metadata information returned may change in structure in future versions without warning.
            The definitions of various metadata values are not yet documented.
            The output of this function should not be used for programmatic decisions.
        Contains information such as row count, column count, model training time, validation scores, hyperparameters, and much more.

        Returns
        -------
        Dictionary of `predictor` metadata.
        """
        self._assert_is_fit('info')
        return self._learner.get_info(include_model_info=True)

    # TODO: Add data argument
    # TODO: Add option to disable OOF generation of newly fitted models
    # TODO: Move code logic to learner/trainer
    # TODO: Add fit() arg to perform this automatically at end of training
    # TODO: Consider adding cutoff arguments such as top-k models
    def fit_weighted_ensemble(self, base_models: list = None, name_suffix='Best', expand_pareto_frontier=False,
                              time_limit=None):
        """
        Fits new weighted ensemble models to combine predictions of previously-trained models.
        `cache_data` must have been set to `True` during the original training to enable this functionality.

        Parameters
        ----------
        base_models : list, default = None
            List of model names the weighted ensemble can consider as candidates.
            If None, all previously trained models are considered except for weighted ensemble models.
            As an example, to train a weighted ensemble that can only have weights assigned to the models 'model_a' and 'model_b', set `base_models=['model_a', 'model_b']`
        name_suffix : str, default = 'Best'
            Name suffix to add to the name of the newly fitted ensemble model.
        expand_pareto_frontier : bool, default = False
            If True, will train N-1 weighted ensemble models instead of 1, where `N=len(base_models)`.
            The final model trained when True is equivalent to the model trained when False.
            These weighted ensemble models will attempt to expand the pareto frontier.
            This will create many different weighted ensembles which have different accuracy/memory/inference-speed trade-offs.
            This is particularly useful when inference speed is an important consideration.
        time_limit : int, default = None
            Time in seconds each weighted ensemble model is allowed to train for. If `expand_pareto_frontier=True`, the `time_limit` value is applied to each model.
            If None, the ensemble models train without time restriction.

        Returns
        -------
        List of newly trained weighted ensemble model names.
        If an exception is encountered while training an ensemble model, that model's name will be absent from the list.
        """
        self._assert_is_fit('fit_weighted_ensemble')
        trainer = self._learner.load_trainer()

        if trainer.bagged_mode:
            X = trainer.load_X()
            y = trainer.load_y()
            fit = True
        else:
            X = trainer.load_X_val()
            y = trainer.load_y_val()
            fit = False

        stack_name = 'aux1'
        if base_models is None:
            base_models = trainer.get_model_names(stack_name='core')

        X_stack_preds = trainer.get_inputs_to_stacker(X=X, base_models=base_models, fit=fit, use_orig_features=False)

        models = []

        if expand_pareto_frontier:
            leaderboard = self.leaderboard(silent=True)
            leaderboard = leaderboard[leaderboard['model'].isin(base_models)]
            leaderboard = leaderboard.sort_values(by='pred_time_val')
            models_to_check = leaderboard['model'].tolist()
            for i in range(1, len(models_to_check) - 1):
                models_to_check_now = models_to_check[:i + 1]
                max_base_model_level = max([trainer.get_model_level(base_model) for base_model in models_to_check_now])
                weighted_ensemble_level = max_base_model_level + 1
                models += trainer.generate_weighted_ensemble(X=X_stack_preds, y=y, level=weighted_ensemble_level,
                                                             stack_name=stack_name,
                                                             base_model_names=models_to_check_now,
                                                             name_suffix=name_suffix + '_Pareto' + str(i),
                                                             time_limit=time_limit)

        max_base_model_level = max([trainer.get_model_level(base_model) for base_model in base_models])
        weighted_ensemble_level = max_base_model_level + 1
        models += trainer.generate_weighted_ensemble(X=X_stack_preds, y=y, level=weighted_ensemble_level,
                                                     stack_name=stack_name, base_model_names=base_models,
                                                     name_suffix=name_suffix, time_limit=time_limit)

        return models

    def get_oof_pred(self, model: str = None, transformed=False, train_data=None, internal_oof=False) -> pd.Series:
        """
        Note: This is advanced functionality not intended for normal usage.

        Returns the out-of-fold (OOF) predictions for every row in the training data.

        For more information, refer to `get_oof_pred_proba()` documentation.

        Parameters
        ----------
        model : str (optional)
            Refer to `get_oof_pred_proba()` documentation.
        transformed : bool, default = False
            Refer to `get_oof_pred_proba()` documentation.
        train_data : pd.DataFrame, default = None
            Refer to `get_oof_pred_proba()` documentation.
        internal_oof : bool, default = False
            Refer to `get_oof_pred_proba()` documentation.

        Returns
        -------
        :class:`pd.Series` object of the out-of-fold training predictions of the model.
        """
        self._assert_is_fit('get_oof_pred')
        y_pred_proba_oof = self.get_oof_pred_proba(model=model,
                                                   transformed=transformed,
                                                   as_multiclass=True,
                                                   train_data=train_data,
                                                   internal_oof=internal_oof)
        y_pred_oof = get_pred_from_proba_df(y_pred_proba_oof, problem_type=self.problem_type)
        if transformed:
            return self._learner.label_cleaner.to_transformed_dtype(y_pred_oof)
        return y_pred_oof

    # TODO: Improve error messages when trying to get oof from refit_full and distilled models.
    # TODO: v0.1 add tutorial related to this method, as it is very powerful.
    # TODO: Remove train_data argument once we start caching the raw original data: Can just load that instead.
    def get_oof_pred_proba(self, model: str = None, transformed=False, as_multiclass=True, train_data=None,
                           internal_oof=False) -> Union[pd.DataFrame, pd.Series]:
        """
        Note: This is advanced functionality not intended for normal usage.

        Returns the out-of-fold (OOF) predicted class probabilities for every row in the training data.
        OOF prediction probabilities may provide unbiased estimates of generalization accuracy (reflecting how predictions will behave on new data)
        Predictions for each row are only made using models that were fit to a subset of data where this row was held-out.

        Warning: This method will raise an exception if called on a model that is not a bagged ensemble. Only bagged models (such a stacker models) can produce OOF predictions.
            This also means that refit_full models and distilled models will raise an exception.
        Warning: If intending to join the output of this method with the original training data, be aware that a rare edge-case issue exists:
            Multiclass problems with rare classes combined with the use of the 'log_loss' eval_metric may have forced AutoGluon to duplicate rows in the training data to satisfy minimum class counts in the data.
            If this has occurred, then the indices and row counts of the returned :class:`pd.Series` in this method may not align with the training data.
            In this case, consider fetching the processed training data using `predictor.load_data_internal()` instead of using the original training data.
            A more benign version of this issue occurs when 'log_loss' wasn't specified as the eval_metric but rare classes were dropped by AutoGluon.
            In this case, not all of the original training data rows will have an OOF prediction. It is recommended to either drop these rows during the join or to get direct predictions on the missing rows via :meth:`TabularPredictor.predict_proba`.

        Parameters
        ----------
        model : str (optional)
            The name of the model to get out-of-fold predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        transformed : bool, default = False
            Whether the output values should be of the original label representation (False) or the internal label representation (True).
            The internal representation for binary and multiclass classification are integers numbering the k possible classes from 0 to k-1, while the original representation is identical to the label classes provided during fit.
            Generally, most users will want the original representation and keep `transformed=False`.
        as_multiclass : bool, default = True
            Whether to return binary classification probabilities as if they were for multiclass classification.
                Output will contain two columns, and if `transformed=False`, the column names will correspond to the binary class labels.
                The columns will be the same order as `predictor.class_labels`.
            If False, output will contain only 1 column for the positive class (get positive_class name via `predictor.positive_class`).
            Only impacts output for binary classification problems.
        train_data : pd.DataFrame, default = None
            Specify the original `train_data` to ensure that any training rows that were originally dropped internally are properly handled.
            If None, then output will not contain all rows if training rows were dropped internally during fit.
        internal_oof : bool, default = False
            [Advanced Option] Return the internal OOF preds rather than the externally facing OOF preds.
            Internal OOF preds may have more/fewer rows than was provided in train_data, and are incompatible with external data.
            If you don't know what this does, keep it as False.

        Returns
        -------
        :class:`pd.Series` or :class:`pd.DataFrame` object of the out-of-fold training prediction probabilities of the model.
        """
        self._assert_is_fit('get_oof_pred_proba')
        if model is None:
            model = self.get_model_best()
        if not self._trainer.bagged_mode:
            raise AssertionError('Predictor must be in bagged mode to get out-of-fold predictions.')
        if model in self._trainer._model_full_dict_val_score:
            # FIXME: This is a hack, add refit tag in a nicer way than via the _model_full_dict_val_score
            # TODO: bagged-with-holdout refit to bagged-no-holdout should still be able to return out-of-fold predictions
            raise AssertionError('_FULL models do not have out-of-fold predictions.')
        if self._trainer.get_model_attribute_full(model=model, attribute='val_in_fit', func=max):
            raise AssertionError(
                f'Model {model} does not have out-of-fold predictions because it used a validation set during training.')
        y_pred_proba_oof_transformed = self.transform_features(base_models=[model], return_original_features=False)
        if not internal_oof:
            is_duplicate_index = y_pred_proba_oof_transformed.index.duplicated(keep='first')
            if is_duplicate_index.any():
                logger.log(20,
                           'Detected duplicate indices... This means that data rows may have been duplicated during training. '
                           'Removing all duplicates except for the first instance.')
                y_pred_proba_oof_transformed = y_pred_proba_oof_transformed[is_duplicate_index == False]
            if self._learner._pre_X_rows is not None and len(y_pred_proba_oof_transformed) < self._learner._pre_X_rows:
                len_diff = self._learner._pre_X_rows - len(y_pred_proba_oof_transformed)
                if train_data is None:
                    logger.warning(f'WARNING: {len_diff} rows of training data were dropped internally during fit. '
                                   f'The output will not contain all original training rows.\n'
                                   f'If attempting to get `oof_pred_proba`, DO NOT pass `train_data` into `predictor.predict_proba` or `predictor.transform_features`!\n'
                                   f'Instead this can be done by the following '
                                   f'(Ensure `train_data` is identical to when it was used in fit):\n'
                                   f'oof_pred_proba = predictor.get_oof_pred_proba(train_data=train_data)\n'
                                   f'oof_pred = predictor.get_oof_pred(train_data=train_data)\n')
                else:
                    missing_idx = list(train_data.index.difference(y_pred_proba_oof_transformed.index))
                    if len(missing_idx) > 0:
                        missing_idx_data = train_data.loc[missing_idx]
                        missing_pred_proba = self.transform_features(data=missing_idx_data, base_models=[model],
                                                                     return_original_features=False)
                        y_pred_proba_oof_transformed = pd.concat([y_pred_proba_oof_transformed, missing_pred_proba])
                        y_pred_proba_oof_transformed = y_pred_proba_oof_transformed.reindex(list(train_data.index))

        if self.problem_type == MULTICLASS and self._learner.label_cleaner.problem_type_transform == MULTICLASS:
            y_pred_proba_oof_transformed.columns = copy.deepcopy(
                self._learner.label_cleaner.ordered_class_labels_transformed)
        elif self.problem_type == QUANTILE:
            y_pred_proba_oof_transformed.columns = self.quantile_levels
        else:
            y_pred_proba_oof_transformed.columns = [self.label]
            y_pred_proba_oof_transformed = y_pred_proba_oof_transformed[self.label]
            if as_multiclass and self.problem_type == BINARY:
                y_pred_proba_oof_transformed = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(
                    y_pred_proba_oof_transformed, as_pandas=True)
            elif self.problem_type == MULTICLASS:
                if transformed:
                    y_pred_proba_oof_transformed = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(
                        y_pred_proba_oof_transformed, as_pandas=True)
                    y_pred_proba_oof_transformed.columns = copy.deepcopy(
                        self._learner.label_cleaner.ordered_class_labels_transformed)
        if transformed:
            return y_pred_proba_oof_transformed
        else:
            return self.transform_labels(labels=y_pred_proba_oof_transformed, inverse=True, proba=True)

    @property
    def positive_class(self):
        """
        Returns the positive class name in binary classification. Useful for computing metrics such as F1 which require a positive and negative class.
        In binary classification, :class:`TabularPredictor.predict_proba(as_multiclass=False)` returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        return self._learner.positive_class

    def load_data_internal(self, data='train', return_X=True, return_y=True):
        """
        Loads the internal data representation used during model training.
        Individual AutoGluon models like the neural network may apply additional feature transformations that are not reflected in this method.
        This method only applies universal transforms employed by all AutoGluon models.
        Warning, the internal representation may:
            Have different features compared to the original data.
            Have different row counts compared to the original data.
            Have indices which do not align with the original data.
            Have label values which differ from those in the original data.
        Internal data representations should NOT be combined with the original data, in most cases this is not possible.

        Parameters
        ----------
        data : str, default = 'train'
            The data to load.
            Valid values are:
                'train':
                    Load the training data used during model training.
                    This is a transformed and augmented version of the `train_data` passed in `fit()`.
                'val':
                    Load the validation data used during model training.
                    This is a transformed and augmented version of the `tuning_data` passed in `fit()`.
                    If `tuning_data=None` was set in `fit()`, then `tuning_data` is an automatically generated validation set created by splitting `train_data`.
                    Warning: Will raise an exception if called by a bagged predictor, as bagged predictors have no validation data.
        return_X : bool, default = True
            Whether to return the internal data features
            If set to `False`, then the first element in the returned tuple will be None.
        return_y : bool, default = True
            Whether to return the internal data labels
            If set to `False`, then the second element in the returned tuple will be None.

        Returns
        -------
        Tuple of (:class:`pd.DataFrame`, :class:`pd.Series`) corresponding to the internal data features and internal data labels, respectively.

        """
        self._assert_is_fit('load_data_internal')
        if data == 'train':
            load_X = self._trainer.load_X
            load_y = self._trainer.load_y
        elif data == 'val':
            load_X = self._trainer.load_X_val
            load_y = self._trainer.load_y_val
        else:
            raise ValueError(f'data must be one of: [\'train\', \'val\'], but was \'{data}\'.')
        X = load_X() if return_X else None
        y = load_y() if return_y else None
        return X, y

    def save_space(self, remove_data=True, remove_fit_stack=True, requires_save=True, reduce_children=False):
        """
        Reduces the memory and disk size of predictor by deleting auxiliary model files that aren't needed for prediction on new data.
        This function has NO impact on inference accuracy.
        It is recommended to invoke this method if the only goal is to use the trained model for prediction.
        However, certain advanced functionality may no longer be available after `save_space()` has been called.

        Parameters
        ----------
        remove_data : bool, default = True
            Whether to remove cached files of the original training and validation data.
            Only reduces disk usage, it has no impact on memory usage.
            This is especially useful when the original data was large.
            This is equivalent to setting `cache_data=False` during the original `fit()`.
                Will disable all advanced functionality that requires `cache_data=True`.
        remove_fit_stack : bool, default = True
            Whether to remove information required to fit new stacking models and continue fitting bagged models with new folds.
            Only reduces disk usage, it has no impact on memory usage.
            This includes:
                out-of-fold (OOF) predictions
            This is useful for multiclass problems with many classes, as OOF predictions can become very large on disk. (1 GB per model in extreme cases)
            This disables `predictor.refit_full()` for stacker models.
        requires_save : bool, default = True
            Whether to remove information that requires the model to be saved again to disk.
            Typically this only includes flag variables that don't have significant impact on memory or disk usage, but should technically be updated due to the removal of more important information.
                An example is the `is_data_saved` boolean variable in `trainer`, which should be updated to `False` if `remove_data=True` was set.
        reduce_children : bool, default = False
            Whether to apply the reduction rules to bagged ensemble children models. These are the models trained for each fold of the bagged ensemble.
            This should generally be kept as `False` since the most important memory and disk reduction techniques are automatically applied to these models during the original `fit()` call.

        """
        self._assert_is_fit('save_space')
        self._trainer.reduce_memory_size(remove_data=remove_data, remove_fit_stack=remove_fit_stack, remove_fit=True,
                                         remove_info=False, requires_save=requires_save,
                                         reduce_children=reduce_children)

    def delete_models(self, models_to_keep=None, models_to_delete=None, allow_delete_cascade=False,
                      delete_from_disk=True, dry_run=True):
        """
        Deletes models from `predictor`.
        This can be helpful to minimize memory usage and disk usage, particularly for model deployment.
        This will remove all references to the models in `predictor`.
            For example, removed models will not appear in `predictor.leaderboard()`.
        WARNING: If `delete_from_disk=True`, this will DELETE ALL FILES in the deleted model directories, regardless if they were created by AutoGluon or not.
            DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.

        Parameters
        ----------
        models_to_keep : str or list, default = None
            Name of model or models to not delete.
            All models that are not specified and are also not required as a dependency of any model in `models_to_keep` will be deleted.
            Specify `models_to_keep='best'` to keep only the best model and its model dependencies.
            `models_to_delete` must be None if `models_to_keep` is set.
            To see the list of possible model names, use: `predictor.get_model_names()` or `predictor.leaderboard()`.
        models_to_delete : str or list, default = None
            Name of model or models to delete.
            All models that are not specified but depend on a model in `models_to_delete` will also be deleted.
            `models_to_keep` must be None if `models_to_delete` is set.
        allow_delete_cascade : bool, default = False
            If `False`, if unspecified dependent models of models in `models_to_delete` exist an exception will be raised instead of deletion occurring.
                An example of a dependent model is m1 if m2 is a stacker model and takes predictions from m1 as inputs. In this case, m1 would be a dependent model of m2.
            If `True`, all dependent models of models in `models_to_delete` will be deleted.
            Has no effect if `models_to_delete=None`.
        delete_from_disk : bool, default = True
            If `True`, deletes the models from disk if they were persisted.
            WARNING: This deletes the entire directory for the deleted models, and ALL FILES located there.
                It is highly recommended to first run with `dry_run=True` to understand which directories will be deleted.
        dry_run : bool, default = True
            If `True`, then deletions don't occur, and logging statements are printed describing what would have occurred.
            Set `dry_run=False` to perform the deletions.

        """
        self._assert_is_fit('delete_models')
        if models_to_keep == 'best':
            models_to_keep = self.get_model_best()
        self._trainer.delete_models(models_to_keep=models_to_keep, models_to_delete=models_to_delete,
                                    allow_delete_cascade=allow_delete_cascade, delete_from_disk=delete_from_disk,
                                    dry_run=dry_run)

    # TODO: v0.1 add documentation for arguments
    def get_model_names(self, stack_name=None, level=None, can_infer: bool = None, models: list = None) -> list:
        """Returns the list of model names trained in this `predictor` object."""
        self._assert_is_fit('get_model_names')
        return self._trainer.get_model_names(stack_name=stack_name, level=level, can_infer=can_infer, models=models)

    def get_model_names_persisted(self) -> list:
        """Returns the list of model names which are persisted in memory."""
        self._assert_is_fit('get_model_names_persisted')
        return list(self._learner.load_trainer().models.keys())

    def distill(self, train_data=None, tuning_data=None, augmentation_data=None, time_limit=None, hyperparameters=None,
                holdout_frac=None,
                teacher_preds='soft', augment_method='spunge', augment_args={'size_factor': 5, 'max_size': int(1e5)},
                models_name_suffix=None, verbosity=None):
        """
        Distill AutoGluon's most accurate ensemble-predictor into single models which are simpler/faster and require less memory/compute.
        Distillation can produce a model that is more accurate than the same model fit directly on the original training data.
        After calling `distill()`, there will be more models available in this Predictor, which can be evaluated using `predictor.leaderboard(test_data)` and deployed with: `predictor.predict(test_data, model=MODEL_NAME)`.
        This will raise an exception if `cache_data=False` was previously set in `fit()`.

        NOTE: Until catboost v0.24 is released, `distill()` with CatBoost students in multiclass classification requires you to first install catboost-dev: `pip install catboost-dev`

        Parameters
        ----------
        train_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`, default = None
            Same as `train_data` argument of `fit()`.
            If None, the same training data will be loaded from `fit()` call used to produce this Predictor.
        tuning_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`, default = None
            Same as `tuning_data` argument of `fit()`.
            If `tuning_data = None` and `train_data = None`: the same training/validation splits will be loaded from `fit()` call used to produce this Predictor,
            unless bagging/stacking was previously used in which case a new training/validation split is performed.
        augmentation_data : :class:`TabularDataset` or :class:`pd.DataFrame`, default = None
            An optional extra dataset of unlabeled rows that can be used for augmenting the dataset used to fit student models during distillation (ignored if None).
        time_limit : int, default = None
            Approximately how long (in seconds) the distillation process should run for.
            If None, no time-constraint will be enforced allowing the distilled models to fully train.
        hyperparameters : dict or str, default = None
            Specifies which models to use as students and what hyperparameter-values to use for them.
            Same as `hyperparameters` argument of `fit()`.
            If = None, then student models will use the same hyperparameters from `fit()` used to produce this Predictor.
            Note: distillation is currently only supported for ['GBM','NN_MXNET','NN_TORCH','RF','CAT'] student models, other models and their hyperparameters are ignored here.
        holdout_frac : float
            Same as `holdout_frac` argument of :meth:`TabularPredictor.fit`.
        teacher_preds : str, default = 'soft'
            What form of teacher predictions to distill from (teacher refers to the most accurate AutoGluon ensemble-predictor).
            If None, we only train with original labels (no data augmentation).
            If 'hard', labels are hard teacher predictions given by: `teacher.predict()`
            If 'soft', labels are soft teacher predictions given by: `teacher.predict_proba()`
            Note: 'hard' and 'soft' are equivalent for regression problems.
            If `augment_method` is not None, teacher predictions are only used to label augmented data (training data keeps original labels).
            To apply label-smoothing: `teacher_preds='onehot'` will use original training data labels converted to one-hot vectors for multiclass problems (no data augmentation).
        augment_method : str, default='spunge'
            Specifies method to use for generating augmented data for distilling student models.
            Options include:
                None : no data augmentation performed.
                'munge' : The MUNGE algorithm (https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf).
                'spunge' : A simpler, more efficient variant of the MUNGE algorithm.
        augment_args : dict, default = {'size_factor':5, 'max_size': int(1e5)}
            Contains the following kwargs that control the chosen `augment_method` (these are ignored if `augment_method=None`):
                'num_augmented_samples': int, number of augmented datapoints used during distillation. Overrides 'size_factor', 'max_size' if specified.
                'max_size': float, the maximum number of augmented datapoints to add (ignored if 'num_augmented_samples' specified).
                'size_factor': float, if n = training data sample-size, we add int(n * size_factor) augmented datapoints, up to 'max_size'.
                Larger values in `augment_args` will slow down the runtime of distill(), and may produce worse results if provided time_limit are too small.
                You can also pass in kwargs for the `spunge_augment`, `munge_augment` functions in `autogluon.tabular.augmentation.distill_utils`.
        models_name_suffix : str, default = None
            Optional suffix that can be appended at the end of all distilled student models' names.
            Note: all distilled models will contain '_DSTL' substring in their name by default.
        verbosity : int, default = None
            Controls amount of printed output during distillation (4 = highest, 0 = lowest).
            Same as `verbosity` parameter of :class:`TabularPredictor`.
            If None, the same `verbosity` used in previous fit is employed again.

        Returns
        -------
        List of names (str) corresponding to the distilled models.

        Examples
        --------
        >>> from autogluon.tabular import TabularDataset, TabularPredictor
        >>> train_data = TabularDataset('train.csv')
        >>> predictor = TabularPredictor(label='class').fit(train_data, auto_stack=True)
        >>> distilled_model_names = predictor.distill()
        >>> test_data = TabularDataset('test.csv')
        >>> ldr = predictor.leaderboard(test_data)
        >>> model_to_deploy = distilled_model_names[0]
        >>> predictor.predict(test_data, model=model_to_deploy)

        """
        self._assert_is_fit('distill')
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)
        return self._learner.distill(X=train_data, X_val=tuning_data, time_limit=time_limit,
                                     hyperparameters=hyperparameters, holdout_frac=holdout_frac,
                                     verbosity=verbosity, models_name_suffix=models_name_suffix,
                                     teacher_preds=teacher_preds,
                                     augmentation_data=augmentation_data, augment_method=augment_method,
                                     augment_args=augment_args)

    def plot_ensemble_model(self, prune_unused_nodes=True) -> str:
        """
        Output the visualized stack ensemble architecture of a model trained by `fit()`.
        The plot is stored to a file, `ensemble_model.png` in folder `predictor.path`

        This function requires `graphviz` and `pygraphviz` to be installed because this visualization depends on those package.
        Unless this function will raise `ImportError` without being able to generate the visual of the ensemble model.

        To install the required package, run the below commands (for Ubuntu linux):

        $ sudo apt-get install graphviz
        $ pip install graphviz

        For other platforms, refer to https://graphviz.org/ for Graphviz install, and https://pygraphviz.github.io/documentation.html for PyGraphviz.


        Parameters
        ----------

        Returns
        -------
        The file name with the full path to the saved graphic
        """
        self._assert_is_fit('plot_ensemble_model')
        try:
            import pygraphviz
        except:
            raise ImportError('Visualizing ensemble network architecture requires pygraphviz library')

        G = self._trainer.model_graph.copy()

        if prune_unused_nodes == True:
            nodes_without_outedge = [node for node, degree in dict(G.degree()).items() if degree < 1]
        else:
            nodes_without_outedge = []

        nodes_no_val_score = [node for node in G if G.nodes[node]['val_score'] == None]

        G.remove_nodes_from(nodes_without_outedge)
        G.remove_nodes_from(nodes_no_val_score)

        root_node = [n for n, d in G.out_degree() if d == 0]
        best_model_node = self.get_model_best()

        A = nx.nx_agraph.to_agraph(G)

        A.graph_attr.update(rankdir='BT')
        A.node_attr.update(fontsize=10)
        A.node_attr.update(shape='rectangle')

        for node in A.iternodes():
            node.attr['label'] = f"{node.name}\nVal score: {float(node.attr['val_score']):.4f}"

            if node.name == best_model_node:
                node.attr['style'] = 'filled'
                node.attr['fillcolor'] = '#ff9900'
                node.attr['shape'] = 'box3d'
            elif nx.has_path(G, node.name, best_model_node):
                node.attr['style'] = 'filled'
                node.attr['fillcolor'] = '#ffcc00'

        model_image_fname = os.path.join(self.path, 'ensemble_model.png')

        A.draw(model_image_fname, format='png', prog='dot')

        return model_image_fname

    @staticmethod
    def _summarize(key, msg, results):
        if key in results:
            print(msg + ": " + str(results[key]))

    @staticmethod
    def __get_dataset(data):
        if isinstance(data, TabularDataset):
            return data
        elif isinstance(data, pd.DataFrame):
            return TabularDataset(data)
        elif isinstance(data, str):
            return TabularDataset(data)
        elif isinstance(data, pd.Series):
            raise TypeError("data must be TabularDataset or pandas.DataFrame, not pandas.Series. \
                   To predict on just single example (ith row of table), use data.iloc[[i]] rather than data.iloc[i]")
        else:
            raise TypeError("data must be TabularDataset or pandas.DataFrame or str file path to data")

    def _validate_hyperparameter_tune_kwargs(self, hyperparameter_tune_kwargs, time_limit=None):
        """
        Returns True if hyperparameter_tune_kwargs is None or can construct a valid scheduler.
        Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return True

        scheduler_cls, scheduler_params = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                                                            time_out=time_limit,
                                                            nthreads_per_trial='auto', ngpus_per_trial='auto')

        if scheduler_params.get('dist_ip_addrs', None):
            logger.warning(
                'Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.')

        if scheduler_params['num_trials'] == 1:
            logger.warning(
                'Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. This can occur if time_limit was not specified in `fit()`.')
            return False

        scheduler_ngpus = scheduler_params['resource'].get('num_gpus', 0)
        if scheduler_ngpus is not None and isinstance(scheduler_ngpus, int) and scheduler_ngpus > 1:
            logger.warning(
                f"Warning: TabularPredictor currently doesn't use >1 GPU per training run. Detected {scheduler_ngpus} GPUs.")

        return True

    def _set_hyperparameter_tune_kwargs_in_ag_args(self, hyperparameter_tune_kwargs, ag_args, time_limit):
        if hyperparameter_tune_kwargs is not None and 'hyperparameter_tune_kwargs' not in ag_args:
            if 'hyperparameter_tune_kwargs' in ag_args:
                AssertionError(
                    'hyperparameter_tune_kwargs was specified in both ag_args and in kwargs. Please only specify once.')
            else:
                ag_args['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
        if not self._validate_hyperparameter_tune_kwargs(ag_args.get('hyperparameter_tune_kwargs', None), time_limit):
            ag_args.pop('hyperparameter_tune_kwargs', None)
        if ag_args.get('hyperparameter_tune_kwargs', None) is not None:
            logger.log(30,
                       'Warning: hyperparameter tuning is currently experimental and may cause the process to hang.')
        return ag_args

    def _set_post_fit_vars(self, learner: AbstractTabularLearner = None):
        if learner is not None:
            self._learner: AbstractTabularLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._learner.persist_trainer(low_memory=True)
            self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object

    @classmethod
    def _load_version_file(cls, path) -> str:
        version_file_path = path + cls._predictor_version_file_name
        version = load_str.load(path=version_file_path)
        return version

    def _save_version_file(self, silent=False):
        from ..version import __version__
        version_file_contents = f'{__version__}'
        version_file_path = self.path + self._predictor_version_file_name
        save_str.save(path=version_file_path, data=version_file_contents, verbose=not silent)

    def save(self, silent=False):
        """
        Save this Predictor to file in directory specified by this Predictor's `path`.
        Note that :meth:`TabularPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).

        Parameters
        ----------
        silent : bool, default = False
            Whether to save without logging a message.
        """
        path = self.path
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        self._learner.save()
        self._learner = None
        self._trainer = None
        save_pkl.save(path=path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer
        self._save_version_file(silent=silent)
        if not silent:
            logger.log(20, f'TabularPredictor saved. To load, use: predictor = TabularPredictor.load("{self.path}")')

    @classmethod
    def _load(cls, path: str):
        """
        Inner load method, called in `load`.
        """
        predictor: TabularPredictor = load_pkl.load(path=path + cls.predictor_file_name)
        learner = predictor._learner_type.load(path)
        predictor._set_post_fit_vars(learner=learner)
        return predictor

    @classmethod
    def load(cls, path: str, verbosity: int = None, require_version_match: bool = True):
        """
        Load a TabularPredictor object previously produced by `fit()` from file and returns this object. It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        Parameters
        ----------
        path : str
            The path to directory in which this Predictor was previously saved.
        verbosity : int, default = None
            Sets the verbosity level of this Predictor after it is loaded.
            Valid values range from 0 (least verbose) to 4 (most verbose).
            If None, logging verbosity is not changed from existing values.
            Specify larger values to see more information printed when using Predictor during inference, smaller values to see less information.
            Refer to TabularPredictor init for more information.
        require_version_match : bool, default = True
            If True, will raise an AssertionError if the `autogluon.tabular` version of the loaded predictor does not match the installed version of `autogluon.tabular`.
            If False, will allow loading of models trained on incompatible versions, but is NOT recommended. Users may run into numerous issues if attempting this.
        """
        if verbosity is not None:
            set_logger_verbosity(verbosity)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        try:
            from ..version import __version__
            version_load = __version__
        except:
            version_load = None

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        try:
            version_init = cls._load_version_file(path=path)
        except:
            logger.warning(f'WARNING: Could not find version file at "{path + cls._predictor_version_file_name}".\n'
                           f'This means that the predictor was fit in a version `<=0.3.1`.')
            version_init = None

        if version_init is None:
            predictor = cls._load(path=path)
            try:
                version_init = predictor._learner.version
            except:
                version_init = None
        else:
            predictor = None
        if version_init is None:
            version_init = 'Unknown (Likely <=0.0.11)'
        if version_load != version_init:
            logger.warning('')
            logger.warning('############################## WARNING ##############################')
            logger.warning('WARNING: AutoGluon version differs from the version used to create the predictor! '
                           'This may lead to instability and it is highly recommended the predictor be loaded '
                           'with the exact AutoGluon version it was created with.')
            logger.warning(f'\tPredictor Version: {version_init}')
            logger.warning(f'\tCurrent Version:   {version_load}')
            logger.warning('############################## WARNING ##############################')
            logger.warning('')
            if require_version_match:
                raise AssertionError(
                    f'Predictor was created on version {version_init} but is being loaded with version {version_load}. '
                    f'Please ensure the versions match to avoid instability. While it is NOT recommended, '
                    f'this error can be bypassed by specifying `require_version_match=False`.')

        if predictor is None:
            predictor = cls._load(path=path)

        return predictor

    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'learner_type',
            'learner_kwargs',
            'quantile_levels',
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
            num_bag_folds=None,
            # TODO: Potentially move to fit_extra, raise exception if value too large / invalid in fit_extra.
            auto_stack=False,
            use_bag_holdout=False,

            # other
            feature_generator='auto',
            unlabeled_data=None,
            _feature_generator_kwargs=None,
        )

        kwargs = self._validate_fit_extra_kwargs(kwargs, extra_valid_keys=list(fit_kwargs_default.keys()))

        kwargs_sanitized = fit_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        return kwargs_sanitized

    def _fit_extra_kwargs_dict(self):
        """
        Returns:
        --------
        dict of fit_extra args:
            verbosity: Which levels of logger should be printed
            pseudo_data: pseudo labeled data to be incorporated into train
                         but not used in validation
            name_suffix: A suffix string to be added to the individual model names
        """
        return dict(
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

            # other
            verbosity=self.verbosity,
            feature_prune_kwargs=None,

            # private
            _save_bag_folds=None,

            # quantile levels
            quantile_levels=None,

            calibrate='auto',

            # pseudo label
            pseudo_data=None,

            name_suffix=None
        )

    def _validate_fit_extra_kwargs(self, kwargs, extra_valid_keys=None):
        fit_extra_kwargs_default = self._fit_extra_kwargs_dict()

        allowed_kwarg_names = list(fit_extra_kwargs_default.keys())
        if extra_valid_keys is not None:
            allowed_kwarg_names += extra_valid_keys
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                public_kwarg_options = [kwarg for kwarg in allowed_kwarg_names if kwarg[0] != '_']
                public_kwarg_options.sort()
                raise ValueError(
                    f"Unknown keyword argument specified: {kwarg_name}\nValid kwargs: {public_kwarg_options}")

        kwargs_sanitized = fit_extra_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        # Deepcopy args to avoid altering outer context
        deepcopy_args = ['ag_args', 'ag_args_fit', 'ag_args_ensemble', 'excluded_model_types']
        for deepcopy_arg in deepcopy_args:
            kwargs_sanitized[deepcopy_arg] = copy.deepcopy(kwargs_sanitized[deepcopy_arg])

        refit_full = kwargs_sanitized['refit_full']
        set_best_to_refit_full = kwargs_sanitized['set_best_to_refit_full']
        if refit_full and not self._learner.cache_data:
            raise ValueError(
                '`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')
        if set_best_to_refit_full and not refit_full:
            raise ValueError(
                '`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')
        valid_calibrate_options = [True, False, 'auto']
        calibrate = kwargs_sanitized['calibrate']
        if calibrate not in valid_calibrate_options:
            raise ValueError(f"`calibrate` must be a value in {valid_calibrate_options}, but is: {calibrate}")

        return kwargs_sanitized

    def _prune_data_features(self, train_features: pd.DataFrame, other_features: pd.DataFrame, is_labeled: bool):
        """
        Removes certain columns from the provided datasets that do not contain predictive features.

        Parameters
        ----------
        train_features : pd.DataFrame
            The features/columns for the incoming training data
        other_features : pd.DataFrame
            Features of other auxiliary data that contains the same covariates as the training data.
            Examples of this could be: tuning data, pseudo data
        is_labeled: bool
            Is other_features dataframe labeled or not
        """
        if self.sample_weight is not None:
            if self.sample_weight in train_features:
                train_features.remove(self.sample_weight)
            if self.sample_weight in other_features:
                other_features.remove(self.sample_weight)
        if self._learner.groups is not None and is_labeled:
            train_features.remove(self._learner.groups)

        return train_features, other_features

    def _validate_fit_data(self, train_data, tuning_data=None, unlabeled_data=None):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        if unlabeled_data is not None and isinstance(unlabeled_data, str):
            unlabeled_data = TabularDataset(unlabeled_data)

        if not isinstance(train_data, pd.DataFrame):
            raise AssertionError(
                f'train_data is required to be a pandas DataFrame, but was instead: {type(train_data)}')

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None:
            if not isinstance(tuning_data, pd.DataFrame):
                raise AssertionError(
                    f'tuning_data is required to be a pandas DataFrame, but was instead: {type(tuning_data)}')
            self._validate_unique_indices(data=tuning_data, name='tuning_data')
            train_features = [column for column in train_data.columns if column != self.label]
            tuning_features = [column for column in tuning_data.columns if column != self.label]
            train_features, tuning_features = self._prune_data_features(train_features=train_features,
                                                                        other_features=tuning_features,
                                                                        is_labeled=True)
            train_features = np.array(train_features)
            tuning_features = np.array(tuning_features)
            if np.any(train_features != tuning_features):
                raise ValueError("Column names must match between training and tuning data")
        if unlabeled_data is not None:
            if not isinstance(unlabeled_data, pd.DataFrame):
                raise AssertionError(
                    f'unlabeled_data is required to be a pandas DataFrame, but was instead: {type(unlabeled_data)}')
            self._validate_unique_indices(data=unlabeled_data, name='unlabeled_data')
            train_features = [column for column in train_data.columns if column != self.label]
            unlabeled_features = [column for column in unlabeled_data.columns]
            train_features, unlabeled_features = self._prune_data_features(train_features=train_features,
                                                                           other_features=unlabeled_features,
                                                                           is_labeled=False)
            train_features = sorted(np.array(train_features))
            unlabeled_features = sorted(np.array(unlabeled_features))
            if np.any(train_features != unlabeled_features):
                raise ValueError("Column names must match between training and unlabeled data.\n"
                                 "Unlabeled data must have not the label column specified in it.\n")
        return train_data, tuning_data, unlabeled_data

    @staticmethod
    def _validate_unique_indices(data, name: str):
        is_duplicate_index = data.index.duplicated(keep=False)
        if is_duplicate_index.any():
            duplicate_count = is_duplicate_index.sum()
            raise AssertionError(f'{name} contains {duplicate_count} duplicated indices. '
                                 'Please ensure DataFrame indices are unique.\n'
                                 f'\tYou can identify the indices which are duplicated via `{name}.index.duplicated(keep=False)`')

    @staticmethod
    def _validate_infer_limit(infer_limit: float, infer_limit_batch_size: int) -> (float, int):
        if infer_limit_batch_size is not None:
            if not isinstance(infer_limit_batch_size, int):
                raise ValueError(f'infer_limit_batch_size must be type int, but was instead type {type(infer_limit_batch_size)}')
            elif infer_limit_batch_size < 1:
                raise AssertionError(f'infer_limit_batch_size must be >=1, value: {infer_limit_batch_size}')
        if infer_limit is not None:
            if not isinstance(infer_limit, (int, float)):
                raise ValueError(f'infer_limit must be type int or float, but was instead type {type(infer_limit)}')
            if infer_limit <= 0:
                raise AssertionError(f'infer_limit must be greater than zero! (infer_limit={infer_limit})')
        if infer_limit is not None and infer_limit_batch_size is None:
            infer_limit_batch_size = 10000
            logger.log(20, f'infer_limit specified, but infer_limit_batch_size was not specified. Setting infer_limit_batch_size={infer_limit_batch_size}')
        return infer_limit, infer_limit_batch_size

    def _set_feature_generator(self, feature_generator='auto', feature_metadata=None, init_kwargs=None):
        if self._learner.feature_generator is not None:
            if isinstance(feature_generator, str) and feature_generator == 'auto':
                feature_generator = self._learner.feature_generator
            else:
                raise AssertionError('FeatureGenerator already exists!')
        self._learner.feature_generator = get_default_feature_generator(feature_generator=feature_generator,
                                                                        feature_metadata=feature_metadata,
                                                                        init_kwargs=init_kwargs)

    def _sanitize_stack_args(self, num_bag_folds, num_bag_sets, num_stack_levels, time_limit, auto_stack,
                             num_train_rows):
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limit? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            if num_bag_folds is None:
                num_bag_folds = min(8, max(5, math.floor(num_train_rows / 100)))
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
            raise ValueError(
                f'num_stack_levels must be 0 if num_bag_folds is 0. (num_stack_levels={num_stack_levels}, num_bag_folds={num_bag_folds})')
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

    def interpretable_models_summary(self, verbosity=0):
        '''Summary of fitted interpretable models along with their corresponding complexities
        '''
        d = self.fit_summary(verbosity=verbosity)
        summaries = pd.DataFrame.from_dict(d)

        complexities = []
        info = self.info()
        for i in range(summaries.shape[0]):
            model_name = summaries.index.values[i]
            complexities.append(info['model_info'][model_name].get('complexity', np.nan))
        summaries.insert(2, 'complexity', complexities)
        summaries = summaries[~pd.isna(summaries.complexity)]  # remove non-interpretable models
        return summaries.sort_values(by=['model_performance', 'complexity'], ascending=[False, True])

    def print_interpretable_rules(self, complexity_threshold: int = 10, model_name: str = None):
        """
        Print the rules of the highest performing model below the complexity threshold.

        Parameters
        ----------
        complexity_threshold : int, default=10
            Threshold for complexity (number of rules) of fitted models to show.
            If not model complexity is below this threshold, prints the model with the lowest complexity.
        model_name : str,  default=None
            Optionally print rules for a particular model, ignoring the complexity threshold.
        """
        if model_name is None:
            summaries = self.interpretable_models_summary()
            summaries_filtered = summaries[summaries.complexity <= complexity_threshold]
            if summaries_filtered.shape[0] == 0:
                summaries_filtered = summaries
            model_name = summaries_filtered.index.values[0]  # best model is at top
        agmodel = self._trainer.load_model(model_name)
        imodel = agmodel.model
        print(imodel)

    def explain_classification_errors(self, data, model = None, print_rules: bool = True):
        """Explain classification errors by fitting a rule-based model to them

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training Dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        print_rules : bool, optional
            Whether to print the learned rules

        Returns
        -------
        cls : imodels.classifier
            Interpretable rule-based classifier with fit/predict methods
        """
        import imodels
        data = self.__get_dataset(data)
        predictions = self._learner.predict(X=data, model=model, as_pandas=True)
        labels = data[self.label]
        cls, columns = imodels.explain_classification_errors(data, predictions, labels, print_rules=print_rules)
        return cls

    def _assert_is_fit(self, message_suffix: str = None):
        if not self._learner.is_fit:
            error_message = "Predictor is not fit. Call `.fit` before calling"
            if message_suffix is None:
                error_message = f"{error_message} this method."
            else:
                error_message = f"{error_message} `.{message_suffix}`."
            raise AssertionError(error_message)

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
            advice_list.append(
                'FeatureGenerator has not been fit, consider calling `predictor.fit_feature_generator(data)`.')
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

    @classmethod
    def from_learner(cls, learner: AbstractTabularLearner):
        predictor = cls(label=learner.label, path=learner.path)
        predictor._set_post_fit_vars(learner=learner)
        return predictor
