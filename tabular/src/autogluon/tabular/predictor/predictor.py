from __future__ import annotations

import copy
import inspect
import logging
import math
import os
import pprint
import shutil
import time
import warnings
from typing import overload, Any, Literal, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from packaging import version

from autogluon.common import FeatureMetadata, TabularDataset
from autogluon.common.loaders import load_json
from autogluon.common.savers import save_json
from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.common.utils.decorators import apply_presets
from autogluon.common.utils.file_utils import get_directory_size, get_directory_size_per_file
from autogluon.common.utils.resource_utils import ResourceManager, get_resource_manager
from autogluon.common.utils.hyperparameter_utils import get_hyperparameter_str_deprecation_msg, is_advanced_hyperparameter_format
from autogluon.common.utils.log_utils import add_log_to_file, set_logger_verbosity, warn_if_mlflow_autologging_is_enabled
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.system_info import get_ag_system_info
from autogluon.common.utils.try_import import try_import_ray
from autogluon.common.utils.utils import check_saved_predictor_version, compare_autogluon_metadata, get_autogluon_metadata, setup_outputdir
from autogluon.core.callbacks import AbstractCallback
from autogluon.core.constants import (
    AUTO_WEIGHT,
    BALANCE_WEIGHT,
    BINARY,
    MULTICLASS,
    PROBLEM_TYPES_CLASSIFICATION,
    PSEUDO_MODEL_SUFFIX,
    QUANTILE,
    REGRESSION,
    SOFTCLASS,
)
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.metrics import Scorer, get_metric
from autogluon.core.problem_type import problem_type_info
from autogluon.core.pseudolabeling.pseudolabeling import filter_ensemble_pseudo, filter_pseudo
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.stacked_overfitting.utils import check_stacked_overfitting_from_leaderboard
from autogluon.core.utils import get_pred_from_proba_df, plot_performance_vs_trials, plot_summary_of_models, plot_tabular_models
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.core.utils.utils import generate_train_test_split_combined

from ..configs.feature_generator_presets import get_default_feature_generator
from ..configs.hyperparameter_configs import get_hyperparameter_config
from ..configs.pipeline_presets import (
    USE_BAG_HOLDOUT_AUTO_THRESHOLD,
    get_validation_and_stacking_method,
)
from ..configs.presets_configs import tabular_presets_alias, tabular_presets_dict
from ..learner import AbstractTabularLearner, DefaultLearner
from ..trainer.abstract_trainer import AbstractTabularTrainer
from ..registry import ag_model_registry
from ..version import __version__

logger = logging.getLogger(__name__)  # return autogluon root logger


# Extra TODOs (Stretch): Can occur post v1.0
# TODO: make core_kwargs a kwargs argument to predictor.fit
# TODO: add aux_kwargs to predictor.fit
# TODO: consider adding kwarg option for data which has already been preprocessed by feature generator to skip feature generation.
# TODO: Resolve raw text feature usage in default feature generator
# TODO: num_bag_sets -> ag_args
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
    eval_metric : str or Scorer, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression, and 'pinball_loss' for quantile.

        Otherwise, options for classification:
            ['accuracy', 'balanced_accuracy', 'log_loss', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_macro', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_macro', 'roc_auc_ovr_micro',
            'roc_auc_ovr_weighted', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'mcc', 'pac_score']
        Options for regression:
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_absolute_percentage_error', 'r2', 'symmetric_mean_absolute_percentage_error']
        For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
        For metric source code, see `autogluon.core.metrics`.

        You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
        For detailed instructions on creating and using a custom metric, refer to https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
    path : Union[str, pathlib.Path], default = None
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
        Verbosity levels:
            0: Only log exceptions
            1: Only log warnings + exceptions
            2: Standard logging
            3: Verbose logging (ex: log validation score every 50 iterations)
            4: Maximally verbose logging (ex: log validation score every iteration)
    log_to_file: bool, default = False
        Whether to save the logs into a file for later reference
    log_file_path: str, default = "auto"
        File path to save the logs.
        If auto, logs will be saved under `predictor_path/logs/predictor_log.txt`.
        Will be ignored if `log_to_file` is set to False
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
    positive_class : str or int, default = None
        Used to determine the positive class in binary classification.
        This is used for certain metrics such as 'f1' which produce different scores depending on which class is considered the positive class.
        If not set, will be inferred as the second element of the existing unique classes after sorting them.
            If classes are [0, 1], then 1 will be selected as the positive class.
            If classes are ['def', 'abc'], then 'def' will be selected as the positive class.
            If classes are [True, False], then True will be selected as the positive class.
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
            trainer_type : AbstractTabularTrainer, default = AutoTrainer
                A class inheriting from `AbstractTabularTrainer` that controls training/ensembling of many models.
                If you don't know what this is, keep it as the default.
        default_base_path : str | Path | None, default = None
            A default base path to use for the time-stamped folder if `path` is None.
            If None, defaults to `AutogluonModels`. Only used if `path` is None, and thus
            only used for local paths, not s3 paths.
    """

    Dataset = TabularDataset
    predictor_file_name = "predictor.pkl"
    _predictor_version_file_name = "version.txt"
    _predictor_metadata_file_name = "metadata.json"
    _predictor_log_file_name = "predictor_log.txt"

    def __init__(
        self,
        label: str,
        problem_type: str = None,
        eval_metric: str | Scorer = None,
        path: str = None,
        verbosity: int = 2,
        log_to_file: bool = False,
        log_file_path: str = "auto",
        sample_weight: str = None,
        weight_evaluation: bool = False,
        groups: str = None,
        positive_class: int | str | None = None,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity)
        if sample_weight == AUTO_WEIGHT:  # TODO: update auto_weight strategy and make it the default
            sample_weight = None
            logger.log(15, f"{AUTO_WEIGHT} currently does not use any sample weights.")
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation  # TODO: sample_weight and weight_evaluation can both be properties that link to self._learner.sample_weight, self._learner.weight_evaluation
        self._decision_threshold = None  # TODO: Each model should have its own decision threshold instead of one global threshold
        if self.sample_weight in [AUTO_WEIGHT, BALANCE_WEIGHT] and self.weight_evaluation:
            logger.warning(
                f"We do not recommend specifying weight_evaluation when sample_weight='{self.sample_weight}', instead specify appropriate eval_metric."
            )
        self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path=path, default_base_path=kwargs.get("default_base_path"))

        learner_type = kwargs.get("learner_type", DefaultLearner)
        learner_kwargs = kwargs.get("learner_kwargs", dict())
        quantile_levels = kwargs.get("quantile_levels", None)
        if positive_class is not None:
            learner_kwargs["positive_class"] = positive_class

        self._learner: AbstractTabularLearner = learner_type(
            path_context=path,
            label=label,
            feature_generator=None,
            eval_metric=eval_metric,
            problem_type=problem_type,
            quantile_levels=quantile_levels,
            sample_weight=self.sample_weight,
            weight_evaluation=self.weight_evaluation,
            groups=groups,
            **learner_kwargs,
        )
        self._learner_type = type(self._learner)
        self._trainer: AbstractTabularTrainer = None
        self._sub_fits: list[str] = []
        self._stacked_overfitting_occurred: bool | None = None
        self._fit_strategy = None

        if log_to_file:
            self._setup_log_to_file(log_file_path=log_file_path)

    @property
    def classes_(self) -> list:
        """
        For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output.
        For binary problems, this list contains the class labels in sorted order of `predict_proba(as_multiclass=True)` output.
            `classes_[0]` corresponds to internal label = 0 (negative class), `classes_[1]` corresponds to internal label = 1 (positive class).
            This is relevant for certain metrics such as F1 where True and False labels impact the metric score differently.
        For other problem types, will equal None.
        For example if `pred = predict_proba(x, as_multiclass=True)`, then ith index of `pred` provides predicted probability that `x` belongs to class given by `classes_[i]`.
        """
        return self._learner.class_labels

    @property
    def class_labels(self) -> list:
        """Alias to self.classes_"""
        return self.classes_

    @property
    def class_labels_internal(self) -> list:
        """
        For multiclass problems, this list contains the internal class labels in sorted order of internal `predict_proba()` output.
        For binary problems, this list contains the internal class labels in sorted order of internal `predict_proba(as_multiclass=True)` output.
            The value will always be `class_labels_internal=[0, 1]` for binary problems, with 0 as the negative class, and 1 as the positive class.
        For other problem types, will equal None.
        """
        return self._learner.label_cleaner.ordered_class_labels_transformed

    @property
    def class_labels_internal_map(self) -> dict:
        """
        For binary and multiclass classification problems, this dictionary contains the mapping of the original labels to the internal labels.
        For example, in binary classification, label values of 'True' and 'False' will be mapped to the internal representation `1` and `0`.
            Therefore, class_labels_internal_map would equal {'True': 1, 'False': 0}
        For other problem types, will equal None.
        For multiclass, it is possible for not all of the label values to have a mapping.
            This indicates that the internal models will never predict those missing labels, and training rows associated with the missing labels were dropped.
        """
        return self._learner.label_cleaner.inv_map

    @property
    def quantile_levels(self) -> list[float]:
        return self._learner.quantile_levels

    @property
    def eval_metric(self) -> Scorer:
        """The metric used to evaluate predictive performance"""
        return self._learner.eval_metric

    @property
    def original_features(self) -> list[str]:
        """Original features user passed in to fit before processing"""
        self._assert_is_fit()
        return self._learner.original_features

    @property
    def problem_type(self) -> str:
        """What type of prediction problem this Predictor has been trained for"""
        return self._learner.problem_type

    @property
    def decision_threshold(self) -> float | None:
        """
        The decision threshold used to convert prediction probabilities to predictions.
        Only relevant for binary classification, otherwise the value will be None.
        Valid values are in the range [0.0, 1.0]
        You can obtain an optimized `decision_threshold` by first calling `predictor.calibrate_decision_threshold()`.
        Useful to set for metrics such as `balanced_accuracy` and `f1` as `0.5` is often not an optimal threshold.
        Predictions are calculated via the following logic on the positive class: `1 if pred > decision_threshold else 0`
        """
        if self._decision_threshold is not None:
            return self._decision_threshold
        elif self.problem_type == BINARY:
            return 0.5
        else:
            return None

    def set_decision_threshold(self, decision_threshold: float):
        """
        Set `predictor.decision_threshold`. Problem type must be 'binary', and the value must be between 0 and 1.
        """
        assert self.problem_type == BINARY
        assert decision_threshold >= 0
        assert decision_threshold <= 1
        if decision_threshold != self.decision_threshold:
            logger.log(
                20,
                f"Updating predictor.decision_threshold from {self.decision_threshold} -> {decision_threshold}\n"
                f"\tThis will impact how prediction probabilities are converted to predictions in binary classification.\n"
                f"\tPrediction probabilities of the positive class >{decision_threshold} "
                f"will be predicted as the positive class ({self.positive_class}). "
                f"This can significantly impact metric scores.\n"
                f"\tYou can update this value via `predictor.set_decision_threshold`.\n"
                f"\tYou can calculate an optimal decision threshold on the validation data via `predictor.calibrate_decision_threshold()`.",
            )
        self._decision_threshold = decision_threshold

    def features(self, feature_stage: str = "original") -> list:
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
        if feature_stage == "original":
            return self.feature_metadata_in.get_features()
        elif feature_stage == "transformed":
            return self.feature_metadata.get_features()
        else:
            raise ValueError(f"Unknown feature_stage: '{feature_stage}'. Must be one of {['original', 'transformed']}")

    @property
    def has_val(self) -> bool:
        """
        Return True if holdout validation data was used during fit, else return False.
        """
        self._assert_is_fit("has_val")
        return self._trainer.has_val

    @property
    def feature_metadata(self) -> FeatureMetadata:
        """
        Returns the internal FeatureMetadata.

        Inferred data type of each predictive variable after preprocessing transformation (i.e. column of training data table used to predict `label`).
        Contains both raw dtype and special dtype information. Each feature has exactly 1 raw dtype (such as 'int', 'float', 'category') and zero to many special dtypes (such as 'datetime_as_int', 'text', 'text_ngram').
        Special dtypes are AutoGluon specific feature types that are used to identify features with meaning beyond what the raw dtype can convey.
            `feature_metadata.type_map_raw`: Dictionary of feature name -> raw dtype mappings.
            `feature_metadata.type_group_map_special`: Dictionary of lists of special feature names, grouped by special feature dtype.
        """
        return self._trainer.feature_metadata

    @property
    def feature_metadata_in(self) -> FeatureMetadata:
        """
        Returns the input FeatureMetadata.

        Inferred data type of each predictive variable before preprocessing transformation.
        Contains both raw dtype and special dtype information. Each feature has exactly 1 raw dtype (such as 'int', 'float', 'category') and zero to many special dtypes (such as 'datetime_as_int', 'text', 'text_ngram').
        Special dtypes are AutoGluon specific feature types that are used to identify features with meaning beyond what the raw dtype can convey.
            `feature_metadata.type_map_raw`: Dictionary of feature name -> raw dtype mappings.
            `feature_metadata.type_group_map_special`: Dictionary of lists of special feature names, grouped by special feature dtype.
        """
        return self._learner.feature_generator.feature_metadata_in

    @property
    def label(self) -> str | int:
        """
        Name of table column that contains data from the variable to predict (often referred to as: labels, response variable, target variable, dependent variable, y, etc).
        """
        return self._learner.label

    @property
    def path(self) -> str:
        """Path to directory where all models used by this Predictor are stored"""
        return self._learner.path

    @apply_presets(tabular_presets_dict, tabular_presets_alias)
    def fit(
        self,
        train_data: pd.DataFrame | str,
        tuning_data: pd.DataFrame | str = None,
        time_limit: float = None,
        presets: list[str] | str = None,
        hyperparameters: dict | str = None,
        feature_metadata: str | FeatureMetadata = "infer",
        infer_limit: float = None,
        infer_limit_batch_size: int = None,
        fit_weighted_ensemble: bool = True,
        fit_full_last_level_weighted_ensemble: bool = True,
        full_weighted_ensemble_additionally: bool = False,
        dynamic_stacking: bool | str = False,
        calibrate_decision_threshold: bool | str = "auto",
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
        fit_strategy: Literal["sequential", "parallel"] = "sequential",
        memory_limit: float | str = "auto",
        callbacks: list[AbstractCallback | list | tuple] = None,
        **kwargs,
    ) -> "TabularPredictor":
        """
        Fit models to predict a column of a data table (label) based on the other columns (features).

        Parameters
        ----------
        train_data : :class:`pd.DataFrame` or str
            Table of the training data as a pandas DataFrame.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        tuning_data : :class:`pd.DataFrame` or str, optional
            Another dataset containing validation data reserved for tuning processes such as early stopping, hyperparameter tuning, and ensembling.
            This dataset should be in the same format as `train_data`.
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: If `refit_full=True` is specified, the final model may be fit on `tuning_data` as well as `train_data`.
            Note: Because `tuning_data` is used to determine which model is the 'best' model, as well as to determine the ensemble weights,
                it should not be considered a fully unseen dataset. It is possible that AutoGluon will be overfit to the `tuning_data`.
                To ensure an unbiased evaluation, use separate unseen test data to evaluate the final model using `predictor.leaderboard(test_data, display=True)`.
                Do not provide your evaluation test data as `tuning_data`!
            If bagging is not enabled and `tuning_data = None`: `fit()` will automatically hold out some random validation samples from `train_data`.
            If bagging is enabled  and `tuning_data = None`: no tuning data will be used. Instead, AutoGluon will perform cross-validation.
            If bagging is enabled: `use_bag_holdout=True` must be specified in order to provide tuning data. If specified, AutoGluon will still perform cross-validation for model fits, but will use `tuning_data` for optimizing the weighted ensemble weights and model calibration.
        time_limit : int, default = None
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until all models have completed training, but will not repeatedly bag models unless `num_bag_sets` is specified.
        presets : list or str or dict, default = ['medium_quality']
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'` (or `extreme_quality` if a GPU is available).
            To get good quality with minimal disk usage, set `presets=['good_quality', 'optimize_for_deployment']`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/tabular/configs/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.

            Available Presets: ['extreme_quality', 'best_quality', 'high_quality', 'good_quality', 'medium_quality', 'experimental_quality', 'optimize_for_deployment', 'interpretable', 'ignore_text']

            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            In-depth Preset Info:
                extreme_quality={...}
                    New in v1.5: The state-of-the-art for tabular machine learning.
                    Requires `pip install autogluon.tabular[tabarena]` to install TabPFN, TabICL, and TabDPT.
                    Significantly more accurate than `best_quality` on datasets <= 100000 samples. Requires a GPU.
                    Will use recent tabular foundation models TabPFNv2, TabICL, TabDPT, and Mitra to maximize performance.
                    Recommended for applications that benefit from the best possible model accuracy.

                best_quality_v150={...}
                    New in v1.5: Better quality than 'best_quality' and 5x+ faster to train. Give it a try!

                best_quality={'auto_stack': True, 'dynamic_stacking': 'auto', 'hyperparameters': 'zeroshot'}
                    Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
                    Recommended for applications that benefit from the best possible model accuracy.

                high_quality_v150={...}
                    New in v1.5: Better quality than 'high_quality' and 5x+ faster to train. Give it a try!

                high_quality={'auto_stack': True, 'dynamic_stacking': 'auto', 'hyperparameters': 'zeroshot', 'refit_full': True, 'set_best_to_refit_full': True, 'save_bag_folds': False}
                    High predictive accuracy with fast inference. ~8x faster inference and ~8x lower disk usage than `best_quality`.
                    Recommended for applications that require reasonable inference speed and/or model size.

                good_quality={'auto_stack': True, 'dynamic_stacking': 'auto', 'hyperparameters': 'light', 'refit_full': True, 'set_best_to_refit_full': True, 'save_bag_folds': False}
                    Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality`.
                    Recommended for applications that require fast inference speed.

                medium_quality={'auto_stack': False}
                    Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality`.
                    This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality` results in significantly better predictive accuracy and faster inference time.

                experimental_quality={'auto_stack': True, 'dynamic_stacking': 'auto', 'hyperparameters': 'experimental', 'fit_strategy': 'parallel', 'num_gpus': 0}
                    This preset acts as a testing ground for cutting edge features and models which could later be added to the `best_quality` preset in future releases.
                    Recommended when `best_quality` was already being used and the user wants to push performance even further.

                optimize_for_deployment={'keep_only_best': True, 'save_space': True}
                    Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
                    Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
                    This will disable numerous advanced functionality, but has no impact on inference.
                    This will make certain functionality less informative, such as `predictor.leaderboard()` and `predictor.fit_summary()`.
                        Because unused models will be deleted under this preset, methods like `predictor.leaderboard()` and `predictor.fit_summary()` will no longer show the full set of models that were trained during `fit()`.
                    Recommended for applications where the inner details of AutoGluon's training is not important and there is no intention of manually choosing between the final models.
                    This preset pairs well with the other presets such as `good_quality` to make a very compact final model.
                    Identical to calling `predictor.delete_models(models_to_keep='best')` and `predictor.save_space()` directly after `fit()`.

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
                Valid `str` options: ['default', 'zeroshot', 'zeroshot_2025_tabfm', 'light', 'very_light', 'toy', 'multimodal']
                    'default': Default AutoGluon hyperparameters intended to get strong accuracy with reasonable disk usage and inference time. Used in the 'medium_quality' preset.
                    'zeroshot': A powerful model portfolio learned from TabRepo's ensemble simulation on 200 datasets. Contains ~100 models and is used in 'best_quality' and 'high_quality' presets.
                    'zeroshot_2025_tabfm': Absolute cutting edge portfolio learned from TabArena's ensemble simulation that leverages tabular foundation models. Contains 22 models and is used in the `extreme_quality` preset.
                    'light': Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy. Used in the 'good_quality' preset.
                    'very_light': Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
                    'toy': Results in extremely small models. Only use this when prototyping, as the model quality will be severely reduced.
                    'multimodal': [EXPERIMENTAL] Trains a multimodal transformer model alongside tabular models. Requires that some text columns appear in the data and GPU.
                        When combined with 'best_quality' `presets` option, this can achieve extremely strong results in multimodal data tables that contain columns with text in addition to numeric/categorical columns.
                Reference `autogluon/tabular/configs/hyperparameter_configs.py` for information on the hyperparameters associated with each preset.
            Keys are strings that indicate which model types to train.
                Stable model options include:
                    'GBM' (LightGBM)
                    'CAT' (CatBoost)
                    'XGB' (XGBoost)
                    'EBM' (Explainable Boosting Machine)
                    'REALMLP' (RealMLP)
                    'TABM' (TabM)
                    'MITRA' (Mitra)
                    'TABICL' (TabICL)
                    'TABPFNV2' (TabPFNv2)
                    'RF' (random forest)
                    'XT' (extremely randomized trees)
                    'KNN' (k-nearest neighbors)
                    'LR' (linear regression)
                    'NN_TORCH' (neural network implemented in Pytorch)
                    'FASTAI' (neural network with FastAI backend)
                    'AG_AUTOMM' (`MultimodalPredictor` from `autogluon.multimodal`. Supports Tabular, Text, and Image modalities. GPU is required.)
                Experimental model options include:
                    'FT_TRANSFORMER' (Tabular Transformer, GPU is recommended. Does not scale well to >100 features. Recommended to use TabM instead.)
                    'FASTTEXT' (FastText. Note: Has not been tested for a long time.)
                    'AG_TEXT_NN' (Multimodal Text+Tabular model, GPU is required. Recommended to instead use its successor, 'AG_AUTOMM'.)
                    'AG_IMAGE_NN' (Image model, GPU is required. Recommended to instead use its successor, 'AG_AUTOMM'.)
                If a certain key is missing from hyperparameters, then `fit()` will not train any models of that type. Omitting a model key from hyperparameters is equivalent to including this model key in `excluded_model_types`.
                For example, set `hyperparameters = { 'NN_TORCH':{...} }` if say you only want to train (PyTorch) neural networks and no other types of models.
            Values = dict of hyperparameter settings for each model type, or list of dicts.
                Each hyperparameter can either be a single fixed value or a search space containing many possible values.
                Unspecified hyperparameters will be set to default values (or default search spaces if `hyperparameter_tune_kwargs='auto'`).
                Caution: Any provided search spaces will error if `hyperparameter_tune_kwargs=None` (Default).
                To train multiple models of a given type, set the value to a list of hyperparameter dictionaries.
                    For example, `hyperparameters = {'RF': [{'criterion': 'gini'}, {'criterion': 'entropy'}]}` will result in 2 random forest models being trained with separate hyperparameters.
            Advanced functionality: Bring your own model / Custom model support
                AutoGluon fully supports custom models. For a detailed tutorial on creating and using custom models with AutoGluon, refer to https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
            Advanced functionality: Custom stack levels
                By default, AutoGluon reuses the same models and model hyperparameters at each level during stack ensembling.
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
                        {
                            "learning_rate": 0.03,
                            "num_leaves": 128,
                            "feature_fraction": 0.9,
                            "min_data_in_leaf": 3,
                            "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
                        },
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
                     See also the FastAI docs: https://docs.fast.ai/tabular.learner.html
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
                        Individual arguments can be passed for ag_args_fit by adding the prefix `ag.`: `hyperparameters = {'RF': {..., 'ag.num_cpus': 1}}`
                        Individual arguments can be passed for ag_args_ensemble by adding the prefix `ag.ens`: `hyperparameters = {'RF': {..., 'ag.ens.fold_fitting_strategy': 'sequential_local'}}`
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
                            max_rows : (int, default=None)
                                If train_data has more rows than `max_rows`, the model will raise an AssertionError at the start of fit.
                            max_features : (int, default=None)
                                If train_data has more features than `max_features`, the model will raise an AssertionError at the start of fit.
                            max_classes : (int, default==None)
                                If train_data has more classes than `max_classes`, the model will raise an AssertionError at the start of fit.
                            problem_types : (list[str], default=None)
                                If the task is not a problem_type in `problem_types`, the model will raise an AssertionError at the start of fit.
                            ignore_constraints : (bool, default=False)
                                If True, will ignore the values of `max_rows`, `max_features`, `max_classes`, and `problem_type`, treating them as None.
                    ag_args_ensemble: Dictionary of hyperparameters shared by all models that control how they are ensembled, if bag mode is enabled.
                        Valid keys:
                            use_orig_features: [True, False, "never"], default True
                                Whether a stack model will use the original features along with the stack features to train (akin to skip-connections).
                                If True, will use the original data features.
                                If False, will discard the original data features and only use stack features, except when no stack features exist (such as in layer 1).
                                If "never", will always discard the original data features. Will be skipped in layer 1.
                            valid_stacker : bool, default True
                                If True, will be marked as valid to include as a stacker model.
                                If False, will only be fit as a base model (layer 1) and will not be fit in stack layers (layer 2+).
                            max_base_models : int, default 0
                                Maximum number of base models whose predictions form the features input to this stacker model.
                                If more than `max_base_models` base models are available, only the top `max_base_models` models with highest validation score are used.
                                If 0, the logic is skipped.
                            max_base_models_per_type : int | str, default "auto"
                                Similar to `max_base_models`. If more than `max_base_models_per_type` of any particular model type are available,
                                only the top `max_base_models_per_type` of that type are used. This occurs before the `max_base_models` filter.
                                If "auto", the value will be adaptively set based on the number of training samples.
                                    More samples will lead to larger values, starting at 1 with <1000 samples, increasing up to 12 at >=50000 samples.
                                If 0, the logic is skipped.
                            num_folds: (int, default=None) If specified, the number of folds to fit in the bagged model.
                                If specified, overrides any other value used to determine the number of folds such as predictor.fit `num_bag_folds` argument.
                            max_sets: (int, default=None) If specified, the maximum sets to fit in the bagged model.
                                The lesser of `max_sets` and the predictor.fit `num_bag_sets` argument will be used for the given model.
                                Useful if a particular model is expensive relative to others and you want to avoid repeated bagging of the expensive model while still repeated bagging the cheaper models.
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

        feature_metadata : :class:`autogluon.common.FeatureMetadata` or str, default = 'infer'
            The feature metadata used in various inner logic in feature preprocessing.
            If 'infer', will automatically construct a FeatureMetadata object based on the properties of `train_data`.
            In this case, `train_data` is input into :meth:`autogluon.common.FeatureMetadata.from_df` to infer `feature_metadata`.
            If 'infer' incorrectly assumes the dtypes of features, consider explicitly specifying `feature_metadata`.
        infer_limit : float, default = None
            The inference time limit in seconds per row to adhere to during fit.
            If infer_limit=0.05 and infer_limit_batch_size=1000, AutoGluon will avoid training models that take longer than 50 ms/row to predict when given a batch of 1000 rows to predict (must predict 1000 rows in no more than 50 seconds).
            If bagging is enabled, the inference time limit will be respected based on estimated inference speed of `_FULL` models after refit_full is called, NOT on the inference speed of the bagged ensembles.
            The inference times calculated for models are assuming `predictor.persist('all')` is called after fit.
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
        fit_weighted_ensemble : bool, default = True
            If True, a WeightedEnsembleModel will be fit in each stack layer.
            A weighted ensemble will often be stronger than an individual model while being very fast to train.
            It is recommended to keep this value set to True to maximize predictive quality.
        fit_full_last_level_weighted_ensemble : bool, default = True
            If True, the WeightedEnsembleModel of the last stacking level will be fit with all (successful) models from all previous layers as base models.
            If stacking is disabled, settings this to True or False makes no difference because the WeightedEnsembleModel L2 always uses all models from L1.
            It is recommended to keep this value set to True to maximize predictive quality.
        full_weighted_ensemble_additionally : bool, default = False
            If True, AutoGluon will fit two WeightedEnsembleModels after training all stacking levels. Setting this to True, simulates calling
            `fit_weighted_ensemble()` after calling `fit()`. Has no affect if `fit_full_last_level_weighted_ensemble` is False and does not fit an additional
            WeightedEnsembleModel if stacking is disabled.
        dynamic_stacking: bool | str, default = False
            If True and `num_stack_levels` > 0, AutoGluon will dynamically determine whether to use stacking or not by first validating AutoGluon's stacking
            behavior. This is done to avoid so-called stacked overfitting that can make traditional multi-layer stacking, as used in AutoGluon, fail drastically
            and produce unreliable validation scores.
            It is recommended to keep this value set to True or "auto" when using stacking,
            as long as it is unknown whether the data is affected by stacked overfitting.
            If it is known that the data is unaffected by stacked overfitting, then setting this value to False is expected to maximize predictive quality.
            If enabled, by default, AutoGluon performs dynamic stacking by spending 25% of the provided time limit for detection and all remaining
            time for fitting AutoGluon. This can be adjusted by specifying `ds_args` with different parameters to `fit()`.
            If "auto", will be set to `not use_bag_holdout`.
            See the documentation of `ds_args` for more information.
        calibrate_decision_threshold : bool | str, default = "auto"
            If True, will automatically calibrate the decision threshold at the end of fit for calls to `.predict` based on the evaluation metric.
            If "auto", will be set to True if `eval_metric.needs_class=True` and `problem_type="binary"`.
            By default, the decision threshold is `0.5`, however for some metrics such as `f1` and `balanced_accuracy`,
            scores can be significantly improved by choosing a threshold other than `0.5`.
            Only valid for `problem_type='binary'`. Ignored for all other problem types.
        num_cpus: int | str, default = "auto"
            The total amount of cpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
            Users generally don't need to set this value
        num_gpus: int | str, default = "auto"
            The total amount of gpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
            Users generally don't need to set this value
        fit_strategy: Literal["sequential", "parallel"], default = "sequential"
            The strategy used to fit models.
            If "sequential", models will be fit sequentially. This is the most stable option with the most readable logging.
            If "parallel", models will be fit in parallel with ray, splitting available compute between them.
                Note: "parallel" is experimental and may run into issues. It was first added in version 1.2.0.
                Note: "parallel" does not yet support running with GPUs.
            For machines with 16 or more CPU cores, it is likely that "parallel" will be faster than "sequential".

            .. versionadded:: 1.2.0

        memory_limit: float | str, default = "auto"
            The total amount of memory in GB you want AutoGluon predictor to use. "auto" means AutoGluon will use all available memory on the system
            (that is detectable by psutil).
            Note that this is only a soft limit! AutoGluon uses this limit to skip training models that are expected to require too much memory or stop
            training a model that would exceed the memory limit. AutoGluon does not guarantee the enforcement of this limit (yet). Nevertheless, we expect
            AutoGluon to abide by the limit in most cases or, at most, go over the limit by a small margin.
            For most virtualized systems (e.g., in the cloud) and local usage on a server or laptop, "auto" is ideal for this parameter. We recommend manually
            setting the memory limit (and any other resources) on systems with shared resources that are controlled by the operating system (e.g., SLURM and
            cgroups). Otherwise, AutoGluon might wrongly assume more resources are available for fitting a model than the operating system allows,
            which can result in model training failing or being very inefficient.
        callbacks : list[AbstractCallback], default = None
            :::{warning}
            Callbacks are an experimental feature and may change in future releases without warning.
            Callback support is preliminary and targeted towards developers.
            :::
            A list of callback objects inheriting from `autogluon.core.callbacks.AbstractCallback`.
            These objects will be called before and after each model fit within trainer.
            They have the ability to skip models or early stop the training process.
            They can also theoretically change the entire logical flow of the trainer code by interacting with the passed `trainer` object.
            For more details, refer to `AbstractCallback` source code.
            If None, no callback objects will be used.

            [Note] Callback objects can be mutated in-place by the fit call if they are stateful.
            Ensure that you avoid reusing a mutated callback object between multiple fit calls.

            [Note] Callback objects are deleted from trainer at the end of the fit call. They will not impact operations such as `refit_full` or `fit_extra`.
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
                Defaults to 1 when unspecified. Value is ignored if `num_bag_folds<=2`.
                Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
                Be warned: This will drastically increase overall runtime, and if using a time limit, can very commonly lead to worse performance.
                It is recommended to increase this value only as a last resort, as it is the least computationally efficient method to improve performance.
            num_stack_levels : int, default = None
                Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `num_stack_levels+1` (set = 0 to disable stack ensembling).
                Disabled by default (0), but we recommend `num_stack_levels=1` to maximize predictive performance.
                To prevent overfitting, `num_bag_folds >= 2` must also be set or else a ValueError will be raised.
            delay_bag_sets : bool, default = False
                Controls when repeats of kfold bagging are executed in AutoGluon when under a time limit.
                We suggest sticking to `False` to avoid overfitting.
                    If True, AutoGluon delays repeating kfold bagging until after evaluating all models
                        from `hyperparameters`, if there is enough time. This allows AutoGluon to explore
                        more hyperparameters to obtain a better final performance but it may lead to
                        more overfitting.
                    If False, AutoGluon repeats kfold bagging immediately after evaluating each model.
                        Thus, AutoGluon might evaluate fewer models with less overfitting.
            holdout_frac : float, default = None
                Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bag_folds != 0` unless `use_bag_holdout == True`).
                Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
                Default value is doubled if `hyperparameter_tune_kwargs` is set, up to a maximum of 0.2.
                Disabled if `num_bag_folds >= 2` unless `use_bag_holdout == True`.
            use_bag_holdout : bool | str, default = False
                If True, a `holdout_frac` portion of the data is held-out from model bagging.
                This held-out data is only used to score models and determine weighted ensemble weights.
                Enable this if there is a large gap between score_val and score_test in stack models.
                Note: If `tuning_data` was specified, `tuning_data` is used as the holdout data.
                Disabled if not bagging.
                If "auto", will be set to True if the training data has >= 1000000 rows, else it will be set to False.
            hyperparameter_tune_kwargs : str or dict, default = None
                Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
                If None, then hyperparameter tuning will not be performed.
                You can either choose to provide a preset
                    Valid preset values:
                        'auto': Performs HPO via bayesian optimization search on NN_TORCH and FASTAI models, and random search on other models using local scheduler.
                        'random': Performs HPO via random search using local scheduler.
                Or provide a dict to specify searchers and schedulers
                    Valid keys:
                        'num_trials': How many HPO trials to run
                        'scheduler': Which scheduler to use
                            Valid values:
                                'local': Local scheduler that schedule trials FIFO
                        'searcher': Which searching algorithm to use
                            'local_random': Uses the 'random' searcher
                            'random': Perform random search
                            'auto': Perform bayesian optimization search on NN_TORCH and FASTAI models. Perform random search on other models.
                    The 'scheduler' and 'searcher' key are required when providing a dict.
                    An example of a valid dict:
                        hyperparameter_tune_kwargs = {
                            'num_trials': 5,
                            'scheduler' : 'local',
                            'searcher': 'auto',
                        }
            feature_prune_kwargs: dict, default = None
                Performs layer-wise feature pruning via recursive feature elimination with permutation feature importance.
                This fits all models in a stack layer once, discovers a pruned set of features, fits all models in the stack layer
                again with the pruned set of features, and updates input feature lists for models whose validation score improved.
                If None, do not perform feature pruning. If empty dictionary, perform feature pruning with default configurations.
                For valid dictionary keys, refer to :class:`autogluon.core.utils.feature_selection.FeatureSelector` and
                `autogluon.core.trainer.abstract_trainer.AbstractTabularTrainer._proxy_model_feature_prune` documentation.
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
            ds_args : dict, see below for default
                Keyword arguments for dynamic stacking, only used if `dynamic_stacking=True`. These keyword arguments control the behavior of dynamic stacking
                and determine how AutoGluon tries to detect stacked overfitting. To detect stacked overfitting, AutoGluon will fit itself (so called sub-fits)
                on a subset (for holdout) or multiple subsets (for repeated cross-validation) and use the predictions of AutoGluon on the validation data to
                detect stacked overfitting. The sub-fits stop and stacking will be disabled if any sub-fit shows stacked overfitting.
                Allowed keys and values are:
                    `detection_time_frac` : float in (0,1), default = 1/4
                        Determines how much of the original training time is used for detecting stacked overfitting.
                        When using (repeated) cross-validation, each sub-fit will be fit for `1/n_splits * detection_time_frac * time_limit`.
                        If no time limit is given to AutoGluon, this parameter is ignored and AutoGluon is fit without a time limit in the sub-fit.
                    `validation_procedure`: str, default = 'holdout'
                        Determines the validation procedure used to detect stacked overfitting. Can be either `cv` or `holdout`.
                            If `validation_procedure='holdout'` and `holdout_data` is not specified (default), then `holdout_frac` determines the holdout data.
                            If `validation_procedure='holdout'` and `holdout_data` is specified, then the provided `holdout_data` is used for validation.
                            If `validation_procedure='cv'`, `n_folds` and `n_repeats` determine the kind cross-validation procedure.
                    `holdout_frac` : float in (0,1), default = 1/9
                        Determines how much of the original training data is used for the holdout data during holdout validation.
                        Ignored if `holdout_data` is not None.
                    `n_folds` : int in [2, +inf), default = 2
                        Number of folds to use for cross-validation.
                    `n_repeats` : int [1, +inf), default = 1
                        Number of repeats to use for repeated cross-validation. If set to 1, performs 1-repeated cross-validation which is equivalent to
                        cross-validation without repeats.
                    `memory_safe_fits` : bool, default = True
                        If True, AutoGluon runs each sub-fit in a ray-based subprocess to avoid memory leakage that exist due to Python's lackluster
                        garbage collector.
                    `clean_up_fits` : bool, default = True
                        If True, AutoGluon will remove all saved information from sub-fits from disk.
                        If False, the sub-fits are kept on disk and `self._sub_fits` will store paths to the sub-fits, which can be loaded just like any other
                        predictor from disk using `TabularPredictor.load()`.
                    `enable_ray_logging` : bool, default = True
                        If True, will log the dynamic stacking sub-fit when ray is used (`memory_safe_fits=True`).
                        Note that because of how ray works, this may cause extra unwanted logging in the main fit process after dynamic stacking completes.
                    `enable_callbacks` : bool, default = False
                        If True, will perform a deepcopy on the specified user callbacks and enable them during the DyStack call.
                        If False, will not include callbacks in the DyStack call.
                    `holdout_data`: str or :class:`pd.DataFrame`, default = None
                        Another dataset containing validation data reserved for detecting stacked overfitting. This dataset should be in the same format as
                        `train_data`. If str is passed, `holdout_data` will be loaded using the str value as the file path.
                        If `holdout_data` is not None, the sub-fit is fit on all of `train_data` and the full fit is fit on all of `train_data` and
                        `holdout_data` combined.
            included_model_types : list, default = None
                To only include listed model types for training during `fit()`.
                Models that are listed in `included_model_types` but not in `hyperparameters` will be ignored.
                Reference `hyperparameters` documentation for what models correspond to each value.
                Useful when only a subset of model needs to be trained and the `hyperparameters` dictionary is difficult or time-consuming.
                    Example: To include both 'GBM' and 'FASTAI' models, specify `included_model_types=['GBM', 'FASTAI']`.
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
            save_bag_folds : bool, default = True
                If True, will save the bagged fold models to disk.
                If False, will not save the bagged fold models, only keeping their metadata and out-of-fold predictions.
                    Note: The bagged models will not be available for prediction, only use this if you intend to call `refit_full`.
                    The purpose of setting it to False is that it greatly decreases the peak disk usage of the predictor during the fit call when bagging.
                    Note that this makes refit_full slightly more likely to crash in scenarios where the dataset is large relative to available system memory.
                    This is because by default, refit_full will fall back to cloning the first fold of the bagged model in case it lacks memory to refit.
                    However, if `save_bag_folds=False`, this fallback isn't possible, as there is not fold model to clone because it wasn't saved.
                    In this scenario, refit will raise an exception for `save_bag_folds=False`, but will succeed if `save_bag_folds=True`.
                Final disk usage of predictor will be identical regardless of the setting after `predictor.delete_models(models_to_keep="best")` is called post-fit.
            set_best_to_refit_full : bool, default = False
                If True, will change the default model that Predictor uses for prediction when model is not specified to the refit_full version of the model that exhibited the highest validation score.
                Only valid if `refit_full` is set.
            keep_only_best : bool, default = False
                If True, only the best model and its ancestor models are saved in the outputted `predictor`. All other models are deleted.
                    If you only care about deploying the most accurate predictor with the smallest file-size and no longer need any of the other trained models or functionality beyond prediction on new data, then set: `keep_only_best=True`, `save_space=True`.
                    This is equivalent to calling `predictor.delete_models(models_to_keep='best')` directly after `fit()`.
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
                [Experimental Parameter] UNUSED.
                Collection of data without labels that we can use to pretrain on.
                This is the same schema as train_data, except without the labels.
                Currently, unlabeled_data is not used by any model.
            verbosity : int
                If specified, overrides the existing `predictor.verbosity` value.
            raise_on_model_failure: bool, default = False
                If True, will raise on any exception during model training.
                    This is useful when using a debugger during development to identify the cause of model failures.
                    This should only be used for debugging.
                If False, will try to skip to the next model if an exception occurred during model training.
                    This is the default logic and is a core principle of AutoGluon's design.

                .. versionadded:: 1.3.0
            raise_on_no_models_fitted: bool, default = True
                If True, will raise a RuntimeError if no models were successfully fit during `fit()`.
            calibrate: bool or str, default = 'auto'
                Note: It is recommended to use ['auto', False] as the values and avoid True.
                If 'auto' will automatically set to True if the problem_type and eval_metric are suitable for calibration.
                If True and the problem_type is classification, temperature scaling will be used to calibrate the Predictor's estimated class probabilities
                (which may improve metrics like log_loss) and will train a scalar parameter on the validation set.
                If True and the problem_type is quantile regression, conformalization will be used to calibrate the Predictor's estimated quantiles
                (which may improve the prediction interval coverage, and bagging could further improve it) and will compute a set of scalar parameters on the validation set.
            test_data : str or :class:`pd.DataFrame`, default = None
                Table of the test data.
                If str is passed, `test_data` will be loaded using the str value as the file path.
                NOTE: This test_data is NEVER SEEN by the model during training and, if specified, is only used for logging purposes (i.e. for learning curve generation).
                This test_data should be treated the same way test data is used in predictor.leaderboard.
            learning_curves : bool or dict, default = None
                If bool and is True, default learning curve hyperparameter ag_args will be initialized for each of the models included in the ensemble.
                    By default, learning curves will include eval_metric scores specified in fit call arguments.
                    This can be overwritten as shown below.
                If dict, user can pass learning_curves parameters to be initialized as ag_args in the following format:
                    learning_curves = {
                        "metrics": str or list(str) or Scorer or list(Scorer):
                            autogluon metric scorer(s) to be calculated at each iteration, represented as Scorer object(s) or scorer name(s) (str)
                        "use_error": bool : whether to use error or score format for metrics listed above
                    }

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
        if self.is_fit:
            raise AssertionError("Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new `Predictor`.")

        verbosity = kwargs.get("verbosity", self.verbosity)
        set_logger_verbosity(verbosity)
        warn_if_mlflow_autologging_is_enabled(logger=logger)

        if verbosity >= 2:
            if verbosity == 2:
                logger.log(20, f"Verbosity: 2 (Standard Logging)")
            elif verbosity == 3:
                logger.log(20, f"Verbosity: 3 (Detailed Logging)")
            elif verbosity >= 4:
                logger.log(20, f"Verbosity: {verbosity} (Maximum Logging)")

        resource_manager: ResourceManager = get_resource_manager()
        include_gpu_count = resource_manager.get_gpu_count_torch() or verbosity >= 3
        sys_msg = get_ag_system_info(path=self.path, include_gpu_count=include_gpu_count)
        logger.log(20, sys_msg)

        if presets:
            if not isinstance(presets, list):
                presets = [presets]
            logger.log(20, f"Presets specified: {presets}")
        else:
            logger.log(
                20,
                "No presets specified! To achieve strong results with AutoGluon, it is recommended to use the available presets. Defaulting to `'medium'`...\n"
                "\tRecommended Presets (For more details refer to https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html#presets):\n"
                "\tpresets='extreme'  : New in v1.5: The state-of-the-art for tabular data. Massively better than 'best' on datasets <100000 samples by using new Tabular Foundation Models (TFMs) meta-learned on https://tabarena.ai: TabPFNv2, TabICL, Mitra, TabDPT, and TabM. Requires a GPU and `pip install autogluon.tabular[tabarena]` to install TabPFN, TabICL, and TabDPT.\n"
                "\tpresets='best'     : Maximize accuracy. Recommended for most users. Use in competitions and benchmarks.\n"
                "\tpresets='best_v150': New in v1.5: Better quality than 'best' and 5x+ faster to train. Give it a try!\n"
                "\tpresets='high'     : Strong accuracy with fast inference speed.\n"
                "\tpresets='high_v150': New in v1.5: Better quality than 'high' and 5x+ faster to train. Give it a try!\n"
                "\tpresets='good'     : Good accuracy with very fast inference speed.\n"
                "\tpresets='medium'   : Fast training time, ideal for initial prototyping.",
            )

        kwargs_orig = kwargs.copy()

        if verbosity >= 3:
            logger.log(20, "============ fit kwarg info ============")
            logger.log(20, "User Specified kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs_orig)}")

        kwargs = self._validate_fit_kwargs(kwargs=kwargs)

        if verbosity >= 3:
            logger.log(20, "Full kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs)}")
            logger.log(20, "========================================")

        self._validate_num_cpus(num_cpus=num_cpus)
        self._validate_num_gpus(num_gpus=num_gpus)
        self._validate_and_set_memory_limit(memory_limit=memory_limit)
        self._validate_calibrate_decision_threshold(calibrate_decision_threshold=calibrate_decision_threshold)
        self._validate_fit_strategy(fit_strategy=fit_strategy)

        auto_stack = kwargs["auto_stack"]
        feature_generator = kwargs["feature_generator"]
        unlabeled_data = kwargs["unlabeled_data"]
        ag_args = kwargs["ag_args"]
        ag_args_fit = kwargs["ag_args_fit"]
        ag_args_ensemble = kwargs["ag_args_ensemble"]
        included_model_types = kwargs["included_model_types"]
        excluded_model_types = kwargs["excluded_model_types"]
        use_bag_holdout = kwargs["use_bag_holdout"]
        ds_args: dict = kwargs["ds_args"]
        delay_bag_sets: bool = kwargs["delay_bag_sets"]
        test_data = kwargs["test_data"]
        learning_curves = kwargs["learning_curves"]
        raise_on_model_failure = kwargs["raise_on_model_failure"]

        if ag_args is None:
            ag_args = {}
        ag_args = self._set_hyperparameter_tune_kwargs_in_ag_args(kwargs["hyperparameter_tune_kwargs"], ag_args, time_limit=time_limit)

        feature_generator_init_kwargs = kwargs["_feature_generator_kwargs"]
        if feature_generator_init_kwargs is None:
            feature_generator_init_kwargs = dict()

        train_data, tuning_data, test_data, unlabeled_data = self._validate_fit_data(
            train_data=train_data, tuning_data=tuning_data, test_data=test_data, unlabeled_data=unlabeled_data
        )
        infer_limit, infer_limit_batch_size = self._validate_infer_limit(infer_limit=infer_limit, infer_limit_batch_size=infer_limit_batch_size)

        # TODO: Temporary for v1.4. Make this more extensible for v1.5 by letting users make their own dynamic hyperparameters.
        dynamic_hyperparameters = kwargs["_experimental_dynamic_hyperparameters"]
        if dynamic_hyperparameters:
            logger.log(20, f"`extreme_v140` preset uses a dynamic portfolio based on dataset size...")
            assert hyperparameters is None, f"hyperparameters must be unspecified when `_experimental_dynamic_hyperparameters=True`."
            n_samples = len(train_data)
            if n_samples > 30000:
                data_size = "large"
            else:
                data_size = "small"
            assert data_size in ["large", "small"]
            if data_size == "large":
                logger.log(20, f"\tDetected data size: large (>30000 samples), using `zeroshot` portfolio (identical to 'best_quality' preset).")
                hyperparameters = "zeroshot"
            else:
                if "num_stack_levels" not in kwargs_orig:
                    # disable stacking for tabfm portfolio
                    num_stack_levels = 0
                    kwargs["num_stack_levels"] = 0
                logger.log(
                    20,
                    f"\tDetected data size: small (<=30000 samples), using `zeroshot_2025_tabfm` portfolio."
                    f"\n\t\tNote: `zeroshot_2025_tabfm` portfolio requires a CUDA compatible GPU for best performance."
                    f"\n\t\tMake sure you have all the relevant dependencies installed: "
                    f"`pip install autogluon.tabular[tabarena]`."
                    f"\n\t\tIt is strongly recommended to use a machine with 64+ GB memory "
                    f"and a CUDA compatible GPU with 32+ GB vRAM when using this preset. "
                    f"\n\t\tThis portfolio will download foundation model weights from HuggingFace during training. "
                    f"Ensure you have an internet connection or have pre-downloaded the weights to use these models."
                    f"\n\t\tThis portfolio was meta-learned with TabArena: https://tabarena.ai"
                )
                hyperparameters = "zeroshot_2025_tabfm"

        if hyperparameters is None:
            hyperparameters = "default"
        if isinstance(hyperparameters, str):
            hyperparameters_str = hyperparameters
            hyperparameters = get_hyperparameter_config(hyperparameters)
            logger.log(
                20,
                f"Using hyperparameters preset: hyperparameters='{hyperparameters_str}'",
            )
        self._validate_hyperparameters(hyperparameters=hyperparameters)
        self.fit_hyperparameters_ = hyperparameters

        if "enable_raw_text_features" not in feature_generator_init_kwargs:
            if self._check_if_hyperparameters_handle_text(hyperparameters=hyperparameters):
                feature_generator_init_kwargs["enable_raw_text_features"] = True

        if feature_metadata is not None and isinstance(feature_metadata, str) and feature_metadata == "infer":
            feature_metadata = None
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=feature_generator_init_kwargs)

        if self.problem_type is not None:
            inferred_problem_type = self.problem_type
        else:
            self._learner.validate_label(X=train_data)
            inferred_problem_type = self._learner.infer_problem_type(y=train_data[self.label], silent=True)

        learning_curves = self._initialize_learning_curve_params(learning_curves=learning_curves, problem_type=inferred_problem_type)
        if len(learning_curves) == 0:
            test_data = None
        if ag_args_fit is not None:
            ag_args_fit.update(learning_curves)
        else:
            ag_args_fit = learning_curves

        use_bag_holdout_was_auto = False
        dynamic_stacking_was_auto = False
        if isinstance(use_bag_holdout,str) and use_bag_holdout == "auto":
            use_bag_holdout = None
            use_bag_holdout_was_auto = True
        if isinstance(dynamic_stacking,str) and dynamic_stacking == "auto":
            dynamic_stacking = None
            dynamic_stacking_was_auto = True

        (
            num_bag_folds,
            num_bag_sets,
            num_stack_levels,
            dynamic_stacking,
            use_bag_holdout,
            holdout_frac,
            refit_full,
        ) = get_validation_and_stacking_method(
            num_bag_folds=kwargs["num_bag_folds"],
            num_bag_sets=kwargs["num_bag_sets"],
            use_bag_holdout=use_bag_holdout,
            holdout_frac=kwargs["holdout_frac"],
            auto_stack=auto_stack,
            num_stack_levels=kwargs["num_stack_levels"],
            dynamic_stacking=dynamic_stacking,
            refit_full=kwargs["refit_full"],
            num_train_rows=len(train_data),
            problem_type=inferred_problem_type,
            hpo_enabled=ag_args.get("hyperparameter_tune_kwargs", None) is not None,
        )

        num_bag_folds, num_bag_sets, num_stack_levels, dynamic_stacking, use_bag_holdout = self._sanitize_stack_args(
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            num_train_rows=len(train_data),
            dynamic_stacking=dynamic_stacking,
            use_bag_holdout=use_bag_holdout,
            use_bag_holdout_was_auto=use_bag_holdout_was_auto,
            dynamic_stacking_was_auto=dynamic_stacking_was_auto,
        )
        if auto_stack:
            logger.log(
                20,
                f"Stack configuration (auto_stack={auto_stack}): "
                f"num_stack_levels={num_stack_levels}, num_bag_folds={num_bag_folds}, num_bag_sets={num_bag_sets}",
            )

        if kwargs["save_bag_folds"] is not None and kwargs["_save_bag_folds"] is not None:
            raise ValueError(
                f"Cannot specify both `save_bag_folds` and `_save_bag_folds` at the same time. "
                f"(save_bag_folds={kwargs['save_bag_folds']}, _save_bag_folds={kwargs['_save_bag_folds']}"
            )
        elif kwargs["_save_bag_folds"] is not None:
            kwargs["save_bag_folds"] = kwargs["_save_bag_folds"]

        if kwargs["save_bag_folds"] is not None:
            assert isinstance(kwargs["save_bag_folds"], bool), f"save_bag_folds must be a bool, found: {type(kwargs['save_bag_folds'])}"
            if use_bag_holdout and not kwargs["save_bag_folds"]:
                logger.log(
                    30,
                    f"WARNING: Attempted to disable saving of bagged fold models when `use_bag_holdout=True`. Forcing `save_bag_folds=True` to avoid errors.",
                )
            else:
                if num_bag_folds > 0 and not kwargs["save_bag_folds"]:
                    logger.log(
                        20,
                        f"Note: `save_bag_folds=False`! This will greatly reduce peak disk usage during fit (by ~{num_bag_folds}x), "
                        f"but runs the risk of an out-of-memory error during model refit if memory is small relative to the data size.\n"
                        f"\tYou can avoid this risk by setting `save_bag_folds=True`.",
                    )
                if ag_args_ensemble is None:
                    ag_args_ensemble = {}
                ag_args_ensemble["save_bag_folds"] = kwargs["save_bag_folds"]

        if time_limit is None:
            mb_mem_usage_train_data = get_approximate_df_mem_usage(train_data, sample_ratio=0.2).sum() / 1e6
            num_rows_train = len(train_data)
            if mb_mem_usage_train_data >= 50 or num_rows_train >= 100000:
                logger.log(
                    20,
                    f"Warning: Training may take a very long time because `time_limit` was not specified and `train_data` is large ({num_rows_train} samples, {round(mb_mem_usage_train_data, 2)} MB).",
                )
                logger.log(
                    20,
                    f"\tConsider setting `time_limit` to ensure training finishes within an expected duration or experiment with a small portion of `train_data` to identify an ideal `presets` and `hyperparameters` configuration.",
                )

        core_kwargs = {
            "total_resources": {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            },
            "ag_args": ag_args,
            "ag_args_ensemble": ag_args_ensemble,
            "ag_args_fit": ag_args_fit,
            "included_model_types": included_model_types,
            "excluded_model_types": excluded_model_types,
            "feature_prune_kwargs": kwargs.get("feature_prune_kwargs", None),
            "delay_bag_sets": delay_bag_sets,
            "fit_strategy": fit_strategy,
        }
        aux_kwargs = {
            "total_resources": {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            },
        }
        if fit_weighted_ensemble is False:
            aux_kwargs["fit_weighted_ensemble"] = False
        aux_kwargs["fit_full_last_level_weighted_ensemble"] = fit_full_last_level_weighted_ensemble
        aux_kwargs["full_weighted_ensemble_additionally"] = full_weighted_ensemble_additionally

        ag_fit_kwargs = dict(
            X=train_data,
            X_val=tuning_data,
            X_test=test_data,
            X_unlabeled=unlabeled_data,
            holdout_frac=holdout_frac,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            hyperparameters=hyperparameters,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
            time_limit=time_limit,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            verbosity=verbosity,
            use_bag_holdout=use_bag_holdout,
            callbacks=callbacks,
            raise_on_model_failure=raise_on_model_failure,
        )
        ag_post_fit_kwargs = dict(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=refit_full,
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
            save_space=kwargs["save_space"],
            calibrate=kwargs["calibrate"],
            calibrate_decision_threshold=calibrate_decision_threshold,
            infer_limit=infer_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            fit_strategy=fit_strategy,
            raise_on_no_models_fitted=kwargs["raise_on_no_models_fitted"],
        )
        if dynamic_stacking:
            logger.log(
                20,
                f"DyStack is enabled (dynamic_stacking={dynamic_stacking}). "
                "AutoGluon will try to determine whether the input data is affected by stacked overfitting and enable or disable stacking as a consequence.",
            )
            num_stack_levels, time_limit = self._dynamic_stacking(**ds_args, ag_fit_kwargs=ag_fit_kwargs, ag_post_fit_kwargs=ag_post_fit_kwargs)
            logger.info(
                f"Starting main fit with num_stack_levels={num_stack_levels}.\n"
                f"\tFor future fit calls on this dataset, you can skip DyStack to save time: "
                f"`predictor.fit(..., dynamic_stacking=False, num_stack_levels={num_stack_levels})`"
            )

            if (time_limit is not None) and (time_limit <= 0):
                raise AssertionError(
                    f"Not enough time left to train models for the full fit. "
                    f"Consider specifying a larger time_limit or setting `dynamic_stacking=False`. Time remaining: {time_limit:.2f}s"
                )

            ag_fit_kwargs["num_stack_levels"] = num_stack_levels
            ag_fit_kwargs["time_limit"] = time_limit

        # keep track of the fit strategy used for future calls
        self._fit_strategy = fit_strategy

        self._fit(ag_fit_kwargs=ag_fit_kwargs, ag_post_fit_kwargs=ag_post_fit_kwargs)

        return self

    def _fit(self, ag_fit_kwargs: dict, ag_post_fit_kwargs: dict):
        self.save(silent=True)  # Save predictor to disk to enable prediction and training after interrupt
        self._learner.fit(**ag_fit_kwargs)
        self._set_post_fit_vars()
        self._post_fit(**ag_post_fit_kwargs)
        self.save()

    # TODO: When >2 layers, will only choose between using all layers or using only base models. Would be better to choose the optimal layer.
    def _dynamic_stacking(
        self,
        ag_fit_kwargs: dict,
        ag_post_fit_kwargs: dict,
        validation_procedure: bool,
        detection_time_frac: float,
        holdout_frac: float,
        n_folds: int,
        n_repeats: int,
        memory_safe_fits: bool,
        clean_up_fits: bool,
        enable_ray_logging: bool,
        enable_callbacks: bool,
        holdout_data: Optional[Union[str, pd.DataFrame, None]] = None,
    ):
        """Dynamically determines if stacking is used or not by validating the behavior of a sub-fit of AutoGluon that uses stacking on held out data.
        See `ds_args` in the docstring of `fit()` for details on the parameters."""
        time_start = time.time()
        time_limit_og = ag_fit_kwargs["time_limit"]
        org_num_stack_levels = ag_fit_kwargs["num_stack_levels"]
        ds_fit_context = os.path.join(self._learner.path_context_og, "ds_sub_fit")
        logger.info(
            "\tThis is used to identify the optimal `num_stack_levels` value. "
            "Copies of AutoGluon will be fit on subsets of the data. "
            "Then holdout validation data is used to detect stacked overfitting."
        )

        if time_limit_og is not None:
            time_limit = int(time_limit_og * detection_time_frac)
            logger.info(f"\tRunning DyStack for up to {time_limit}s of the {time_limit_og}s of remaining time ({detection_time_frac*100:.0f}%).")
        else:
            logger.info(f"\tWarning: No time limit provided for DyStack. This could take awhile.")
            time_limit = None

        # -- Avoid copying data
        X = ag_fit_kwargs.pop("X")
        X_val = ag_fit_kwargs.pop("X_val")
        X_unlabeled = ag_fit_kwargs.pop("X_unlabeled")
        callbacks = None
        if not enable_callbacks:
            callbacks = ag_fit_kwargs.pop("callbacks")
        inner_ag_fit_kwargs = copy.deepcopy(ag_fit_kwargs)
        if not enable_callbacks:
            ag_fit_kwargs["callbacks"] = callbacks
        inner_ag_fit_kwargs["X_val"] = X_val
        inner_ag_fit_kwargs["X_unlabeled"] = X_unlabeled
        inner_ag_post_fit_kwargs = copy.deepcopy(ag_post_fit_kwargs)
        inner_ag_post_fit_kwargs["keep_only_best"] = False  # Do not keep only best, otherwise it eliminates the purpose of the comparison
        inner_ag_post_fit_kwargs["calibrate"] = False  # Do not calibrate as calibration is only applied to the model with the best validation score
        # FIXME: Ensure all weighted ensembles have skip connections

        # Verify problem type is set
        if self.problem_type is None:
            self._learner.problem_type = self._learner.infer_problem_type(y=X[self.label], silent=True)

        ds_fit_kwargs = dict(
            clean_up_fits=clean_up_fits,
            memory_safe_fits=memory_safe_fits,
            enable_ray_logging=enable_ray_logging,
        )

        # -- Validation Method
        if validation_procedure == "holdout":
            if holdout_data is None:
                ds_fit_kwargs.update(dict(holdout_frac=holdout_frac, ds_fit_context=os.path.join(ds_fit_context, "sub_fit_ho")))
            else:
                _, holdout_data, _, _ = self._validate_fit_data(train_data=X, tuning_data=holdout_data)
                ds_fit_kwargs["ds_fit_context"] = os.path.join(ds_fit_context, "sub_fit_custom_ho")

            stacked_overfitting = self._sub_fit_memory_save_wrapper(
                train_data=X,
                time_limit=time_limit,
                time_start=time_start,
                ds_fit_kwargs=ds_fit_kwargs,
                ag_fit_kwargs=inner_ag_fit_kwargs,
                ag_post_fit_kwargs=inner_ag_post_fit_kwargs,
                holdout_data=holdout_data,
            )
        else:
            # Holdout is false, use (repeated) cross-validation
            is_stratified = self.problem_type in [BINARY, MULTICLASS]
            is_binned = self.problem_type in [REGRESSION, QUANTILE]
            self._learner._validate_groups(X=X, X_val=X_val)  # Validate splits before splitting
            splits = CVSplitter(
                n_splits=n_folds,
                n_repeats=n_repeats,
                groups=self._learner.groups,
                stratify=is_stratified,
                bin=is_binned,
                random_state=42,
            ).split(
                X=X.drop(self.label, axis=1),
                y=X[self.label]
            )
            n_splits = len(splits)
            logger.info(
                f'\tStarting (repeated-)cross-validation-based sub-fits for dynamic stacking. Context path: "{ds_fit_context}"'
                f"Run at most {n_splits} sub-fits based on {n_repeats}-repeated {n_folds}-fold cross-validation."
            )
            np.random.RandomState(42).shuffle(splits)  # shuffle splits to mix up order such that if only one of the repeats shows leakage we might stop early.
            for split_index, (train_indices, val_indices) in enumerate(splits):
                if time_limit is None:
                    sub_fit_time = None
                else:
                    time_spend_sub_fits_so_far = int(time.time() - time_start)
                    rest_time = time_limit - time_spend_sub_fits_so_far
                    sub_fit_time = int(1 / (n_splits - split_index) * rest_time)  # if we are faster, give more time to rest of the folds.
                    if sub_fit_time <= 0:
                        logger.info(f"\tStop cross-validation during dynamic stacking early as no more time left. Consider specifying a larger time_limit.")
                        break
                ds_fit_kwargs.update(
                    dict(
                        train_indices=train_indices,
                        val_indices=val_indices,
                        ds_fit_context=os.path.join(ds_fit_context, f"sub_fit_{split_index}"),
                    )
                )
                logger.info(
                    f"\tStarting sub-fit {split_index + 1}. Time limit for the sub-fit of this split is: {'unlimited' if sub_fit_time is None else sub_fit_time}."
                )
                stacked_overfitting = self._sub_fit_memory_save_wrapper(
                    train_data=X,
                    time_limit=time_limit,
                    time_start=time_start,
                    ds_fit_kwargs=ds_fit_kwargs,
                    ag_fit_kwargs=inner_ag_fit_kwargs,
                    ag_post_fit_kwargs=inner_ag_post_fit_kwargs,
                    holdout_data=holdout_data,
                )
                if stacked_overfitting:
                    break

            del splits

        if clean_up_fits:
            try:
                shutil.rmtree(path=ds_fit_context)
            except FileNotFoundError as e:
                pass

        # -- Determine rest time and new num_stack_levels
        time_spend_sub_fits = time.time() - time_start
        num_stack_levels = 0 if stacked_overfitting else org_num_stack_levels
        self._stacked_overfitting_occurred = stacked_overfitting

        logger.info(f"\t{num_stack_levels}\t = Optimal   num_stack_levels (Stacked Overfitting Occurred: {self._stacked_overfitting_occurred})")
        log_str = f"\t{round(time_spend_sub_fits)}s\t = DyStack   runtime"
        if time_limit_og is None:
            time_limit_fit_full = None
        else:
            time_limit_fit_full = time_limit_og - time_spend_sub_fits
            log_str += f" |\t{round(time_limit_fit_full)}s\t = Remaining runtime"
        logger.info(log_str)

        # -- Revert back
        del inner_ag_fit_kwargs
        if holdout_data is None:
            ag_fit_kwargs["X"] = X
        else:
            logger.log(20, "\tConcatenating holdout data from dynamic stacking to the training data for the full fit (and reset the index).")
            ag_fit_kwargs["X"] = pd.concat([X, holdout_data], ignore_index=True)

        ag_fit_kwargs["X_val"] = X_val
        ag_fit_kwargs["X_unlabeled"] = X_unlabeled

        return num_stack_levels, time_limit_fit_full

    def _sub_fit_memory_save_wrapper(
        self,
        train_data: Union[str, pd.DataFrame],
        time_limit: float,
        time_start: float,
        ds_fit_kwargs: dict,
        ag_fit_kwargs: dict,
        ag_post_fit_kwargs: dict,
        holdout_data=Union[str, pd.DataFrame, None],
    ):
        """Tries to run the sub-fit in a subprocess (managed by ray). Similar to AutoGluon's parallel fold fitting strategies,
        this code does not shut down ray after usage. Otherwise, we would also kill outer-scope ray usage."""
        memory_safe_fits = ds_fit_kwargs.get("memory_safe_fits", True)
        enable_ray_logging = ds_fit_kwargs.get("enable_ray_logging", True)
        normal_fit = False
        total_resources = ag_fit_kwargs["core_kwargs"]["total_resources"]

        if memory_safe_fits == "auto":
            num_gpus = total_resources.get("num_gpus", "auto")
            if num_gpus == "auto":
                num_gpus = ResourceManager.get_gpu_count_torch()
                if num_gpus > 0:
                    logger.log(
                        30,
                        f"DyStack: Disabling memory safe fit mode in DyStack "
                        f"because GPUs were detected and num_gpus='auto' (GPUs cannot be used in memory safe fit mode). "
                        f"If you want to use memory safe fit mode, manually set `num_gpus=0`."
                    )
            if num_gpus > 0:
                memory_safe_fits = False
            else:
                memory_safe_fits = True


        if memory_safe_fits:
            try:
                _ds_ray = try_import_ray()
                if not _ds_ray.is_initialized():
                    if enable_ray_logging:
                        logger.info(
                            f"\tRunning DyStack sub-fit in a ray process to avoid memory leakage. "
                            "Enabling ray logging (enable_ray_logging=True). Specify `ds_args={'enable_ray_logging': False}` if you experience logging issues."
                        )
                        _ds_ray.init()
                    else:
                        logger.info(
                            f"\tRunning DyStack sub-fit in a ray process to avoid memory leakage. "
                            "Logs will not be shown until this process is complete (enable_ray_logging=False). "
                            "You can experimentally enable logging by specifying `ds_args={'enable_ray_logging': True}`."
                        )
                        _ds_ray.init(
                            logging_level=logging.ERROR,
                            log_to_driver=False,
                        )
            except Exception as e:
                warnings.warn(f"Failed to use ray for memory safe fits. Falling back to normal fit. Error: {repr(e)}", stacklevel=2)
                _ds_ray = None

            if time_limit is not None:
                # Subtract time taken to initialize ray
                time_limit -= time.time() - time_start
                if time_limit <= 0:
                    logger.log(30, f"Warning: Not enough time to fit DyStack! Skipping...")
                    return False

            if holdout_data is None:
                logger.info(f"\t\tContext path: \"{ds_fit_kwargs['ds_fit_context']}\"")
            else:
                logger.info(f"\t\tRunning DyStack holdout-based sub-fit with custom validation data. Context path: \"{ds_fit_kwargs['ds_fit_context']}\"")

            if _ds_ray is not None:
                # Handle resources
                # FIXME: what about distributed?

                num_cpus = total_resources.get("num_cpus", "auto")

                if num_cpus == "auto":
                    num_cpus = ResourceManager.get_cpu_count()

                # num_gpus is treated oddly in ray, commented out until we find a better solution
                # num_gpus = total_resources.get("num_gpus", "auto")
                # if num_gpus == "auto":
                #     num_gpus = ResourceManager.get_gpu_count()

                # Handle expensive data via put
                ag_fit_kwargs_ref = _ds_ray.put(ag_fit_kwargs)
                predictor_ref = _ds_ray.put(self)
                train_data_ref = _ds_ray.put(train_data)

                if holdout_data is not None:
                    holdout_data_ref = _ds_ray.put(holdout_data)
                else:
                    holdout_data_ref = None

                # Call sub fit in its own subprocess via ray
                sub_fit_caller = _ds_ray.remote(max_calls=1)(_dystack)
                # FIXME: For some reason ray does not treat `num_cpus` and `num_gpus` the same.
                #  For `num_gpus`, the process will reserve the capacity and is unable to share it to child ray processes, causing a deadlock.
                #  For `num_cpus`, the value is completely ignored by children, and they can even use more num_cpus than the parent.
                #  Because of this, num_gpus is set to 0 here to avoid a deadlock, but num_cpus does not need to be changed.
                #  For more info, refer to Ray documentation: https://docs.ray.io/en/latest/ray-core/tasks/nested-tasks.html#yielding-resources-while-blocked
                ref = sub_fit_caller.options(num_cpus=num_cpus, num_gpus=0).remote(
                    predictor=predictor_ref,
                    train_data=train_data_ref,
                    time_limit=time_limit,
                    ds_fit_kwargs=ds_fit_kwargs,
                    ag_fit_kwargs=ag_fit_kwargs_ref,
                    ag_post_fit_kwargs=ag_post_fit_kwargs,
                    holdout_data=holdout_data_ref,
                )
                finished, unfinished = _ds_ray.wait([ref], num_returns=1)
                stacked_overfitting, ho_leaderboard, exception = _ds_ray.get(finished[0])

                # TODO: This is present to ensure worker logs are properly logged and don't get skipped / printed out of order.
                #  Ideally find a faster way to do this that doesn't introduce a 100 ms overhead.
                time.sleep(0.1)
            else:
                normal_fit = True
        else:
            normal_fit = True

        if normal_fit:
            stacked_overfitting, ho_leaderboard, exception = _dystack(
                predictor=self,
                train_data=train_data,
                time_limit=time_limit,
                ds_fit_kwargs=ds_fit_kwargs,
                ag_fit_kwargs=ag_fit_kwargs,
                ag_post_fit_kwargs=ag_post_fit_kwargs,
                holdout_data=holdout_data,
            )

        if exception is not None:
            logger.log(40, f"Warning: Exception encountered during DyStack sub-fit:\n\t{exception}")
        if ho_leaderboard is not None:
            logger.log(20, "Leaderboard on holdout data (DyStack):")
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                # Rename to avoid confusion for the user
                logger.log(20, ho_leaderboard.rename({"score_test": "score_holdout"}, axis=1))

        if not normal_fit and enable_ray_logging:
            try:
                # Disables ray logging to avoid log spam in main process after DyStack completes
                # This is somewhat of a hack, and needs to use private APIs of ray. It is unclear how to do this in a different way.
                # Note: Once this is done, it cannot be undone,
                # and the only known way to re-enable ray logging in the process is to call `ray.shutdown()` and `ray.init()`
                _ds_ray._private.ray_logging.global_worker_stdstream_dispatcher.remove_handler("ray_print_logs")
            except Exception as e:
                logger.log(
                    40,
                    "WARNING: ray logging verbosity fix raised an exception. Ray might give overly verbose logging output. "
                    "Please open a GitHub issue to notify the AutoGluon developers of this issue. "
                    "You can avoid this issue by specifying `ds_args={'enable_ray_logging': False}`. Exception detailed below:"
                    f"\n{e}",
                )

        return stacked_overfitting

    def _post_fit(
        self,
        keep_only_best=False,
        refit_full=False,
        set_best_to_refit_full=False,
        save_space=False,
        calibrate=False,
        calibrate_decision_threshold=False,
        infer_limit=None,
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
        refit_full_kwargs: dict = None,
        fit_strategy: Literal["auto", "sequential", "parallel"] = "sequential",
        raise_on_no_models_fitted: bool = True,
    ):
        if refit_full_kwargs is None:
            refit_full_kwargs = {}
        if not self.model_names():
            if raise_on_no_models_fitted:
                raise RuntimeError(
                    "No models were trained successfully during fit()."
                    " Inspect the log output or increase verbosity to determine why no models were fit."
                    " Alternatively, set `raise_on_no_models_fitted` to False during the fit call."
                )

            logger.log(30, "Warning: No models found, skipping post_fit logic...")
            return

        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = "best"
                else:
                    logger.warning(
                        f"refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted."
                    )
                    refit_full = False
            else:
                refit_full = "all"

        if refit_full is not False:
            if infer_limit is not None:
                infer_limit = infer_limit - self._learner.preprocess_1_time
            trainer_model_best = self._trainer.get_model_best(infer_limit=infer_limit, infer_limit_as_child=True)
            logger.log(20, "Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`")
            if set_best_to_refit_full:
                _set_best_to_refit_full = trainer_model_best
            else:
                _set_best_to_refit_full = False
            if refit_full == "best":
                self.refit_full(
                    model=trainer_model_best,
                    set_best_to_refit_full=_set_best_to_refit_full,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    fit_strategy=fit_strategy,
                    **refit_full_kwargs,
                )
            else:
                self.refit_full(
                    model=refit_full,
                    set_best_to_refit_full=_set_best_to_refit_full,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    fit_strategy=fit_strategy,
                    **refit_full_kwargs,
                )

        if calibrate == "auto":
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION and self.eval_metric.needs_proba:
                calibrate = True
            elif self.problem_type == QUANTILE:
                calibrate = True
            else:
                calibrate = False
            if calibrate:
                num_rows_val_for_calibration = self._trainer.num_rows_val_for_calibration
                if self.problem_type == BINARY:
                    # Tested on "adult" dataset
                    min_val_rows_for_calibration = 3000
                elif self.problem_type == MULTICLASS:
                    # Tested on "covertype" dataset
                    min_val_rows_for_calibration = 500
                else:
                    # problem_type == "quantile"
                    # TODO: Haven't benchmarked, this is just a guess
                    min_val_rows_for_calibration = 1000
                if num_rows_val_for_calibration < min_val_rows_for_calibration:
                    calibrate = False
                    logger.log(
                        30,
                        f"Disabling calibration for metric `{self.eval_metric.name}` due to having "
                        f"fewer than {min_val_rows_for_calibration} rows of validation data for calibration, "
                        f"to avoid overfitting ({num_rows_val_for_calibration} rows). "
                        f"Force calibration via specifying `calibrate=True`. (calibrate='auto')",
                    )

        if calibrate:
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                self._trainer.calibrate_model()
            elif self.problem_type == QUANTILE:
                self._trainer.calibrate_model()
            else:
                logger.log(30, "WARNING: `calibrate=True` is only applicable to classification or quantile regression problems. Skipping calibration...")

        if isinstance(calibrate_decision_threshold, str) and calibrate_decision_threshold == "auto":
            calibrate_decision_threshold = self._can_calibrate_decision_threshold()
            if calibrate_decision_threshold and self.eval_metric.name == "precision":
                # precision becomes undefined when no true positives exist.
                # This interacts weirdly with threshold calibration where val score will be 1.0, but test score can be 0.0 due to being undefined.
                calibrate_decision_threshold = False
                logger.log(
                    30,
                    f"Disabling decision threshold calibration for metric `precision` to avoid undefined results. "
                    f"Force calibration via specifying `calibrate_decision_threshold=True`.",
                )
            elif calibrate_decision_threshold and self.eval_metric.name == "accuracy":
                num_rows_val_for_calibration = self._trainer.num_rows_val_for_calibration
                min_val_rows_for_calibration = 10000
                if num_rows_val_for_calibration < min_val_rows_for_calibration:
                    calibrate_decision_threshold = False
                    logger.log(
                        20,
                        f"Disabling decision threshold calibration for metric `accuracy` due to having "
                        f"fewer than {min_val_rows_for_calibration} rows of validation data for calibration, "
                        f"to avoid overfitting ({num_rows_val_for_calibration} rows)."
                        f"\n\t`accuracy` is generally not improved through threshold calibration. "
                        f"Force calibration via specifying `calibrate_decision_threshold=True`.",
                    )
            elif calibrate_decision_threshold:
                num_rows_val_for_calibration = self._trainer.num_rows_val_for_calibration
                min_val_rows_for_calibration = 50
                if num_rows_val_for_calibration < min_val_rows_for_calibration:
                    calibrate_decision_threshold = False
                    logger.log(
                        30,
                        f"Disabling decision threshold calibration for metric `{self.eval_metric.name}` due to having "
                        f"fewer than {min_val_rows_for_calibration} rows of validation data for calibration "
                        f"to avoid overfitting ({num_rows_val_for_calibration} rows). "
                        f"Force calibration via specifying `calibrate_decision_threshold=True`.",
                    )
            if calibrate_decision_threshold:
                logger.log(20, f"Enabling decision threshold calibration (calibrate_decision_threshold='auto', metric is valid, problem_type is 'binary')")
        if calibrate_decision_threshold:
            if self.problem_type != BINARY:
                logger.log(30, "WARNING: `calibrate_decision_threshold=True` is only applicable to binary classification. Skipping calibration...")
            else:
                best_threshold = self.calibrate_decision_threshold()
                self.set_decision_threshold(decision_threshold=best_threshold)

        if keep_only_best:
            self.delete_models(models_to_keep="best", dry_run=False)

        if save_space:
            self.save_space()

    def _can_calibrate_decision_threshold(self) -> bool:
        return self.eval_metric.needs_class and self.problem_type == BINARY

    # TODO: Consider adding infer_limit to fit_extra
    def fit_extra(
        self,
        hyperparameters: str | dict[str, Any],
        time_limit: float = None,
        base_model_names: list[str] = None,
        fit_weighted_ensemble: bool = True,
        fit_full_last_level_weighted_ensemble: bool = True,
        full_weighted_ensemble_additionally: bool = False,
        num_cpus: str | int = "auto",
        num_gpus: str | int = "auto",
        fit_strategy: Literal["auto", "sequential", "parallel"] = "auto",
        memory_limit: float | str = "auto",
        **kwargs,
    ) -> "TabularPredictor":
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
        base_model_names : list[str], default = None
            The names of the models to use as base models for this fit call.
            Base models will provide their out-of-fold predictions as additional features to the models in `hyperparameters`.
            If specified, all models trained will be stack ensembles.
            If None, models will be trained as if they were specified in :meth:`TabularPredictor.fit`, without depending on existing models.
            Only valid if bagging is enabled.
        fit_weighted_ensemble : bool, default = True
            If True, a WeightedEnsembleModel will be fit in each stack layer.
            A weighted ensemble will often be stronger than an individual model while being very fast to train.
            It is recommended to keep this value set to True to maximize predictive quality.
        fit_full_last_level_weighted_ensemble : bool, default = True
            If True, the WeightedEnsembleModel of the last stacking level will be fit with all (successful) models from all previous layers as base models.
            If stacking is disabled, settings this to True or False makes no difference because the WeightedEnsembleModel L2 always uses all models from L1.
            It is recommended to keep this value set to True to maximize predictive quality.
        full_weighted_ensemble_additionally : bool, default = False
            If True, AutoGluon will fit two WeightedEnsembleModels after training all stacking levels. Setting this to True, simulates calling
            `fit_weighted_ensemble()` after calling `fit()`. Has no affect if `fit_full_last_level_weighted_ensemble` is False and does not fit an additional
            WeightedEnsembleModel if stacking is disabled.
        num_cpus: int, default = "auto"
            The total amount of cpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
            Users generally don't need to set this value
        num_gpus: int, default = "auto"
            The total amount of gpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
            Users generally don't need to set this value
        fit_strategy: Literal["auto", "sequential", "parallel"], default = "auto"
            The strategy used to fit models.
            If "auto", uses the same fit_strategy as used in the original :meth:`TabularPredictor.fit` call.
            If "sequential", models will be fit sequentially. This is the most stable option with the most readable logging.
            If "parallel", models will be fit in parallel with ray, splitting available compute between them.
                Note: "parallel" is experimental and may run into issues. It was first added in version 1.2.0.
            For machines with 16 or more CPU cores, it is likely that "parallel" will be faster than "sequential".

            .. versionadded:: 1.2.0

        memory_limit: float | str, default = "auto"
            The total amount of memory in GB you want AutoGluon predictor to use. "auto" means AutoGluon will use all available memory on the system
            (that is detectable by psutil).
            Note that this is only a soft limit! AutoGluon uses this limit to skip training models that are expected to require too much memory or stop
            training a model that would exceed the memory limit. AutoGluon does not guarantee the enforcement of this limit (yet). Nevertheless, we expect
            AutoGluon to abide by the limit in most cases or, at most, go over the limit by a small margin.
            For most virtualized systems (e.g., in the cloud) and local usage on a server or laptop, "auto" is ideal for this parameter. We recommend manually
            setting the memory limit (and any other resources) on systems with shared resources that are controlled by the operating system (e.g., SLURM and
            cgroups). Otherwise, AutoGluon might wrongly assume more resources are available for fitting a model than the operating system allows,
            which can result in model training failing or being very inefficient.
        **kwargs :
            Refer to kwargs documentation in :meth:`TabularPredictor.fit`.
            Note that the following kwargs are not available in `fit_extra` as they cannot be changed from their values set in `fit()`:
                [`holdout_frac`, `num_bag_folds`, `auto_stack`, `feature_generator`, `unlabeled_data`]
            Moreover, `dynamic_stacking` is also not available in `fit_extra` as the detection of stacked overfitting is only supported at the first fit time.
            pseudo_data : pd.DataFrame, default = None
                Data that has been self labeled by Autogluon model and will be incorporated into training during 'fit_extra'
        """
        self._assert_is_fit("fit_extra")
        time_start = time.time()

        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_extra_kwargs(kwargs)

        verbosity = kwargs.get("verbosity", self.verbosity)
        set_logger_verbosity(verbosity)

        if verbosity >= 3:
            logger.log(20, "============ fit kwarg info ============")
            logger.log(20, "User Specified kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs_orig)}")
            logger.log(20, "Full kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs)}")
            logger.log(20, "========================================")

        self._validate_num_cpus(num_cpus=num_cpus)
        self._validate_num_gpus(num_gpus=num_gpus)
        self._validate_and_set_memory_limit(memory_limit=memory_limit)

        if fit_strategy == "auto":
            fit_strategy = self._fit_strategy
        self._validate_fit_strategy(fit_strategy=fit_strategy)

        # TODO: Allow disable aux (default to disabled)
        # TODO: num_bag_sets
        # num_bag_sets = kwargs['num_bag_sets']
        num_stack_levels = kwargs["num_stack_levels"]
        # save_bag_folds = kwargs['save_bag_folds']  # TODO: Enable

        ag_args = kwargs["ag_args"]
        ag_args_fit = kwargs["ag_args_fit"]
        ag_args_ensemble = kwargs["ag_args_ensemble"]
        excluded_model_types = kwargs["excluded_model_types"]
        pseudo_data = kwargs.get("pseudo_data", None)

        # TODO: Since data preprocessor is fitted on original train_data it cannot account for if
        # labeled pseudo data has new labels unseen in the original train. Probably need to refit
        # data preprocessor if this is the case.
        if pseudo_data is not None:
            X_pseudo, y_pseudo, y_pseudo_og = self._sanitize_pseudo_data(pseudo_data=pseudo_data)
        else:
            X_pseudo = None
            y_pseudo = None
            y_pseudo_og = None

        if ag_args is None:
            ag_args = {}
        ag_args = self._set_hyperparameter_tune_kwargs_in_ag_args(kwargs["hyperparameter_tune_kwargs"], ag_args, time_limit=time_limit)

        fit_new_weighted_ensemble = False  # TODO: Add as option
        aux_kwargs = {
            "total_resources": {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            },
        }
        if fit_weighted_ensemble is False:
            aux_kwargs = {"fit_weighted_ensemble": False}
        aux_kwargs["fit_full_last_level_weighted_ensemble"] = fit_full_last_level_weighted_ensemble
        aux_kwargs["full_weighted_ensemble_additionally"] = full_weighted_ensemble_additionally

        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        if num_stack_levels is None:
            hyperparameter_keys = list(hyperparameters.keys())
            highest_level = 1
            for key in hyperparameter_keys:
                if isinstance(key, int):
                    highest_level = max(key, highest_level)
            num_stack_levels = highest_level - 1

        # TODO: make core_kwargs a kwargs argument to predictor.fit, add aux_kwargs to predictor.fit
        core_kwargs = {
            "total_resources": {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            },
            "ag_args": ag_args,
            "ag_args_ensemble": ag_args_ensemble,
            "ag_args_fit": ag_args_fit,
            "excluded_model_types": excluded_model_types,
            "fit_strategy": fit_strategy,
        }

        # FIXME: v1.2 pseudo_data can be passed in `fit()` but it is ignored!
        if X_pseudo is not None and y_pseudo is not None:
            core_kwargs["X_pseudo"] = X_pseudo
            core_kwargs["y_pseudo"] = y_pseudo

        # TODO: Add special error message if called and training/val data was not cached.
        X, y, X_val, y_val = self._trainer.load_data()

        if y_pseudo is not None and self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            y_og = self._learner.label_cleaner.inverse_transform(y)
            y_og_classes = y_og.unique()
            y_pseudo_classes = y_pseudo_og.unique()
            matching_classes = np.in1d(y_pseudo_classes, y_og_classes)

            if not matching_classes.all():
                raise Exception(f"Pseudo training data contains classes not in original train data: {y_pseudo_classes[~matching_classes]}")

        name_suffix = kwargs.get("name_suffix", "")

        fit_models = self._trainer.train_multi_levels(
            X=X,
            y=y,
            hyperparameters=hyperparameters,
            X_val=X_val,
            y_val=y_val,
            base_model_names=base_model_names,
            time_limit=time_limit,
            relative_stack=True,
            level_end=num_stack_levels + 1,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
            name_suffix=name_suffix,
        )

        if time_limit is not None:
            time_limit = time_limit - (time.time() - time_start)

        if fit_new_weighted_ensemble:
            if time_limit is not None:
                time_limit_weighted = max(time_limit, 60)
            else:
                time_limit_weighted = None
            fit_models += self.fit_weighted_ensemble(time_limit=time_limit_weighted)

        refit_full_kwargs = dict(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
        )

        self._post_fit(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=kwargs["refit_full"],
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
            save_space=kwargs["save_space"],
            calibrate=kwargs["calibrate"],
            refit_full_kwargs=refit_full_kwargs,
        )
        self.save()
        return self

    def _get_all_fit_extra_args(self):
        ret = list(self._fit_extra_kwargs_dict().keys()) + list(inspect.signature(self.fit_extra).parameters.keys())
        ret.remove("kwargs")

        return ret

    def _fit_weighted_ensemble_pseudo(self):
        """
        Fits weighted ensemble on top models trained with pseudo labeling, then if new
        weighted ensemble model is best model then sets `model_best` in trainer to
        weighted ensemble model.
        """
        logger.log(15, "Fitting weighted ensemble using top models")
        weighted_ensemble_model_name = self.fit_weighted_ensemble()[0]

        # TODO: This is a hack! self.predict_prob does not update to use weighted ensemble
        # if it's the best model.
        # TODO: There should also be PL added to weighted ensemble model name to notify
        # users it is a model trained with PL models if they are indeed ensembled
        model_best_name = self._trainer.leaderboard().iloc[0]["model"]
        if model_best_name == weighted_ensemble_model_name:
            self._trainer.model_best = model_best_name
            self._trainer.save()
            logger.log(15, "Weighted ensemble was the best model for current iteration of pseudo labeling")
        else:
            logger.log(15, "Weighted ensemble was not the best model for current iteration of pseudo labeling")

    def _predict_pseudo(self, X_test: pd.DataFrame, use_ensemble: bool):
        if use_ensemble:
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                test_pseudo_idxes_true, y_pred_proba, y_pred = filter_ensemble_pseudo(predictor=self, unlabeled_data=X_test)
            else:
                test_pseudo_idxes_true, y_pred = filter_ensemble_pseudo(predictor=self, unlabeled_data=X_test)
                y_pred_proba = y_pred.copy()
        else:
            if self.can_predict_proba:
                y_pred_proba = self.predict_proba(data=X_test, as_multiclass=True)
                y_pred = get_pred_from_proba_df(y_pred_proba, problem_type=self.problem_type)
            else:
                y_pred = self.predict(data=X_test)
                y_pred_proba = y_pred
            test_pseudo_idxes_true = filter_pseudo(y_pred_proba_og=y_pred_proba, problem_type=self.problem_type)
        return y_pred, y_pred_proba, test_pseudo_idxes_true

    def _run_pseudolabeling(
        self,
        unlabeled_data: pd.DataFrame,
        max_iter: int,
        return_pred_prob: bool = False,
        use_ensemble: bool = False,
        fit_ensemble: bool = False,
        fit_ensemble_every_iter: bool = False,
        **kwargs,
    ):
        """
        Runs pseudolabeling algorithm using the same hyperparameters and model and fit settings
        used in original model unless specified by the user. This is an internal function that iteratively
        self labels unlabeled test data then incorporates all self labeled data above a threshold into training.
        Will keep incorporating self labeled data into training until validation score does not improve

        Parameters:
        -----------
        unlabeled_data: pd.DataFrame
            Extra unlabeled data (could be the test data) to assign pseudolabels to and incorporate as extra training data.
        max_iter: int
            Maximum allowed number of iterations, where in each iteration, the data are pseudolabeled
            by the current predictor and the predictor is refit including the pseudolabled data in its training set.
        return_pred_proba: bool, default = False
            Transductive learning setting, will return predictive probabilities of unlabeled_data
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
        previous_score = self.leaderboard(set_refit_score_to_parent=True).set_index("model", drop=True).loc[self.model_best]["score_val"]
        y_pseudo_og = pd.Series()
        X_test = unlabeled_data.copy()

        y_pred, y_pred_proba, test_pseudo_idxes_true = self._predict_pseudo(X_test=X_test, use_ensemble=use_ensemble)
        y_pred_proba_og = y_pred_proba

        for i in range(max_iter):
            if len(X_test) == 0:
                logger.log(20, f"No more unlabeled data to pseudolabel. Done with pseudolabeling...")
                break

            iter_print = str(i + 1)
            logger.log(20, f"Beginning iteration {iter_print} of pseudolabeling out of max {max_iter}")

            if len(test_pseudo_idxes_true) < 1:
                logger.log(20, f"Could not confidently assign pseudolabels for any of the provided rows in iteration {iter_print}. Done with pseudolabeling...")
                break
            else:
                logger.log(
                    20,
                    f"Pseudolabeling algorithm confidently assigned pseudolabels to {len(test_pseudo_idxes_true)} rows of data "
                    f"on iteration {iter_print}. Adding to train data",
                )

            test_pseudo_idxes = pd.Series(data=False, index=y_pred_proba.index)
            test_pseudo_idxes_false = test_pseudo_idxes[~test_pseudo_idxes.index.isin(test_pseudo_idxes_true.index)]
            test_pseudo_idxes[test_pseudo_idxes_true.index] = True

            if len(test_pseudo_idxes_true) != 0:
                if len(y_pseudo_og) == 0:
                    y_pseudo_og = y_pred.loc[test_pseudo_idxes_true.index].copy()
                else:
                    y_pseudo_og = pd.concat([y_pseudo_og, y_pred.loc[test_pseudo_idxes_true.index]], verify_integrity=True)

            pseudo_data = unlabeled_data.loc[y_pseudo_og.index]
            pseudo_data[self.label] = y_pseudo_og
            self.fit_extra(pseudo_data=pseudo_data, name_suffix=PSEUDO_MODEL_SUFFIX.format(iter=(i + 1)), **kwargs)

            if fit_ensemble and fit_ensemble_every_iter:
                self._fit_weighted_ensemble_pseudo()

            current_score = self.leaderboard(set_refit_score_to_parent=True).set_index("model", drop=True).loc[self.model_best]["score_val"]
            logger.log(
                20,
                f"Pseudolabeling algorithm changed validation score from: {previous_score}, to: {current_score}"
                f" using evaluation metric: {self.eval_metric.name}",
            )

            if previous_score >= current_score:
                # No improvement from pseudo labelling this iteration, stop iterating
                break
            else:
                # Cut down X_test to not include pseudo labeled data
                X_test = X_test.loc[test_pseudo_idxes[~test_pseudo_idxes].index]
                previous_score = current_score

                # Update y_pred_proba and test_pseudo_idxes_true based on the latest pseudolabelled iteration
                y_pred, y_pred_proba, test_pseudo_idxes_true = self._predict_pseudo(X_test=X_test, use_ensemble=use_ensemble)
                # Update the y_pred_proba_og variable if an improvement was achieved
                if return_pred_prob and test_pseudo_idxes_false is not None:
                    y_pred_proba_og.loc[test_pseudo_idxes_false.index] = y_pred_proba.loc[test_pseudo_idxes_false.index]

        if fit_ensemble and not fit_ensemble_every_iter:
            self._fit_weighted_ensemble_pseudo()
            if return_pred_prob:
                if self.can_predict_proba:
                    y_pred_proba_og = self.predict_proba(unlabeled_data)
                else:
                    y_pred_proba_og = self.predict(unlabeled_data)

        if return_pred_prob:
            return self, y_pred_proba_og
        else:
            return self

    # TODO: `fit_ensemble` and `use_ensemble` seem redundant, and don't use calibration, making them worse than when they are disabled.
    # TODO: Supporting L2+ models is very complicated. It requires predicting with the original models via `predictor.predict_proba_multi` on `pseudo_data`,
    #  then keeping track of these pred_proba and passing them to the appropriate models at fit time.
    @apply_presets(tabular_presets_dict, tabular_presets_alias)
    def fit_pseudolabel(
        self,
        pseudo_data: pd.DataFrame,
        max_iter: int = 3,
        return_pred_prob: bool = False,
        use_ensemble: bool = False,
        fit_ensemble: bool = False,
        fit_ensemble_every_iter: bool = False,
        **kwargs,
    ):
        """
        [Advanced] Uses additional data (`pseudo_data`) to try to achieve better model quality.
        Pseudo data can come either with or without the `label` column.

        If `pseudo_data` is labeled, then models will be refit using the `pseudo_data` as additional training data.
        If bagging, each fold of the bagged ensemble will use all the `pseudo_data` as additional training data.
        `pseudo_data` will never be used for validation/scoring.

        If the data is unlabeled, such as providing the batched test data without ground truth available, then transductive learning is leveraged.
        In transductive learning, the existing predictor will predict on `pseudo_data`
        to identify the most confident rows (For example all rows with predictive probability above 95%).
        These rows will then be pseudo-labelled, given the label of the most confident class.
        The pseudo-labelled rows will then be used as additional training data when fitting the models.
        Then, if `max_iter > 1`, this process can repeat itself, using the new models to predict on the unused `pseudo_data` rows
        to see if any new rows should be used in the next iteration as training data.
        We recommend specifying `return_pred_prob=True` if the data is unlabeled to get the correct prediction probabilities on the `pseudo_data`,
        rather than calling `predictor.predict_proba(pseudo_data)`.

        For example:
            Original fit: 10000 `train_data` rows with 10-fold bagging
                Bagged fold models will use 9000 `train_data` rows for training, and 1000 for validation.
            `fit_pseudolabel` is called with 5000 row labelled `pseudo_data`.
                Bagged fold models are then fit again with `_PSEUDO` suffix.
                10000 train_data rows with 10-fold bagging + 5000 `pseudo_data` rows.
                Bagged fold models will use 9000 `train_data` rows + 5000 `pseudo_data` rows = 14000 rows for training, and 1000 for validation.
                    Note: The same validation rows will be used as was done in the original fit, so that validation scores are directly comparable.
            Alternatively, `fit_pseduolabel` is called with 5000 rows unlabelled `pseudo_data`.
                Predictor predicts on the `pseudo_data`, finds 965 rows with confident predictions.
                Set the ground truth of those 965 rows as the most confident prediction.
                Bagged fold models are then fit with `_PSEUDO` suffix.
                10000 train_data rows with 10-fold bagging + 965 labelled `pseudo_data` rows.
                Bagged fold models will use 9000 `train_data` rows + 965 `pseudo_data` rows = 9965 rows for training, and 1000 for validation.
                    Note: The same validation rows will be used as was done in the original fit, so that validation scores are directly comparable.
                Repeat the process using the new pseudo-labelled predictor on the remaining `pseudo_data`.
                In the example, lets assume 188 new `pseudo_data` rows have confident predictions.
                Now the total labelled `pseudo_data` rows is 965 + 188 = 1153.
                Then repeat the process, up to `max_iter` times: ex 10000 train_data rows with 10-fold bagging + 1153 `pseudo_data` rows.
                Early stopping will trigger if validation score improvement is not observed.

        Note: pseudo_data is only used for L1 models. Support for L2+ models is not yet implemented. L2+ models will only use the original train_data.

        Parameters
        ----------
        pseudo_data : :class:`pd.DataFrame`
            Extra data to incorporate into training. Pre-labeled test data allowed. If no labels
            then pseudo-labeling algorithm will predict and filter out which rows to incorporate into training
        max_iter: int, default = 3
            Maximum iterations of pseudo-labeling allowed
        return_pred_prob: bool, default = False
            Returns held-out predictive probabilities from pseudo-labeling. If test_data is labeled then
            returns model's predictive probabilities.
        use_ensemble: bool, default = False
            If True will use ensemble pseudo labeling algorithm. If False will just use best model
            for pseudo labeling algorithm.
        fit_ensemble: bool, default = False
            If True with fit weighted ensemble model using combination of best models.
            Fitting weighted ensemble will be done after fitting has
            been completed unless otherwise specified. If False will not fit weighted ensemble
            over models trained with pseudo labeling and models trained without it.
        fit_ensemble_every_iter: bool, default = False
            If True fits weighted ensemble model for every iteration of pseudo labeling algorithm. If False
            and fit_ensemble is True will fit after all pseudo labeling training is done.
        **kwargs:
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
            raise Exception("No pseudo data given")

        self._validate_unique_indices(pseudo_data, "pseudo_data")

        was_fit = self.is_fit
        if not was_fit:
            if "train_data" not in kwargs.keys():
                Exception(
                    "Autogluon is required to be fit or given 'train_data' in order to run 'fit_pseudolabel'."
                    " Autogluon is not fit and 'train_data' was not given"
                )

            logger.log(20, f"Predictor not fit prior to pseudolabeling. Fitting now...")
            self.fit(**kwargs)

        if self.problem_type is MULTICLASS and self.eval_metric.name != "accuracy":
            logger.warning(
                "AutoGluon has detected the problem type as 'multiclass' and "
                f"eval_metric is {self.eval_metric.name}, we recommend using"
                f"fit_pseudolabeling when eval metric is 'accuracy'"
            )

        is_labeled = self.label in pseudo_data.columns

        hyperparameters = kwargs.get("hyperparameters", None)
        if hyperparameters is None:
            if self.is_fit:
                hyperparameters = self.fit_hyperparameters_
        elif isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        kwargs["hyperparameters"] = hyperparameters
        fit_extra_args = self._get_all_fit_extra_args()
        fit_extra_kwargs = {key: value for key, value in kwargs.items() if key in fit_extra_args}

        # If first fit was in this method call and `num_stack_levels` wasn't specified, reuse the number of stack levels used in the first fit.
        # TODO: Consider making calculating this information easier, such as keeping track of meta-info from the latest/original fit call.
        #  Currently we use `stack_name == core` to figure out the number of stack levels, but this is somewhat brittle.
        if "num_stack_levels" not in fit_extra_kwargs and not was_fit:
            models_core: list[str] = [m for m, stack_name in self._trainer.get_models_attribute_dict(attribute="stack_name").items() if stack_name == "core"]
            num_stack_levels = max(self._trainer.get_models_attribute_dict(attribute="level", models=models_core).values()) - 1
            fit_extra_kwargs["num_stack_levels"] = num_stack_levels
        if is_labeled:
            logger.log(20, "Fitting predictor using the provided pseudolabeled examples as extra training data...")
            self.fit_extra(pseudo_data=pseudo_data, name_suffix=PSEUDO_MODEL_SUFFIX.format(iter="")[:-1], **fit_extra_kwargs)

            if fit_ensemble:
                logger.log(15, "Fitting weighted ensemble model using best models")
                self.fit_weighted_ensemble()

            if return_pred_prob:
                y_pred_proba = self.predict_proba(pseudo_data) if self.can_predict_proba else self.predict(pseudo_data)
                return self, y_pred_proba
            else:
                return self
        else:
            logger.log(
                20,
                "Given test_data for pseudo labeling did not contain labels. "
                "AutoGluon will assign pseudo labels to data and use it for extra training data...",
            )
            return self._run_pseudolabeling(
                unlabeled_data=pseudo_data,
                max_iter=max_iter,
                return_pred_prob=return_pred_prob,
                use_ensemble=use_ensemble,
                fit_ensemble=fit_ensemble,
                fit_ensemble_every_iter=fit_ensemble_every_iter,
                **fit_extra_kwargs,
            )

    def predict(
        self,
        data: pd.DataFrame | str,
        model: str | None = None,
        as_pandas: bool = True,
        transform_features: bool = True,
        *,
        decision_threshold: float | None = None,
    ) -> pd.Series | np.ndarray:
        """
        Use trained models to produce predictions of `label` column values for new data.

        Parameters
        ----------
        data : :class:`pd.DataFrame` or str
            The data to make predictions for. Should contain same column names as training data and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`
        as_pandas : bool, default = True
            Whether to return the output as a :class:`pd.Series` (True) or :class:`np.ndarray` (False).
        transform_features : bool, default = True
            If True, preprocesses data before predicting with models.
            If False, skips global feature preprocessing.
                This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.
        decision_threshold : float, default = None
            The decision threshold used to convert prediction probabilities to predictions.
            Only relevant for binary classification, otherwise ignored.
            If None, defaults to `predictor.decision_threshold`.
            Valid values are in the range [0.0, 1.0]
            You can obtain an optimized `decision_threshold` by first calling `predictor.calibrate_decision_threshold()`.
            Useful to set for metrics such as `balanced_accuracy` and `f1` as `0.5` is often not an optimal threshold.
            Predictions are calculated via the following logic on the positive class: `1 if pred > decision_threshold else 0`

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset. Either :class:`np.ndarray` or :class:`pd.Series` depending on `as_pandas` argument.
        """
        self._assert_is_fit("predict")
        data = self._get_dataset(data)
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        return self._learner.predict(X=data, model=model, as_pandas=as_pandas, transform_features=transform_features, decision_threshold=decision_threshold)

    def predict_proba(
        self,
        data: pd.DataFrame | str,
        model: str | None = None,
        as_pandas: bool = True,
        as_multiclass: bool = True,
        transform_features: bool = True,
    ) -> pd.DataFrame | pd.Series | np.ndarray:
        """
        Use trained models to produce predicted class probabilities rather than class-labels (if task is classification).
        If `predictor.problem_type` is regression or quantile, this will raise an AssertionError.

        Parameters
        ----------
        data : :class:`pd.DataFrame` or str
            The data to make predictions for. Should contain same column names as training dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
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
        self._assert_is_fit("predict_proba")
        if not self.can_predict_proba:
            raise AssertionError(
                f'`predictor.predict_proba` is not supported when problem_type="{self.problem_type}". '
                f"Please call `predictor.predict` instead. "
                f"You can check the value of `predictor.can_predict_proba` to tell if predict_proba is valid."
            )
        data = self._get_dataset(data)
        return self._learner.predict_proba(X=data, model=model, as_pandas=as_pandas, as_multiclass=as_multiclass, transform_features=transform_features)

    def predict_from_proba(self, y_pred_proba: pd.DataFrame | np.ndarray, decision_threshold: float | None = None) -> pd.Series | np.array:
        """
        Given prediction probabilities, convert to predictions.

        Parameters
        ----------
        y_pred_proba : :class:`pd.DataFrame` or :class:`np.ndarray`
            The prediction probabilities to convert to predictions.
            Obtainable via the output of `predictor.predict_proba`.
        decision_threshold : float, default = None
            The decision threshold used to convert prediction probabilities to predictions.
            Only relevant for binary classification, otherwise ignored.
            If None, defaults to `predictor.decision_threshold`.
            Valid values are in the range [0.0, 1.0]
            You can obtain an optimized `decision_threshold` by first calling `predictor.calibrate_decision_threshold()`.
            Useful to set for metrics such as `balanced_accuracy` and `f1` as `0.5` is often not an optimal threshold.
            Predictions are calculated via the following logic on the positive class: `1 if pred > decision_threshold else 0`

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset. Either :class:`np.ndarray` or :class:`pd.Series` depending on `y_pred_proba` dtype.

        Examples
        --------
        >>> from autogluon.tabular import TabularPredictor
        >>> predictor = TabularPredictor(label='class').fit('train.csv', label='class')
        >>> y_pred_proba = predictor.predict_proba('test.csv')
        >>>
        >>> # y_pred and y_pred_from_proba are identical
        >>> y_pred = predictor.predict('test.csv')
        >>> y_pred_from_proba = predictor.predict_from_proba(y_pred_proba=y_pred_proba)
        """
        if not self.can_predict_proba:
            raise AssertionError(f'`predictor.predict_from_proba` is not supported when problem_type="{self.problem_type}".')
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        return self._learner.get_pred_from_proba(y_pred_proba=y_pred_proba, decision_threshold=decision_threshold)

    @property
    def can_predict_proba(self) -> bool:
        """
        Return True if predictor can return prediction probabilities via `.predict_proba`, otherwise return False.
        Raises an AssertionError if called before fitting.
        """
        self._assert_is_fit("can_predict_proba")
        return problem_type_info.can_predict_proba(problem_type=self.problem_type)

    @property
    def is_fit(self) -> bool:
        """
        Return True if `predictor.fit` has been called, otherwise return False.
        """
        return self._learner.is_fit

    def evaluate(
        self,
        data: pd.DataFrame | str,
        model: str = None,
        decision_threshold: float = None,
        display: bool = False,
        auxiliary_metrics: bool = True,
        detailed_report: bool = False,
        **kwargs,
    ) -> dict:
        """
        Report the predictive performance evaluated over a given dataset.
        This is basically a shortcut for: `pred_proba = predict_proba(data); evaluate_predictions(data[label], pred_proba)`.

        Parameters
        ----------
        data : str or :class:`pd.DataFrame`
            This dataset must also contain the `label` with the same column-name as previously specified.
            If str is passed, `data` will be loaded using the str value as the file path.
            If `self.sample_weight` is set and `self.weight_evaluation==True`, then a column with the sample weight name is checked and used for weighted metric evaluation if it exists.
        model : str (optional)
            The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
        decision_threshold : float, default = None
            The decision threshold to use when converting prediction probabilities to predictions.
            This will impact the scores of metrics such as `f1` and `accuracy`.
            If None, defaults to `predictor.decision_threshold`. Ignored unless `problem_type='binary'`.
            Refer to the `predictor.decision_threshold` docstring for more information.
        display : bool, default = False
            If True, performance results are printed.
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
        if "silent" in kwargs:
            # keep `silent` logic for backwards compatibility
            assert isinstance(kwargs["silent"], bool)
            display = not kwargs.pop("silent")
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TabularPredictor.evaluate() got an unexpected keyword argument '{key}'")
        self._assert_is_fit("evaluate")
        data = self._get_dataset(data)
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        if self.can_predict_proba:
            y_pred = self.predict_proba(data=data, model=model)
        else:
            y_pred = self.predict(data=data, model=model)
        if self.sample_weight is not None and self.weight_evaluation and self.sample_weight in data:
            sample_weight = data[self.sample_weight]
        else:
            sample_weight = None
        return self.evaluate_predictions(
            y_true=data[self.label],
            y_pred=y_pred,
            sample_weight=sample_weight,
            decision_threshold=decision_threshold,
            display=display,
            auxiliary_metrics=auxiliary_metrics,
            detailed_report=detailed_report,
        )

    def evaluate_predictions(
        self, y_true, y_pred, sample_weight=None, decision_threshold=None, display: bool = False, auxiliary_metrics=True, detailed_report=False, **kwargs
    ) -> dict:
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
        decision_threshold : float, default = None
            The decision threshold to use when converting prediction probabilities to predictions.
            This will impact the scores of metrics such as `f1` and `accuracy`.
            If None, defaults to `predictor.decision_threshold`. Ignored unless `problem_type='binary'`.
            Refer to the `predictor.decision_threshold` docstring for more information.
        display : bool, default = False
            If True, performance results are printed.
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
        if "silent" in kwargs:
            # keep `silent` logic for backwards compatibility
            assert isinstance(kwargs["silent"], bool)
            display = not kwargs.pop("silent")
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TabularPredictor.evaluate_predictions() got an unexpected keyword argument '{key}'")
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        return self._learner.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            decision_threshold=decision_threshold,
            display=display,
            auxiliary_metrics=auxiliary_metrics,
            detailed_report=detailed_report,
        )

    def leaderboard(
        self,
        data: pd.DataFrame | str | None = None,
        extra_info: bool = False,
        extra_metrics: list | None = None,
        decision_threshold: float | None = None,
        score_format: str = "score",
        only_pareto_frontier: bool = False,
        skip_score: bool = False,
        refit_full: bool | None = None,
        set_refit_score_to_parent: bool = False,
        display: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Output summary of information about models produced during `fit()` as a :class:`pd.DataFrame`.
        Includes information on test and validation scores for all models, model training times, inference times, and stack levels.
        Output DataFrame columns include:
            'model': The name of the model.

            'score_val': The validation score of the model on the 'eval_metric'.
                NOTE: Metrics scores always show in higher is better form.
                This means that metrics such as log_loss and root_mean_squared_error will have their signs FLIPPED, and values will be negative.
                This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.
            'eval_metric': The evaluation metric name used to calculate the scores.
                This should be identical to `predictor.eval_metric.name`.
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
        data : str or :class:`pd.DataFrame` (optional)
            This dataset must also contain the label-column with the same column-name as specified during fit().
            If extra_metrics=None and skip_score=True, then the label column is not required.
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
        decision_threshold : float, default = None
            The decision threshold to use when converting prediction probabilities to predictions.
            This will impact the scores of metrics such as `f1` and `accuracy`.
            If None, defaults to `predictor.decision_threshold`. Ignored unless `problem_type='binary'`.
            Refer to the `predictor.decision_threshold` docstring for more information.
            NOTE: `score_val` will not be impacted by this value in v0.8.
                `score_val` will always show the validation scores achieved with a decision threshold of `0.5`.
                Only test scores will be properly updated.
        score_format : {'score', 'error'}
            If "score", leaderboard is returned as normal.
            If "error", the column "score_val" is converted to "metric_error_val", and "score_test" is converted to "metric_error_test".
                "metric_error" is calculated by taking `predictor.eval_metric.convert_score_to_error(score)`.
                This will result in errors where 0 is perfect and lower is better.
        only_pareto_frontier : bool, default = False
            If `True`, only return model information of models in the Pareto frontier of the accuracy/latency trade-off (models which achieve the highest score within their end-to-end inference time).
            At minimum this will include the model with the highest score and the model with the lowest inference time.
            This is useful when deciding which model to use during inference if inference time is a consideration.
            Models filtered out by this process would never be optimal choices for a user that only cares about model inference time and score.
        skip_score : bool, default = False
            [Advanced, primarily for developers]
            If `True`, will skip computing `score_test` if `data` is specified. `score_test` will be set to NaN for all models.
            `pred_time_test` and related columns will still be computed.
        refit_full : bool, default = None
            If True, will return only models that have been refit (ex: have `_FULL` in the name).
            If False, will return only models that have not been refit.
            If None, will return all models.
        set_refit_score_to_parent : bool, default = False
            If True, the `score_val` of refit models will be set to the `score_val` of their parent.
            While this does not represent the genuine validation score of the refit model, it is a reasonable proxy.
        display : bool, default = False
            If True, the output DataFrame is printed to stdout.

        Returns
        -------
        :class:`pd.DataFrame` of model performance summary information.
        """
        if "silent" in kwargs:
            # keep `silent` logic for backwards compatibility
            assert isinstance(kwargs["silent"], bool)
            display = not kwargs.pop("silent")
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TabularPredictor.leaderboard() got an unexpected keyword argument '{key}'")
        self._assert_is_fit("leaderboard")
        data = self._get_dataset(data, allow_nan=True)
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        return self._learner.leaderboard(
            X=data,
            extra_info=extra_info,
            extra_metrics=extra_metrics,
            decision_threshold=decision_threshold,
            only_pareto_frontier=only_pareto_frontier,
            score_format=score_format,
            skip_score=skip_score,
            refit_full=refit_full,
            set_refit_score_to_parent=set_refit_score_to_parent,
            display=display,
        )

    def learning_curves(self) -> tuple[dict, dict]:
        """
        Retrieves learning curves generated during predictor.fit().
        Will not work if the learning_curves flag was not set during training.
        Note that learning curves are only generated for iterative learners with
        learning curve support.

        Parameters
        ----------
        None

        Returns
        -------
        metadata: dict
            A dictionary containing metadata related to the training process.

        model_data: dict
            A dictionary containing the learning curves across all models.
            To see curve_data format, refer to AbstractModel's save_learning_curves() method.
                {
                    "model": curve_data,
                    "model": curve_data,
                    "model": curve_data,
                    "model": curve_data,
                }
        """
        metadata = self.info()
        metadata = {
            "problem_type": metadata["problem_type"],
            "eval_metric": metadata["eval_metric"],
            "num_classes": metadata["num_classes"],
            "num_rows_train": metadata["num_rows_train"],
            "num_rows_val": metadata["num_rows_val"],
            "num_rows_test": metadata["num_rows_test"],
            "models": {
                model: {
                    "model_name": info["name"],
                    "model_type": info["model_type"],
                    "stopping_metric": info["stopping_metric"],
                    "hyperparameters": info["hyperparameters"],
                    "hyperparameters_fit": info["hyperparameters_fit"],
                    "ag_args_fit": info["ag_args_fit"],
                    "predict_time": info["predict_time"],
                    "fit_time": info["fit_time"],
                    "val_score": info["val_score"],
                }
                for model, info in metadata["model_info"].items()
                if info.get("has_learning_curves", False)
            },
        }

        model_data = {}
        for model in metadata["models"].values():
            model_name = model["model_name"]
            model_data[model_name] = self._trainer.get_model_learning_curves(model=model_name)

        return metadata, model_data

    def model_failures(self, verbose: bool = False) -> pd.DataFrame:
        """
        [Advanced] Get the model failures that occurred during the fitting of this model, in the form of a pandas DataFrame.

        This is useful for in-depth debugging of model failures and identifying bugs.

        For more information on model failures, refer to `predictor.info()['model_info_failures']`

        Parameters
        ----------
        verbose: bool, default = False
            If True, the output DataFrame is printed to stdout.

        Returns
        -------
        model_failures_df: pd.DataFrame
            A DataFrame of model failures. Each row corresponds to a model failure, and columns correspond to meta information about that model.

            Included Columns:
                "model": The name of the model that failed
                "exc_type": The class name of the exception raised
                "total_time": The total time in seconds taken by the model prior to the exception (lost time due to the failure)
                "model_type": The class name of the model
                "child_model_type": The child class name of the model
                "is_initialized"
                "is_fit"
                "is_valid"
                "can_infer"
                "num_features"
                "num_models"
                "memory_size"
                "hyperparameters"
                "hyperparameters_fit"
                "child_hyperparameters"
                "child_hyperparameters_fit"
                "exc_str": The string message contained in the raised exception
                "exc_traceback": The full traceback message of the exception as a string
                "exc_order": The order of the model failure (starting from 1)
        """
        self._assert_is_fit("model_failures")
        model_failures_df = self._trainer.model_failures()
        if verbose:
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                print(model_failures_df)
        return model_failures_df

    def predict_proba_multi(
        self,
        data: pd.DataFrame = None,
        models: list[str] = None,
        as_pandas: bool = True,
        as_multiclass: bool = True,
        transform_features: bool = True,
        inverse_transform: bool = True,
    ) -> dict[str, pd.DataFrame] | dict[str, pd.Series] | dict[str, np.ndarray]:
        """
        Returns a dictionary of prediction probabilities where the key is
        the model name and the value is the model's prediction probabilities on the data.

        Equivalent output to:
        ```
        predict_proba_dict = {}
        for m in models:
            predict_proba_dict[m] = predictor.predict_proba(data, model=m)
        ```

        Note that this will generally be much faster than calling :meth:`TabularPredictor.predict_proba` separately for each model
        because this method leverages the model dependency graph to avoid redundant computation.

        Parameters
        ----------
        data : str or DataFrame, default = None
            The data to predict on.
            If None:
                If self.has_val, the validation data is used.
                Else, prediction is skipped and the out-of-fold (OOF) prediction probabilities are returned, equivalent to:
                ```
                predict_proba_dict = {}
                for m in models:
                    predict_proba_dict[m] = predictor.predict_proba_oof(model=m)
                ```
        models : list[str], default = None
            The list of models to get predictions for.
            If None, all models that can infer are used.
        as_pandas : bool, default = True
            Whether to return the output of each model as a pandas object (True) or numpy array (False).
            Pandas object is a :class:`pd.DataFrame` if this is a multiclass problem or `as_multiclass=True`, otherwise it is a :class:`pd.Series`.
            If the output is a :class:`pd.DataFrame`, the column order will be equivalent to `predictor.classes_`.
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
        inverse_transform : bool, default = True
            If True, will return prediction probabilities in the original format.
            If False (advanced), will return prediction probabilities in AutoGluon's internal format.

        Returns
        -------
        dict
            Dictionary with model names as keys and model prediction probabilities as values.
        """
        self._assert_is_fit("predict_proba_multi")
        if not self.can_predict_proba:
            raise AssertionError(
                f'`predictor.predict_proba_multi` is not supported when problem_type="{self.problem_type}". '
                f"Please call `predictor.predict_multi` instead. "
                f"You can check the value of `predictor.can_predict_proba` to tell if predict_proba_multi is valid."
            )
        data = self._get_dataset(data, allow_nan=True)
        return self._learner.predict_proba_multi(
            X=data,
            models=models,
            as_pandas=as_pandas,
            as_multiclass=as_multiclass,
            transform_features=transform_features,
            inverse_transform=inverse_transform,
            use_refit_parent_oof=True,
        )

    @overload
    def predict_multi(
        self,
        data: pd.DataFrame = None,
        models: list[str] = None,
        as_pandas: Literal[True] = True,
        transform_features: bool = True,
        inverse_transform: bool = True,
        decision_threshold: float = None,
    ) -> dict[str, pd.Series]: ...

    @overload
    def predict_multi(
        self,
        data: pd.DataFrame = None,
        models: list[str] = None,
        *,
        as_pandas: Literal[False],
        transform_features: bool = True,
        inverse_transform: bool = True,
        decision_threshold: float = None,
    ) -> dict[str, np.ndarray]: ...

    def predict_multi(
        self,
        data: pd.DataFrame = None,
        models: list[str] = None,
        as_pandas: bool = True,
        transform_features: bool = True,
        inverse_transform: bool = True,
        *,
        decision_threshold: float = None,
    ) -> dict[str, pd.Series] | dict[str, np.ndarray]:
        """
        Returns a dictionary of predictions where the key is
        the model name and the value is the model's prediction probabilities on the data.

        Equivalent output to:
        ```
        predict_dict = {}
        for m in models:
            predict_dict[m] = predictor.predict(data, model=m)
        ```

        Note that this will generally be much faster than calling :meth:`TabularPredictor.predict` separately for each model
        because this method leverages the model dependency graph to avoid redundant computation.

        Parameters
        ----------
        data : DataFrame, default = None
            The data to predict on.
            If None:
                If self.has_val, the validation data is used.
                Else, prediction is skipped and the out-of-fold (OOF) predictions are returned, equivalent to:
                ```
                predict_dict = {}
                for m in models:
                    predict_dict[m] = predictor.predict_oof(model=m)
                ```
        models : list[str], default = None
            The list of models to get predictions for.
            If None, all models that can infer are used.
        as_pandas : bool, default = True
            Whether to return the output of each model as a :class:`pd.Series` (True) or :class:`np.ndarray` (False).
        transform_features : bool, default = True
            If True, preprocesses data before predicting with models.
            If False, skips global feature preprocessing.
                This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.
        inverse_transform : bool, default = True
            If True, will return predictions in the original format.
            If False (advanced), will return predictions in AutoGluon's internal format.
        decision_threshold : float, default = None
            The decision threshold used to convert prediction probabilities to predictions.
            Only relevant for binary classification, otherwise ignored.
            If None, defaults to `0.5`.
            Valid values are in the range [0.0, 1.0]
            You can obtain an optimized `decision_threshold` by first calling :meth:`TabularPredictor.calibrate_decision_threshold`.
            Useful to set for metrics such as `balanced_accuracy` and `f1` as `0.5` is often not an optimal threshold.
            Predictions are calculated via the following logic on the positive class: `1 if pred > decision_threshold else 0`

        Returns
        -------
        dict[str, pd.Series] | dict[str, np.ndarray]
            Dictionary with model names as keys and model predictions as values.
        """
        self._assert_is_fit("predict_multi")
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        data = self._get_dataset(data, allow_nan=True)
        return self._learner.predict_multi(
            X=data,
            models=models,
            as_pandas=as_pandas,
            transform_features=transform_features,
            inverse_transform=inverse_transform,
            decision_threshold=decision_threshold,
        )

    def fit_summary(self, verbosity: int = 3, show_plot: bool = False) -> dict:
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
        self._assert_is_fit("fit_summary")
        # hpo_used = len(self._trainer.hpo_results) > 0
        hpo_used = False  # Disabled until a more memory efficient hpo_results object is implemented.
        model_types = self._trainer.get_models_attribute_dict(attribute="type")
        model_inner_types = self._trainer.get_models_attribute_dict(attribute="type_inner")
        model_typenames = {key: model_types[key].__name__ for key in model_types}
        model_innertypenames = {key: model_inner_types[key].__name__ for key in model_types if key in model_inner_types}
        MODEL_STR = "Model"
        ENSEMBLE_STR = "Ensemble"
        for model in model_typenames:
            if (model in model_innertypenames) and (ENSEMBLE_STR not in model_innertypenames[model]) and (ENSEMBLE_STR in model_typenames[model]):
                new_model_typename = model_typenames[model] + "_" + model_innertypenames[model]
                if new_model_typename.endswith(MODEL_STR):
                    new_model_typename = new_model_typename[: -len(MODEL_STR)]
                model_typenames[model] = new_model_typename

        unique_model_types = set(model_typenames.values())  # no more class info
        # all fit() information that is returned:
        results = {
            "model_types": model_typenames,  # dict with key = model-name, value = type of model (class-name)
            "model_performance": self._trainer.get_models_attribute_dict("val_score"),
            # dict with key = model-name, value = validation performance
            "model_best": self._trainer.model_best,  # the name of the best model (on validation data)
            "model_paths": self._trainer.get_models_attribute_dict("path"),
            # dict with key = model-name, value = path to model file
            "model_fit_times": self._trainer.get_models_attribute_dict("fit_time"),
            "model_pred_times": self._trainer.get_models_attribute_dict("predict_time"),
            "num_bag_folds": self._trainer.k_fold,
            "max_stack_level": self._trainer.get_max_level(),
        }
        if self.problem_type == QUANTILE:
            results["num_quantiles"] = len(self.quantile_levels)
        elif self.problem_type != REGRESSION:
            results["num_classes"] = self._trainer.num_classes
        # if hpo_used:
        #     results['hpo_results'] = self._trainer.hpo_results
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in self._trainer.get_model_names():
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params
        results["model_hyperparams"] = model_hyperparams

        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            results["leaderboard"] = self._learner.leaderboard(display=True)
            # self._summarize('model_performance', 'Validation performance of individual models', results)
            #  self._summarize('model_best', 'Best model (based on validation performance)', results)
            # self._summarize('hyperparameter_tune', 'Hyperparameter-tuning used', results)
            print("Number of models trained: %s" % len(results["model_performance"]))
            print("Types of models trained:")
            print(unique_model_types)
            num_fold_str = ""
            bagging_used = results["num_bag_folds"] > 0
            if bagging_used:
                num_fold_str = f" (with {results['num_bag_folds']} folds)"
            print("Bagging used: %s %s" % (bagging_used, num_fold_str))
            num_stack_str = ""
            stacking_used = results["max_stack_level"] > 2
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
            plot_tabular_models(
                results, output_directory=self.path, save_file="SummaryOfModels.html", plot_title="Models produced during fit()", show_plot=show_plot
            )
            if hpo_used:
                for model_type in results["hpo_results"]:
                    if "trial_info" in results["hpo_results"][model_type]:
                        plot_summary_of_models(
                            results["hpo_results"][model_type],
                            output_directory=self.path,
                            save_file=model_type + "_HPOmodelsummary.html",
                            plot_title=f"Models produced during {model_type} HPO",
                            show_plot=show_plot,
                        )
                        plot_performance_vs_trials(
                            results["hpo_results"][model_type],
                            output_directory=self.path,
                            save_file=model_type + "_HPOperformanceVStrials.png",
                            plot_title=f"HPO trials for {model_type} models",
                            show_plot=show_plot,
                        )
        if verbosity > 2:  # print detailed information
            if hpo_used:
                hpo_results = results["hpo_results"]
                print("*** Details of Hyperparameter optimization ***")
                for model_type in hpo_results:
                    hpo_model = hpo_results[model_type]
                    if "trial_info" in hpo_model:
                        print(
                            f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s, Search strategy = {hpo_model['search_strategy']}"
                        )
                        print(f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):")
                        print(hpo_model["best_config"])
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

    def transform_features(
        self,
        data: pd.DataFrame | str = None,
        model: str = None,
        base_models: list[str] = None,
        return_original_features: bool = True,
    ) -> pd.DataFrame:
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
        data: :class:`pd.DataFrame` or str (optional)
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
        model: str, default = None
            Model to generate input features for.
            The output data will be equivalent to the input data that would be sent into `model.predict_proba(data)`.
                Note: This only applies to cases where `data` is not the training data.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
            Specifying a `refit_full` model will cause an exception if `data=None`.
            `base_models=None` is a requirement when specifying `model`.
        base_models: list[str], default = None
            List of model names to use as base_models for a hypothetical stacker model when generating input features.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
            If a stacker model S exists with `base_models=M`, then setting `base_models=M` is equivalent to setting `model=S`.
            `model=None` is a requirement when specifying `base_models`.
        return_original_features: bool, default = True
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
        self._assert_is_fit("transform_features")
        data = self._get_dataset(data, allow_nan=True)
        return self._learner.get_inputs_to_stacker(dataset=data, model=model, base_models=base_models, use_orig_features=return_original_features)

    def transform_labels(self, labels: np.ndarray | pd.Series, inverse: bool = False, proba: bool = False) -> pd.Series | pd.DataFrame:
        """
        Transforms data labels to the internal label representation.
        This can be useful for training your own models on the same data label representation as AutoGluon.
        Regression problems do not differ between original and internal representation, and thus this method will return the provided labels.
        Warning: When `inverse=False`, it is possible for the output to contain NaN label values in multiclass problems if the provided label was dropped during training.

        Parameters
        ----------
        labels: :class:`np.ndarray` or :class:`pd.Series`
            Labels to transform.
            If `proba=False`, an example input would be the output of `predictor.predict(test_data)`.
            If `proba=True`, an example input would be the output of `predictor.predict_proba(test_data, as_multiclass=False)`.
        inverse: bool, default = False
            When `True`, the input labels are treated as being in the internal representation and the original representation is outputted.
        proba: bool, default = False
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
        self._assert_is_fit("transform_labels")
        return self._learner.transform_labels(y=labels, inverse=inverse, proba=proba)

    def feature_importance(
        self,
        data=None,
        model: str = None,
        features: list = None,
        feature_stage: str = "original",
        subsample_size: int = 5000,
        time_limit: float = None,
        num_shuffle_sets: int = None,
        include_confidence_band: bool = True,
        confidence_level: float = 0.99,
        silent: bool = False,
    ):
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
        data : str or :class:`pd.DataFrame` (optional)
            This data must also contain the label-column with the same column-name as specified during `fit()`.
            If specified, then the data is used to calculate the feature importance scores.
            If str is passed, `data` will be loaded using the str value as the file path.
            If not specified, the original data used during `fit()` will be used if `cache_data=True`. Otherwise, an exception will be raised.
            Do not pass the training data through this argument, as the feature importance scores calculated will be biased due to overfitting.
                More accurate feature importances will be obtained from new data that was held-out during `fit()`.
        model : str, default = None
            Model to get feature importances for, if None the best model is chosen.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`
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
        self._assert_is_fit("feature_importance")
        data = self._get_dataset(data, allow_nan=True)
        if (data is None) and (not self._trainer.is_data_saved):
            raise AssertionError(
                "No data was provided and there is no cached data to load for feature importance calculation. `cache_data=True` must be set in the `TabularPredictor` init `learner_kwargs` argument call to enable this functionality when data is not specified."
            )
        if data is not None:
            self._validate_unique_indices(data, "data")

        if num_shuffle_sets is None:
            num_shuffle_sets = 10 if time_limit else 5

        fi_df = self._learner.get_feature_importance(
            model=model,
            X=data,
            features=features,
            feature_stage=feature_stage,
            subsample_size=subsample_size,
            time_limit=time_limit,
            num_shuffle_sets=num_shuffle_sets,
            silent=silent,
        )

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
                mean = fi["importance"]
                stddev = fi["stddev"]
                n = fi["n"]
                if stddev == np.nan or n == np.nan or mean == np.nan or n == 1:
                    ci_high = np.nan
                    ci_low = np.nan
                else:
                    t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                    ci_high = mean + t_val * stddev / math.sqrt(n)
                    ci_low = mean - t_val * stddev / math.sqrt(n)
                ci_high_dict[fi.name] = ci_high
                ci_low_dict[fi.name] = ci_low
            high_str = "p" + ci_str + "_high"
            low_str = "p" + ci_str + "_low"
            fi_df[high_str] = pd.Series(ci_high_dict)
            fi_df[low_str] = pd.Series(ci_low_dict)
        return fi_df

    def compile(self, models="best", with_ancestors=True, compiler_configs="auto"):
        """
        Compile models for accelerated prediction.
        This can be helpful to reduce prediction latency and improve throughput.

        Note that this is currently an experimental feature, the supported compilers can be ['native', 'onnx'].

        In order to compile with a specific compiler, that compiler must be installed in the Python environment.

        Parameters
        ----------
        models : list of str or str, default = 'best'
            Model names of models to compile.
            If 'best' then the model with the highest validation score is compiled (this is the model used for prediction by default).
            If 'all' then all models are compiled.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
        with_ancestors : bool, default = True
            If True, all ancestor models of the provided models will also be compiled.
        compiler_configs : dict or str, default = "auto"
            If "auto", defaults to the following:
                compiler_configs = {
                    "RF": {"compiler": "onnx"},
                    "XT": {"compiler": "onnx"},
                    "NN_TORCH": {"compiler": "onnx"},
                }
            Otherwise, specify a compiler_configs dictionary manually. Keys can be exact model names or model types.
            Exact model names take priority over types if both are valid for a model.
            Types can be either the true type such as RandomForestModel or the shorthand "RF".
            The dictionary key logic for types is identical to the logic in the hyperparameters argument of `predictor.fit`

            Example values within the configs:
                compiler : str, default = None
                    The compiler that is used for model compilation.
                batch_size : int, default = None
                    The batch size that is optimized for model prediction.
                    By default, the batch size is None. This means the compiler would try to leverage dynamic shape for prediction.
                    Using batch_size=1 would be more suitable for online prediction, which expects a result from one data point.
                    However, it can be slow for batch processing, because of the overhead of multiple kernel execution.
                    Increasing batch size to a number that is larger than 1 would help increase the prediction throughput.
                    This comes with an expense of utilizing larger memory for prediction.
        """
        self._assert_is_fit("compile")
        if isinstance(compiler_configs, str):
            if compiler_configs == "auto":
                compiler_configs = {
                    "RF": {"compiler": "onnx"},
                    "XT": {"compiler": "onnx"},
                    "NN_TORCH": {"compiler": "onnx"},
                }
            else:
                raise ValueError(f'Unknown compiler_configs preset: "{compiler_configs}"')
        self._trainer.compile(model_names=models, with_ancestors=with_ancestors, compiler_configs=compiler_configs)

    def persist(self, models="best", with_ancestors=True, max_memory=0.4) -> list[str]:
        """
        Persist models in memory for reduced inference latency. This is particularly important if the models are being used for online-inference where low latency is critical.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.

        Parameters
        ----------
        models : list of str or str, default = 'best'
            Model names of models to persist.
            If 'best' then the model with the highest validation score is persisted (this is the model used for prediction by default).
            If 'all' then all models are persisted.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
        with_ancestors : bool, default = True
            If True, all ancestor models of the provided models will also be persisted.
            If False, stacker models will not have the models they depend on persisted unless those models were specified in `models`. This will slow down inference as the ancestor models will still need to be loaded from disk for each predict call.
            Only relevant for stacker models.
        max_memory : float, default = 0.4
            Proportion of total available memory to allow for the persisted models to use.
            If the models' summed memory usage requires a larger proportion of memory than max_memory, they are not persisted. In this case, the output will be an empty list.
            If None, then models are persisted regardless of estimated memory usage. This can cause out-of-memory errors.

        Returns
        -------
        List of persisted model names.
        """
        self._assert_is_fit("persist")
        try:
            return self._learner.persist_trainer(low_memory=False, models=models, with_ancestors=with_ancestors, max_memory=max_memory)
        except Exception as e:
            valid_models = self.model_names()
            if isinstance(models, list):
                invalid_models = [m for m in models if m not in valid_models]
                if invalid_models:
                    raise ValueError(f"Invalid models specified. The following models do not exist:\n\t{invalid_models}\nValid models:\n\t{valid_models}")
            raise e

    def unpersist(self, models="all") -> list[str]:
        """
        Unpersist models in memory for reduced memory usage.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.
        Note: Another way to reset the predictor and unpersist models is to reload the predictor from disk via `predictor = TabularPredictor.load(predictor.path)`.

        Parameters
        ----------
        models : list of str or str, default = 'all'
            Model names of models to unpersist.
            If 'all' then all models are unpersisted.
            Valid models are listed in this `predictor` by calling `predictor.model_names(persisted=True)`.

        Returns
        -------
        List of unpersisted model names.
        """
        self._assert_is_fit("unpersist")
        return self._learner.load_trainer().unpersist(model_names=models)

    # TODO: `total_resources = None` during refit, fix this.
    #  refit_full doesn't account for user-specified resources at fit time, nor does it allow them to specify for refit.
    def refit_full(
        self,
        model: str | list[str] = "all",
        set_best_to_refit_full: bool = True,
        train_data_extra: pd.DataFrame = None,
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
        fit_strategy: Literal["auto", "sequential", "parallel"] = "auto",
        **kwargs,
    ) -> dict[str, str]:
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
        model : str | list[str], default = 'all'
            Model name of model(s) to refit.
                If 'all' then all models are refitted.
                If 'best' then the model with the highest validation score is refit.
            All ancestor models will also be refit in the case that the selected model is a weighted or stacker ensemble.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`.
        set_best_to_refit_full : bool | str, default = True
            If True, sets best model to the refit_full version of the prior best model.
            This means the model used when `predictor.predict(data)` is called will be the refit_full version instead of the original version of the model.
            Ignored if `model` is not the best model.
            If str, interprets as a model name and sets best model to the refit_full version of the model `set_best_to_refit_full`.
        train_data_extra : pd.DataFrame, default = None
            If specified, will be used as additional rows of training data when refitting models.
            Requires label column. Will only be used for L1 models.
        num_cpus: int | str, default = "auto"
            The total amount of cpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
            Users generally don't need to set this value
        num_gpus: int | str, default = "auto"
            The total amount of gpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
            Users generally don't need to set this value
        fit_strategy: Literal["auto", "sequential", "parallel"], default = "auto"
            The strategy used to fit models.
            If "auto", uses the same fit_strategy as used in the original :meth:`TabularPredictor.fit` call.
            If "sequential", models will be fit sequentially. This is the most stable option with the most readable logging.
            If "parallel", models will be fit in parallel with ray, splitting available compute between them.
                Note: "parallel" is experimental and may run into issues. It was first added in version 1.2.0.
            For machines with 16 or more CPU cores, it is likely that "parallel" will be faster than "sequential".

            .. versionadded:: 1.2.0

        **kwargs
            [Advanced] Developer debugging arguments.

        Returns
        -------
        Dictionary of original model names -> refit_full model names.
        """
        self._assert_is_fit("refit_full")
        ts = time.time()
        model_best = self._model_best(can_infer=None)
        if model == "best":
            model = model_best
        logger.log(
            20,
            "Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...\n"
            '\tModels trained in this way will have the suffix "_FULL" and have NaN validation score.\n'
            "\tThis process is not bound by time_limit, but should take less time than the original `predictor.fit` call.\n"
            '\tTo learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.',
        )

        self._validate_num_cpus(num_cpus=num_cpus)
        self._validate_num_gpus(num_gpus=num_gpus)
        total_resources = {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        }

        if fit_strategy == "auto":
            fit_strategy = self._fit_strategy
        self._validate_fit_strategy(fit_strategy=fit_strategy)

        if train_data_extra is not None:
            assert kwargs.get("X_pseudo", None) is None, f"Cannot pass both train_data_extra and X_pseudo arguments"
            assert kwargs.get("y_pseudo", None) is None, f"Cannot pass both train_data_extra and y_pseudo arguments"
            X_pseudo, y_pseudo, _ = self._sanitize_pseudo_data(pseudo_data=train_data_extra, name="train_data_extra")
            kwargs["X_pseudo"] = X_pseudo
            kwargs["y_pseudo"] = y_pseudo
        refit_full_dict = self._learner.refit_ensemble_full(model=model, total_resources=total_resources, fit_strategy=fit_strategy, **kwargs)

        if set_best_to_refit_full:
            if isinstance(set_best_to_refit_full, str):
                model_to_set_best = set_best_to_refit_full
            else:
                model_to_set_best = model_best
            model_refit_map = self._trainer.model_refit_map()
            if model_to_set_best in model_refit_map:
                self._trainer.model_best = model_refit_map[model_to_set_best]
                # Note: model_best will be overwritten if additional training is done with new models,
                # since model_best will have validation score of None and any new model will have a better validation score.
                # This has the side effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                self._trainer.save()
                logger.log(
                    20,
                    f'Updated best model to "{self._trainer.model_best}" (Previously "{model_best}"). '
                    f'AutoGluon will default to using "{self._trainer.model_best}" for predict() and predict_proba().',
                )
            elif model_to_set_best in model_refit_map.values():
                # Model best is already a refit full model
                prev_best = self._trainer.model_best
                self._trainer.model_best = model_to_set_best
                self._trainer.save()
                logger.log(
                    20,
                    f'Updated best model to "{self._trainer.model_best}" (Previously "{prev_best}"). '
                    f'AutoGluon will default to using "{self._trainer.model_best}" for predict() and predict_proba().',
                )
            else:
                logger.warning(
                    f'Best model ("{model_to_set_best}") is not present in refit_full dictionary. '
                    f'Training may have failed on the refit model. AutoGluon will default to using "{model_best}" for predict() and predict_proba().'
                )

        te = time.time()
        logger.log(20, f'Refit complete, total runtime = {round(te - ts, 2)}s ... Best model: "{self._trainer.model_best}"')
        return refit_full_dict

    @property
    def model_best(self) -> str:
        """
        Returns the string model name of the best model by validation score that can infer.
        This is the same model used during inference when `predictor.predict` is called without specifying a model.
        This can be updated to be a model other than the model with best validation score by methods such as refit_full and set_model_best.

        Returns
        -------
        String model name of the best model
        """
        return self._model_best(can_infer=True)

    def _model_best(self, can_infer=None) -> str:
        self._assert_is_fit("model_best")
        # TODO: Set self._trainer.model_best to the best model at end of fit instead of best WeightedEnsemble.
        if self._trainer.model_best is not None:
            models = self._trainer.get_model_names(can_infer=can_infer)
            if self._trainer.model_best in models:
                return self._trainer.model_best
        return self._trainer.get_model_best(can_infer=can_infer)

    def set_model_best(self, model: str, save_trainer: bool = False):
        """
        Sets the model to be used by default when calling `predictor.predict(data)`.
        By default, this is the model with the best validation score, but this is not always the case.
        If manually set, this can be overwritten internally if further training occurs, such as through fit_extra, refit_full, or distill.

        Parameters
        ----------
        model : str
            Name of model to set to best. If model does not exist or cannot infer, raises an AssertionError.
        save_trainer : bool, default = False
            If True, self._trainer is saved with the new model_best value, such that it is reflected when predictor is loaded in future from disk.
        """
        self._assert_is_fit("set_model_best")
        models = self._trainer.get_model_names(can_infer=True)
        if model in models:
            self._trainer.model_best = model
        else:
            raise AssertionError(f'Model "{model}" is not a valid model to specify as best! Valid models: {models}')
        if save_trainer:
            self._trainer.save()

    def model_refit_map(self, inverse=False) -> dict[str, str]:
        """
        Returns a dictionary of original model name -> refit full model name.
        Empty unless `refit_full=True` was set during fit or `predictor.refit_full()` was called.
        This can be useful when determining the best model based off of `predictor.leaderboard()`, then getting the _FULL version of the model by passing its name as the key to this dictionary.

        Parameters
        ----------
        inverse : bool, default = False
            If True, instead returns a dictionary of refit full model name -> original model name (Swap keys with values)

        Returns
        -------
        Dictionary of original model name -> refit full model name.
        """
        self._assert_is_fit("model_refit_map")
        return self._trainer.model_refit_map(inverse=inverse)

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
        self._assert_is_fit("info")
        return self._learner.get_info(include_model_info=True, include_model_failures=True)

    def model_info(self, model: str) -> dict:
        """
        Returns metadata information about the given model.
        Equivalent output to `predictor.info()["model_info"][model]`

        Parameters
        ----------
        model: str
            The name of the model to get info for.

        Returns
        -------
        model_info: dict
            Model info dictionary

        """
        return self._trainer.get_model_info(model=model)

    # TODO: Add entire `hyperparameters` dict method for multiple models (including stack ensemble)
    # TODO: Add unit test
    def model_hyperparameters(
        self,
        model: str,
        include_ag_args_ensemble: bool = True,
        output_format: Literal["user", "all"] = "user",
    ) -> dict:
        """
        Returns the hyperparameters of a given model.

        Parameters
        ----------
        model: str
            The name of the model to get hyperparameters for.
        include_ag_args_ensemble: bool, default True
            If True, includes the ag_args_ensemble parameters if they exist (for example, when bagging is enabled).
        output_format: {"user", "all"}, default "user"
            If "user", returns the same hyperparameters specified by the user (only non-defaults).
            If "all", returns all hyperparameters used by the model (including default hyperparameters not specified by the user)
            Regardless of the output_format, they both are functionally equivalent if passed to AutoGluon.

        Returns
        -------
        model_hyperparameters: dict
            Dictionary of model hyperparameters.
            Equivalent to the model_hyperparameters specified by the user for this model in:
                `predictor.fit(..., hyperparameters={..., model_key: [..., model_hyperparameters]})`

        """
        # TODO: Move logic into trainer?
        info_model = self.model_info(model=model)
        if output_format == "user":
            if "bagged_info" in info_model:
                hyperparameters = info_model["bagged_info"]["child_hyperparameters_user"].copy()
                if include_ag_args_ensemble and info_model["hyperparameters_user"]:
                    hyperparameters["ag_args_ensemble"] = info_model["hyperparameters_user"]
            else:
                hyperparameters = info_model["hyperparameters_user"]
        elif output_format == "all":
            if "bagged_info" in info_model:
                hyperparameters = info_model["bagged_info"]["child_hyperparameters"].copy()
                if info_model["bagged_info"]["child_ag_args_fit"]:
                    hyperparameters["ag_args_fit"] = info_model["bagged_info"]["child_ag_args_fit"]
                if include_ag_args_ensemble:
                    bag_hyperparameters = info_model["hyperparameters"].copy()
                    if info_model["ag_args_fit"]:
                        bag_hyperparameters["ag_args_fit"] = info_model["ag_args_fit"]
                    if bag_hyperparameters:
                        hyperparameters["ag_args_ensemble"] = bag_hyperparameters
            else:
                hyperparameters = info_model["hyperparameters"]
        else:
            raise ValueError(f"output_format={output_format} is unknown!")
        return hyperparameters

    # TODO: Add data argument
    # TODO: Add option to disable OOF generation of newly fitted models
    # TODO: Move code logic to learner/trainer
    # TODO: Add fit() arg to perform this automatically at end of training
    # TODO: Consider adding cutoff arguments such as top-k models
    def fit_weighted_ensemble(
        self,
        base_models: list = None,
        name_suffix: str = "Best",
        expand_pareto_frontier: bool = False,
        time_limit: float = None,
        refit_full: bool = False,
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
    ):
        """
        Fits new weighted ensemble models to combine predictions of previously-trained models.
        `cache_data` must have been set to `True` during the original training to enable this functionality.

        Parameters
        ----------
        base_models: list, default = None
            List of model names the weighted ensemble can consider as candidates.
            If None, all previously trained models are considered except for weighted ensemble models.
            As an example, to train a weighted ensemble that can only have weights assigned to the models 'model_a' and 'model_b', set `base_models=['model_a', 'model_b']`
        name_suffix: str, default = 'Best'
            Name suffix to add to the name of the newly fitted ensemble model.
        expand_pareto_frontier: bool, default = False
            If True, will train N-1 weighted ensemble models instead of 1, where `N=len(base_models)`.
            The final model trained when True is equivalent to the model trained when False.
            These weighted ensemble models will attempt to expand the pareto frontier.
            This will create many different weighted ensembles which have different accuracy/memory/inference-speed trade-offs.
            This is particularly useful when inference speed is an important consideration.
        time_limit: float, default = None
            Time in seconds each weighted ensemble model is allowed to train for. If `expand_pareto_frontier=True`, the `time_limit` value is applied to each model.
            If None, the ensemble models train without time restriction.
        refit_full : bool, default = False
            If True, will apply refit_full to all weighted ensembles created during this call.
            Identical to calling `predictor.refit_full(model=predictor.fit_weighted_ensemble(...))`
        num_cpus: int | str, default = "auto"
            The total amount of cpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
            Users generally don't need to set this value
        num_gpus: int | str, default = "auto"
            The total amount of gpus you want AutoGluon predictor to use.
            Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
            Users generally don't need to set this value

        Returns
        -------
        List of newly trained weighted ensemble model names.
        If an exception is encountered while training an ensemble model, that model's name will be absent from the list.
        """
        self._assert_is_fit("fit_weighted_ensemble")
        trainer = self._learner.load_trainer()

        if trainer.bagged_mode:
            X = trainer.load_X()
            y = trainer.load_y()
            fit = True
        else:
            X = trainer.load_X_val()
            y = trainer.load_y_val()
            fit = False

        total_resources = {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        }

        stack_name = "aux1"
        if base_models is None:
            base_models = trainer.get_model_names(stack_name="core")

        X_stack_preds = trainer.get_inputs_to_stacker(X=X, base_models=base_models, fit=fit, use_orig_features=False, use_val_cache=True)

        models = []

        if expand_pareto_frontier:
            leaderboard = self.leaderboard()
            leaderboard = leaderboard[leaderboard["model"].isin(base_models)]
            leaderboard = leaderboard.sort_values(by="pred_time_val")
            models_to_check = leaderboard["model"].tolist()
            for i in range(1, len(models_to_check) - 1):
                models_to_check_now = models_to_check[: i + 1]
                max_base_model_level = max([trainer.get_model_level(base_model) for base_model in models_to_check_now])
                weighted_ensemble_level = max_base_model_level + 1
                models += trainer.generate_weighted_ensemble(
                    X=X_stack_preds,
                    y=y,
                    level=weighted_ensemble_level,
                    stack_name=stack_name,
                    base_model_names=models_to_check_now,
                    name_suffix=name_suffix + "_Pareto" + str(i),
                    time_limit=time_limit,
                    total_resources=total_resources,
                )

        max_base_model_level = max([trainer.get_model_level(base_model) for base_model in base_models])
        weighted_ensemble_level = max_base_model_level + 1
        models += trainer.generate_weighted_ensemble(
            X=X_stack_preds,
            y=y,
            level=weighted_ensemble_level,
            stack_name=stack_name,
            base_model_names=base_models,
            name_suffix=name_suffix,
            time_limit=time_limit,
            total_resources=total_resources,
        )

        if refit_full:
            refit_models_dict = self.refit_full(model=models, num_cpus=num_cpus, num_gpus=num_gpus)
            models += [refit_models_dict[m] for m in models]

        return models

    def calibrate_decision_threshold(
        self,
        data: pd.DataFrame | str | None = None,
        metric: str | Scorer | None = None,
        model: str = "best",
        decision_thresholds: int | list[float] = 25,
        secondary_decision_thresholds: int | None = 19,
        subsample_size: int | None = 1000000,
        verbose: bool = True,
    ) -> float:
        """
        Calibrate the decision threshold in binary classification to optimize a given metric.
        You can pass the output of this method as input to `predictor.set_decision_threshold` to update the predictor.
        Will raise an AssertionError if `predictor.problem_type != 'binary'`.

        Note that while calibrating the decision threshold can help to improve a given metric,
        other metrics may end up having worse scores.
        For example, calibrating on `balanced_accuracy` will often harm `accuracy`.
        Users should keep this in mind while leveraging decision threshold calibration.

        Parameters
        ----------
        data : pd.DataFrame or str, optional
            The data to use for calibration. Must contain the label column.
            We recommend to keep this value as None unless you are an advanced user and understand the implications.
            If None, will use internal data such as the holdout validation data or out-of-fold predictions.
        metric : autogluon.core.metrics.Scorer or str, default = None
            The metric to optimize during calibration.
            If None, uses `predictor.eval_metric`.
        model : str, default = 'best'
            The model to use prediction probabilities of when calibrating the threshold.
            If 'best', will use `predictor.model_best`.
        decision_thresholds : int | list[float], default = 25
            The number of decision thresholds on either side of `0.5` to search.
            The default of 25 will result in 51 searched thresholds: [0.00, 0.02, 0.04, ..., 0.48, 0.50, 0.52, ..., 0.96, 0.98, 1.00]
            Alternatively, a list of decision thresholds can be passed and only the thresholds in the list will be searched.
        secondary_decision_thresholds : int | None, default = 19
            The number of secondary decision thresholds to check on either side of the threshold identified in the first phase.
            Skipped if None.
            For example, if decision_thresholds=50 and 0.14 was identified as the optimal threshold, while secondary_decision_threshold=9,
                Then the following additional thresholds are checked:
                    [0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149]
        subsample_size : int | None, default = 1000000
            When `subsample_size` is not None and `data` contains more rows than `subsample_size`, samples to `subsample_size` rows to speed up calibration.
            Usually it is not necessary to use more than 1 million rows for calibration.
        verbose : bool, default = True
            If True, will log information about the calibration process.

        Returns
        -------
        Decision Threshold: A float between 0 and 1 defining the decision boundary for predictions that
        maximizes the `metric` score on the `data` for the `model`.
        """
        # TODO: v1.2
        #  Calculate optimal threshold for each model separately when deciding best model
        #  time limit
        #  update validation scores of models based on threshold
        #  speed up the logic / search for optimal threshold more efficiently
        #  make threshold calibration part of internal optimization, such as during fit_weighted_ensemble.
        #  precision has strange edge-cases where it flips from 1.0 to 0.0 score due to becoming undefined
        #    consider warning users who pass this metric,
        #    or edit this metric so they do not flip value when undefined.
        #      UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
        #      Use `zero_division` parameter to control this behavior.

        self._assert_is_fit("calibrate_decision_threshold")
        assert self.problem_type == BINARY, f'calibrate_decision_threshold is only available for `problem_type="{BINARY}"`'
        data = self._get_dataset(data, allow_nan=True)

        if metric is None:
            metric = self.eval_metric
        if model == "best":
            model = self.model_best

        return self._learner.calibrate_decision_threshold(
            data=data,
            metric=metric,
            model=model,
            decision_thresholds=decision_thresholds,
            secondary_decision_thresholds=secondary_decision_thresholds,
            subsample_size=subsample_size,
            verbose=verbose,
        )

    def predict_oof(self, model: str = None, *, transformed=False, train_data=None, internal_oof=False, decision_threshold=None, can_infer=None) -> pd.Series:
        """
        Note: This is advanced functionality not intended for normal usage.

        Returns the out-of-fold (OOF) predictions for every row in the training data.

        For a similar method, refer to :meth:`TabularPredictor.predict_multi` with `data=None`.
        For more information, refer to `predict_proba_oof()` documentation.

        Parameters
        ----------
        model : str (optional)
            Refer to `predict_proba_oof()` documentation.
        transformed : bool, default = False
            Refer to `predict_proba_oof()` documentation.
        train_data : pd.DataFrame, default = None
            Refer to `predict_proba_oof()` documentation.
        internal_oof : bool, default = False
            Refer to `predict_proba_oof()` documentation.
        decision_threshold : float, default = None
            Refer to `predict_multi` documentation.
        can_infer : bool, default = None
            Refer to `predict_proba_oof()` documentation.

        Returns
        -------
        :class:`pd.Series` object of the out-of-fold training predictions of the model.
        """
        self._assert_is_fit("predict_oof")
        if decision_threshold is None:
            decision_threshold = self.decision_threshold
        y_pred_proba_oof = self.predict_proba_oof(
            model=model, transformed=transformed, as_multiclass=True, train_data=train_data, internal_oof=internal_oof, can_infer=can_infer
        )
        y_pred_oof = get_pred_from_proba_df(y_pred_proba_oof, problem_type=self.problem_type, decision_threshold=decision_threshold)
        if transformed:
            return self._learner.label_cleaner.to_transformed_dtype(y_pred_oof)
        return y_pred_oof

    # TODO: Remove train_data argument once we start caching the raw original data: Can just load that instead.
    def predict_proba_oof(
        self, model: str = None, *, transformed=False, as_multiclass=True, train_data=None, internal_oof=False, can_infer=None
    ) -> pd.DataFrame | pd.Series:
        """
        Note: This is advanced functionality not intended for normal usage.

        Returns the out-of-fold (OOF) predicted class probabilities for every row in the training data.
        OOF prediction probabilities may provide unbiased estimates of generalization accuracy (reflecting how predictions will behave on new data)
        Predictions for each row are only made using models that were fit to a subset of data where this row was held-out.

        For a similar method, refer to :meth:`TabularPredictor.predict_proba_multi` with `data=None`.

        Warning: This method will raise an exception if called on a model that is not a bagged ensemble. Only bagged models (such a stacker models) can produce OOF predictions.
            This also means that refit_full models and distilled models will raise an exception.
        Warning: If intending to join the output of this method with the original training data, be aware that a rare edge-case issue exists:
            Multiclass problems with rare classes combined with the use of the 'log_loss' eval_metric may have forced AutoGluon to duplicate rows in the training data to satisfy minimum class counts in the data.
            If this has occurred, then the indices and row counts of the returned :class:`pd.Series` in this method may not align with the training data.
            In this case, consider fetching the processed training data using `predictor.load_data_internal()` instead of using the original training data.
            A more benign version of this issue occurs when 'log_loss' wasn't specified as the eval_metric but rare classes were dropped by AutoGluon.
            In this case, not all original training data rows will have an OOF prediction. It is recommended to either drop these rows during the join or to get direct predictions on the missing rows via :meth:`TabularPredictor.predict_proba`.

        Parameters
        ----------
        model : str (optional)
            The name of the model to get out-of-fold predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`
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
            If `train_data` is specified and `model` is unable to predict and rows were dropped internally, an exception will be raised.
        internal_oof : bool, default = False
            [Advanced Option] Return the internal OOF preds rather than the externally facing OOF preds.
            Internal OOF preds may have more/fewer rows than was provided in train_data, and are incompatible with external data.
            If you don't know what this does, keep it as False.
        can_infer : bool, default = None
            Only used if `model` is not specified.
            This is used to determine if the best model must be one that is able to predict on new data (True).
            If None, the best model does not need to be able to infer on new data.

        Returns
        -------
        :class:`pd.Series` or :class:`pd.DataFrame` object of the out-of-fold training prediction probabilities of the model.
        """
        self._assert_is_fit("predict_proba_oof")
        if model is None:
            model = self._model_best(can_infer=can_infer)
        if not self._trainer.bagged_mode:
            raise AssertionError("Predictor must be in bagged mode to get out-of-fold predictions.")
        if self._trainer.get_model_attribute(model=model, attribute="refit_full", default=False):
            model_to_get_oof = self._trainer.get_model_attribute(model=model, attribute="refit_full_parent")
            # TODO: bagged-with-holdout refit to bagged-no-holdout should still be able to return out-of-fold predictions
        else:
            model_to_get_oof = model
        if model != model_to_get_oof:
            logger.log(20, f'Using OOF from "{model_to_get_oof}" as a proxy for "{model}".')
        if self._trainer.get_model_attribute_full(model=model_to_get_oof, attribute="val_in_fit", func=max):
            raise AssertionError(f"Model {model_to_get_oof} does not have out-of-fold predictions because it used a validation set during training.")
        y_pred_proba_oof_transformed = self.transform_features(base_models=[model_to_get_oof], return_original_features=False)
        if not internal_oof:
            is_duplicate_index = y_pred_proba_oof_transformed.index.duplicated(keep="first")
            if is_duplicate_index.any():
                logger.log(
                    20,
                    "Detected duplicate indices... This means that data rows may have been duplicated during training. "
                    "Removing all duplicates except for the first instance.",
                )
                y_pred_proba_oof_transformed = y_pred_proba_oof_transformed[is_duplicate_index == False]
            if self._learner._pre_X_rows is not None and len(y_pred_proba_oof_transformed) < self._learner._pre_X_rows:
                len_diff = self._learner._pre_X_rows - len(y_pred_proba_oof_transformed)
                if train_data is None:
                    logger.warning(
                        f"WARNING: {len_diff} rows of training data were dropped internally during fit. "
                        f"The output will not contain all original training rows.\n"
                        f"If attempting to get `oof_pred_proba`, DO NOT pass `train_data` into `predictor.predict_proba` or `predictor.transform_features`!\n"
                        f"Instead this can be done by the following "
                        f"(Ensure `train_data` is identical to when it was used in fit):\n"
                        f"oof_pred_proba = predictor.predict_proba_oof(train_data=train_data)\n"
                        f"oof_pred = predictor.predict_oof(train_data=train_data)\n"
                    )
                else:
                    missing_idx = list(train_data.index.difference(y_pred_proba_oof_transformed.index))
                    if len(missing_idx) > 0:
                        missing_idx_data = train_data.loc[missing_idx]
                        missing_pred_proba = self.transform_features(data=missing_idx_data, base_models=[model], return_original_features=False)
                        y_pred_proba_oof_transformed = pd.concat([y_pred_proba_oof_transformed, missing_pred_proba])
                        y_pred_proba_oof_transformed = y_pred_proba_oof_transformed.reindex(list(train_data.index))

        if self.problem_type == MULTICLASS and self._learner.label_cleaner.problem_type_transform == MULTICLASS:
            y_pred_proba_oof_transformed.columns = copy.deepcopy(self._learner.label_cleaner.ordered_class_labels_transformed)
        elif self.problem_type == QUANTILE:
            y_pred_proba_oof_transformed.columns = self.quantile_levels
        else:
            y_pred_proba_oof_transformed.columns = [self.label]
            y_pred_proba_oof_transformed = y_pred_proba_oof_transformed[self.label]
            if as_multiclass and self.problem_type == BINARY:
                y_pred_proba_oof_transformed = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(
                    y_pred_proba_oof_transformed, as_pandas=True
                )
            elif self.problem_type == MULTICLASS:
                if transformed:
                    y_pred_proba_oof_transformed = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(
                        y_pred_proba_oof_transformed, as_pandas=True
                    )
                    y_pred_proba_oof_transformed.columns = copy.deepcopy(self._learner.label_cleaner.ordered_class_labels_transformed)
        if transformed:
            return y_pred_proba_oof_transformed
        else:
            return self.transform_labels(labels=y_pred_proba_oof_transformed, inverse=True, proba=True)

    @property
    def positive_class(self) -> int | str:
        """
        Returns the positive class name in binary classification. Useful for computing metrics such as F1 which require a positive and negative class.
        In binary classification, :class:`TabularPredictor.predict_proba(as_multiclass=False)` returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        return self._learner.positive_class

    def load_data_internal(self, data="train", return_X=True, return_y=True):
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
        self._assert_is_fit("load_data_internal")
        if data == "train":
            load_X = self._trainer.load_X
            load_y = self._trainer.load_y
        elif data == "val":
            load_X = self._trainer.load_X_val
            load_y = self._trainer.load_y_val
        else:
            raise ValueError(f"data must be one of: ['train', 'val'], but was '{data}'.")
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
        self._assert_is_fit("save_space")
        self._trainer.reduce_memory_size(
            remove_data=remove_data,
            remove_fit_stack=remove_fit_stack,
            remove_fit=True,
            remove_info=False,
            requires_save=requires_save,
            reduce_children=reduce_children,
        )

    def delete_models(
        self,
        models_to_keep: str | list[str] | None = None,
        models_to_delete: str | list[str] | None = None,
        allow_delete_cascade: bool = False,
        delete_from_disk: bool = True,
        dry_run: bool = False,
    ):
        """
        Deletes models from `predictor`.
        This can be helpful to minimize memory usage and disk usage, particularly for model deployment.
        This will remove all references to the models in `predictor`.
            For example, removed models will not appear in `predictor.leaderboard()`.
        WARNING: If `delete_from_disk=True`, this will DELETE ALL FILES in the deleted model directories, regardless if they were created by AutoGluon or not.
            DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.

        Parameters
        ----------
        models_to_keep : str or list[str], default = None
            Name of model or models to not delete.
            All models that are not specified and are also not required as a dependency of any model in `models_to_keep` will be deleted.
            Specify `models_to_keep='best'` to keep only the best model and its model dependencies.
            `models_to_delete` must be None if `models_to_keep` is set.
            To see the list of possible model names, use: `predictor.model_names()` or `predictor.leaderboard()`.
        models_to_delete : str or list[str], default = None
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
        dry_run : bool, default = False
            If `True`, then deletions don't occur, and logging statements are printed describing what would have occurred.
            Set `dry_run=False` to perform the deletions.

        """
        self._assert_is_fit("delete_models")
        if models_to_keep == "best":
            models_to_keep = self.model_best
        self._trainer.delete_models(
            models_to_keep=models_to_keep,
            models_to_delete=models_to_delete,
            allow_delete_cascade=allow_delete_cascade,
            delete_from_disk=delete_from_disk,
            dry_run=dry_run,
        )

    def disk_usage(self) -> int:
        """
        Returns the combined size of all files under the `predictor.path` directory in bytes.
        """
        return get_directory_size(self.path)

    def disk_usage_per_file(self, *, sort_by: str = "size", include_path_in_name: bool = False) -> pd.Series:
        """
        Returns the size of each file under the `predictor.path` directory in bytes.

        Parameters
        ----------
        sort_by : str, default = "size"
            If None, output files will be ordered based on order of search in os.walk(path).
            If "size", output files will be ordered in descending order of file size.
            If "name", output files will be ordered by name in ascending alphabetical order.
        include_path_in_name : bool, default = False
            If True, includes the full path of the file including the input `path` as part of the index in the output pd.Series.
            If False, removes the `path` prefix of the file path in the index of the output pd.Series.

            For example, for a file located at `foo/bar/model.pkl`, with path='foo/'
                If True, index will be `foo/bar/model.pkl`
                If False, index will be `bar/model.pkl`

        Returns
        -------
        pd.Series with index file path and value file size in bytes.
        """
        return get_directory_size_per_file(self.path, sort_by=sort_by, include_path_in_name=include_path_in_name)

    def model_names(
        self,
        stack_name: str = None,
        level: int = None,
        can_infer: bool = None,
        models: list[str] = None,
        persisted: bool = None,
    ) -> list[str]:
        """
        Returns the list of model names trained in this `predictor` object.

        Parameters
        ----------
        stack_name: str, default = None
            If specified, returns only models under a given stack name.
        level: int, default = None
            If specified, returns only models at the given stack level.
        can_infer: bool, default = None
            If specified, returns only models that can/cannot infer on new data.
        models: list[str], default = None
            The list of model names to consider.
            If None, considers all models.
        persisted: bool, default = None
            If None: no filtering will occur based on persisted status
            If True: will return only the models that are persisted in memory via `predictor.persist()`
            If False: will return only the models that are not persisted in memory via `predictor.persist()`

        Returns
        -------
        List of model names
        """
        self._assert_is_fit("model_names")
        model_names = self._trainer.get_model_names(stack_name=stack_name, level=level, can_infer=can_infer, models=models)
        if persisted is not None:
            persisted_model_names = list(self._trainer.models.keys())
            if persisted:
                model_names = [m for m in model_names if m in persisted_model_names]
            else:
                model_names = [m for m in model_names if m not in persisted_model_names]
        return model_names

    def distill(
        self,
        train_data: pd.DataFrame | str = None,
        tuning_data: pd.DataFrame | str = None,
        augmentation_data: pd.DataFrame = None,
        time_limit: float = None,
        hyperparameters: dict | str = None,
        holdout_frac: float = None,
        teacher_preds: str = "soft",
        augment_method: str = "spunge",
        augment_args: dict = {"size_factor": 5, "max_size": int(1e5)},
        models_name_suffix: str = None,
        verbosity: int = None,
    ):
        """
        [EXPERIMENTAL]
        Distill AutoGluon's most accurate ensemble-predictor into single models which are simpler/faster and require less memory/compute.
        Distillation can produce a model that is more accurate than the same model fit directly on the original training data.
        After calling `distill()`, there will be more models available in this Predictor, which can be evaluated using `predictor.leaderboard(test_data)` and deployed with: `predictor.predict(test_data, model=MODEL_NAME)`.
        This will raise an exception if `cache_data=False` was previously set in `fit()`.

        NOTE: Until catboost v0.24 is released, `distill()` with CatBoost students in multiclass classification requires you to first install catboost-dev: `pip install catboost-dev`

        Parameters
        ----------
        train_data : str or :class:`pd.DataFrame`, default = None
            Same as `train_data` argument of `fit()`.
            If None, the same training data will be loaded from `fit()` call used to produce this Predictor.
        tuning_data : str or :class:`pd.DataFrame`, default = None
            Same as `tuning_data` argument of `fit()`.
            If `tuning_data = None` and `train_data = None`: the same training/validation splits will be loaded from `fit()` call used to produce this Predictor,
            unless bagging/stacking was previously used in which case a new training/validation split is performed.
        augmentation_data : :class:`pd.DataFrame`, default = None
            An optional extra dataset of unlabeled rows that can be used for augmenting the dataset used to fit student models during distillation (ignored if None).
        time_limit : int, default = None
            Approximately how long (in seconds) the distillation process should run for.
            If None, no time-constraint will be enforced allowing the distilled models to fully train.
        hyperparameters : dict or str, default = None
            Specifies which models to use as students and what hyperparameter-values to use for them.
            Same as `hyperparameters` argument of `fit()`.
            If = None, then student models will use the same hyperparameters from `fit()` used to produce this Predictor.
            Note: distillation is currently only supported for ['GBM','NN_TORCH','RF','CAT'] student models, other models and their hyperparameters are ignored here.
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
        self._assert_is_fit("distill")
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)
        return self._learner.distill(
            X=train_data,
            X_val=tuning_data,
            time_limit=time_limit,
            hyperparameters=hyperparameters,
            holdout_frac=holdout_frac,
            verbosity=verbosity,
            models_name_suffix=models_name_suffix,
            teacher_preds=teacher_preds,
            augmentation_data=augmentation_data,
            augment_method=augment_method,
            augment_args=augment_args,
        )

    # TODO: v1.0 Move core logic to `trainer` level.
    # TODO: v1.0 Make it use leaderboard directly, allow to specify columns to include in the plot.
    # TODO: See if we can incorporate into tutorials (without causing crashes for users who try them)
    #  Might require using a different tool than pygraphviz to avoid the apt-get commands
    # TODO: v1.0 Rename to `plot_model_graph`
    # TODO: v1.0 Maybe add ensemble weights to the edges.
    def plot_ensemble_model(self, model: str = "best", *, prune_unused_nodes: bool = True, filename: str = "ensemble_model.png") -> str:
        """
        Output the visualized stack ensemble architecture of a model trained by `fit()`.
        The plot is stored to a file, `ensemble_model.png` in folder `predictor.path` (or by the name specified in `filename`)

        This function requires `graphviz` and `pygraphviz` to be installed because this visualization depends on those package.
        Unless this function will raise `ImportError` without being able to generate the visual of the ensemble model.

        To install the required package, run the below commands (for Ubuntu linux):

        $ sudo apt-get install graphviz graphviz-dev
        $ pip install pygraphviz

        For other platforms, refer to https://graphviz.org/ for Graphviz install, and https://pygraphviz.github.io/ for PyGraphviz.

        Parameters
        ----------
        model : str, default 'best'
            The model to highlight in golden orange, with all component models highlighted in yellow.
            If 'best', will default to the best model returned from `self.model_best`
        prune_unused_nodes : bool, default True
            If True, only plot the models that are components of the specified `model`.
            If False, will plot all models.
        filename : str, default 'ensemble_model.png'
            The filename to save the plot as. Will be located under the `self.path` folder.

        Returns
        -------
        The file name with the full path to the saved graphic on disk.

        Examples
        --------
        >>> from autogluon.tabular import TabularDataset, TabularPredictor
        >>> train_data = TabularDataset('train.csv')
        >>> predictor = TabularPredictor(label='class').fit(train_data)
        >>> path_to_png = predictor.plot_ensemble_model()
        >>>
        >>> # To view the plot inside a Jupyter Notebook, use the below code:
        >>> from IPython.display import Image, display
        >>> display(Image(filename=path_to_png))

        """
        self._assert_is_fit("plot_ensemble_model")
        try:
            import pygraphviz
        except:
            raise ImportError(
                "Visualizing ensemble network architecture requires the `pygraphviz` library. "
                "Try `sudo apt-get install graphviz graphviz-dev` followed by `pip install pygraphviz` to install on Linux, "
                "or refer to the method docstring for detailed installation instructions for other operating systems."
            )

        G = self._trainer.model_graph.copy()

        primary_model = model
        if primary_model == "best":
            primary_model = self.model_best
        all_models = self.model_names()
        assert primary_model in all_models, f'Unknown model "{primary_model}"! Valid models: {all_models}'
        if prune_unused_nodes == True:
            models_to_keep = self._trainer.get_minimum_model_set(model=primary_model)
            G = nx.subgraph(G, models_to_keep)

        models = list(G.nodes)
        fit_times = self._trainer.get_models_attribute_full(models=models, attribute="fit_time")
        predict_times = self._trainer.get_models_attribute_full(models=models, attribute="predict_time")

        A = nx.nx_agraph.to_agraph(G)

        for node in A.iternodes():
            node_name = node.name
            fit_time = fit_times[node_name]
            predict_time = predict_times[node_name]
            if fit_time is None:
                fit_time_str = "NaN"
            else:
                fit_time_str = f"{fit_time:.1f}s"
            if predict_time is None:
                predict_time_str = "NaN"
            else:
                predict_time_str = f"{predict_time:.2f}s"

            node_val_score = node.attr["val_score"]
            if node_val_score is None or (isinstance(node_val_score, str) and node_val_score == "None"):
                node_val_score_str = "NaN"
            else:
                node_val_score_str = f"{float(node.attr['val_score']):.4f}"
            label = f"{node.name}" f"\nscore_val: {node_val_score_str}" f"\nfit_time: {fit_time_str}" f"\npred_time_val: {predict_time_str}"
            # Remove unnecessary attributes
            node.attr.clear()
            node.attr["label"] = label

        A.graph_attr.update(rankdir="BT")
        A.node_attr.update(fontsize=10)
        A.node_attr.update(shape="rectangle")

        for node in A.iternodes():
            if node.name == primary_model:
                # Golden Orange
                node.attr["style"] = "filled"
                node.attr["fillcolor"] = "#ff9900"
                node.attr["shape"] = "box3d"
            elif nx.has_path(G, node.name, primary_model):
                # Yellow
                node.attr["style"] = "filled"
                node.attr["fillcolor"] = "#ffcc00"
            # Else: White

        model_image_fname = os.path.join(self.path, filename)
        A.draw(model_image_fname, format="png", prog="dot")
        return model_image_fname

    @staticmethod
    def _summarize(key, msg, results):
        if key in results:
            print(msg + ": " + str(results[key]))

    @staticmethod
    def _get_dataset(data, allow_nan: bool = False) -> pd.DataFrame | None:
        if data is None:
            if allow_nan:
                return data
            else:
                raise TypeError("data=None is invalid. data must be a pd.DataFrame or str file path to data")
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            return TabularDataset(data)
        elif isinstance(data, pd.Series):
            raise TypeError(
                "data must be a pd.DataFrame, not pd.Series. \
                   To predict on just single example (ith row of table), use data.iloc[[i]] rather than data.iloc[i]"
            )
        else:
            raise TypeError("data must be a pd.DataFrame or str file path to data")

    def _validate_hyperparameter_tune_kwargs(self, hyperparameter_tune_kwargs, time_limit=None):
        """
        Returns True if hyperparameter_tune_kwargs is None or can construct a valid scheduler.
        Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return True

        scheduler_cls, scheduler_params = scheduler_factory(
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, time_out=time_limit, nthreads_per_trial="auto", ngpus_per_trial="auto"
        )

        if scheduler_params.get("dist_ip_addrs", None):
            logger.warning("Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.")

        if scheduler_params["num_trials"] == 1:
            logger.warning(
                "Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. This can occur if time_limit was not specified in `fit()`."
            )
            return False

        scheduler_ngpus = scheduler_params["resource"].get("num_gpus", 0)
        if scheduler_ngpus is not None and isinstance(scheduler_ngpus, int) and scheduler_ngpus > 1:
            logger.warning(f"Warning: TabularPredictor currently doesn't use >1 GPU per training run. Detected {scheduler_ngpus} GPUs.")

        return True

    def _set_hyperparameter_tune_kwargs_in_ag_args(self, hyperparameter_tune_kwargs, ag_args, time_limit):
        if hyperparameter_tune_kwargs is not None and "hyperparameter_tune_kwargs" not in ag_args:
            if "hyperparameter_tune_kwargs" in ag_args:
                AssertionError("hyperparameter_tune_kwargs was specified in both ag_args and in kwargs. Please only specify once.")
            else:
                ag_args["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs
        if ag_args.get("hyperparameter_tune_kwargs", None) is not None:
            logger.log(30, "Warning: hyperparameter tuning is currently experimental and may cause the process to hang.")
        return ag_args

    def _set_post_fit_vars(self, learner: AbstractTabularLearner = None):
        if learner is not None:
            self._learner: AbstractTabularLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._learner.persist_trainer(low_memory=True)
            self._trainer: AbstractTabularTrainer = self._learner.load_trainer()  # Trainer object

    @classmethod
    def _load_version_file(cls, path: str) -> str:
        """
        Loads the version file that is part of the saved predictor artifact.
        The version file contains a string matching `predictor._learner.version`.

        Parameters
        ----------
        path: str
            The path that would be used to load the predictor via `predictor.load(path)`

        Returns
        -------
        The version of AutoGluon used to fit the predictor, as a string.

        """
        version_file_path = os.path.join(path, cls._predictor_version_file_name)
        try:
            version = load_str.load(path=version_file_path)
        except:
            # Loads the old version file used in `autogluon.tabular<=1.1.0`, named `__version__`.
            # This file name was changed because Kaggle does not allow uploading files named `__version__`.
            version_file_path = os.path.join(path, "__version__")
            version = load_str.load(path=version_file_path)
        return version

    @classmethod
    def _load_metadata_file(cls, path: str, silent: bool = True):
        metadata_file_path = os.path.join(path, cls._predictor_metadata_file_name)
        return load_json.load(path=metadata_file_path, verbose=not silent)

    def _save_version_file(self, silent: bool = False):
        version_file_contents = f"{__version__}"
        version_file_path = os.path.join(self.path, self._predictor_version_file_name)
        save_str.save(path=version_file_path, data=version_file_contents, verbose=not silent)

    def _save_metadata_file(self, silent: bool = False):
        """
        Save metadata json file to disk containing information such as
        python version, autogluon version, installed packages, operating system, etc.
        """
        metadata_file_path = os.path.join(self.path, self._predictor_metadata_file_name)

        metadata = get_autogluon_metadata()

        save_json.save(path=metadata_file_path, obj=metadata)
        if not silent:
            logger.log(15, f"Saving {metadata_file_path}")

    def save(self, silent: bool = False):
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
        save_pkl.save(path=os.path.join(path, self.predictor_file_name), object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer
        self._save_version_file(silent=silent)
        try:
            self._save_metadata_file(silent=silent)
        except Exception as e:
            logger.log(30, f"Failed to save metadata file due to exception {e}, skipping...")
        if not silent:
            logger.log(20, f'TabularPredictor saved. To load, use: predictor = TabularPredictor.load("{self.path}")')

    @classmethod
    def _load(cls, path: str) -> "TabularPredictor":
        """
        Inner load method, called in `load`.
        """
        predictor: TabularPredictor = load_pkl.load(path=os.path.join(path, cls.predictor_file_name))
        learner = predictor._learner_type.load(path)
        predictor._set_post_fit_vars(learner=learner)
        return predictor

    @classmethod
    def load(
        cls,
        path: str,
        verbosity: int = None,
        require_version_match: bool = True,
        require_py_version_match: bool = True,
        check_packages: bool = False,
    ) -> "TabularPredictor":
        """
        Load a TabularPredictor object previously produced by `fit()` from file and returns this object. It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        .. warning::

            :meth:`autogluon.tabular.TabularPredictor.load` uses `pickle` module implicitly, which is known to
            be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during
            unpickling. Never load data that could have come from an untrusted source, or that could have been tampered
            with. **Only load data you trust.**

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
        require_py_version_match : bool, default = True
            If True, will raise an AssertionError if the Python version of the loaded predictor does not match the installed Python version.
                Micro version differences such as 3.9.2 and 3.9.7 will log a warning but will not raise an exception.
            If False, will allow loading of models trained on incompatible python versions, but is NOT recommended. Users may run into numerous issues if attempting this.
        check_packages : bool, default = False
            If True, checks package versions of the loaded predictor against the package versions of the current environment.
            Warnings will be logged for each mismatch of package version.

        Returns
        -------
        predictor : TabularPredictor

        Examples
        --------
        >>> predictor = TabularPredictor.load(path_to_predictor)

        """
        if verbosity is not None:
            set_logger_verbosity(verbosity)  # Reset logging after load (could be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        try:
            from ..version import __version__

            version_current = __version__
        except:
            version_current = None

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        try:
            version_saved = cls._load_version_file(path=path)
        except:
            logger.warning(
                f'WARNING: Could not find version file at "{os.path.join(path, cls._predictor_version_file_name)}".\n'
                f"This means that the predictor was fit in an AutoGluon version `<=0.3.1`."
            )
            version_saved = None

        if version_saved is None:
            predictor = cls._load(path=path)
            try:
                version_saved = predictor._learner.version
            except:
                version_saved = None
        else:
            predictor = None
        if version_saved is None:
            version_saved = "Unknown (Likely <=0.0.11)"

        check_saved_predictor_version(
            version_current=version_current,
            version_saved=version_saved,
            require_version_match=require_version_match,
            logger=logger,
        )

        try:
            metadata_init = cls._load_metadata_file(path=path)
        except:
            logger.warning(
                f'WARNING: Could not find metadata file at "{os.path.join(path, cls._predictor_metadata_file_name)}".\n'
                f"This could mean that the predictor was fit in a version `<=0.5.2`."
            )
            metadata_init = None

        metadata_load = get_autogluon_metadata()

        if metadata_init is not None:
            try:
                compare_autogluon_metadata(original=metadata_init, current=metadata_load, check_packages=check_packages)
            except:
                logger.log(30, "WARNING: Exception raised while comparing metadata files, skipping comparison...")
            if require_py_version_match:
                if metadata_init["py_version"] != metadata_load["py_version"]:
                    raise AssertionError(
                        f'Predictor was created on Python version {metadata_init["py_version"]} '
                        f'but is being loaded with Python version {metadata_load["py_version"]}. '
                        f"Please ensure the versions match to avoid instability. While it is NOT recommended, "
                        f"this error can be bypassed by specifying `require_py_version_match=False`."
                    )

        if predictor is None:
            predictor = cls._load(path=path)

        return predictor

    @classmethod
    def load_log(cls, predictor_path: str = None, log_file_path: Optional[str] = None) -> list[str]:
        """
        Load log files of a predictor

        Parameters
        ----------
        predictor_path: Optional[str], default = None
            Path to the predictor to load the log.
            This can be used when the predictor was initialized with `log_file_path="auto"` to fetch the log file automatically
        log_file_path: Optional[str], default = None
            Path to the log file.
            If you specified a `log_file_path` while initializing the predictor, you should use `log_file_path` to load the log file instead.
            At least one of `predictor_path` or `log_file_path` must to be specified

        Returns
        -------
        list[str]
            A list containing lines of the log file
        """
        file_path = log_file_path
        if file_path is None:
            assert predictor_path is not None, "Please either provide `predictor_path` or `log_file_path` to load the log file"
            file_path = os.path.join(predictor_path, "logs", cls._predictor_log_file_name)
        assert os.path.isfile(file_path), f"Log file does not exist at {file_path}"
        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        return lines

    def _setup_log_to_file(self, log_file_path: str):
        if log_file_path == "auto":
            log_file_path = os.path.join(self.path, "logs", self._predictor_log_file_name)
        log_file_path = os.path.abspath(os.path.normpath(log_file_path))
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        add_log_to_file(log_file_path)

    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            "learner_type",
            "learner_kwargs",
            "quantile_levels",
            "default_base_path",
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f"Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}")

    def _validate_fit_kwargs(self, *, kwargs: dict) -> dict:
        # TODO:
        #  Valid core_kwargs values:
        #  ag_args, ag_args_fit, ag_args_ensemble, stack_name, ensemble_type, name_suffix, time_limit
        #  Valid aux_kwargs values:
        #  name_suffix, time_limit, stack_name, aux_hyperparameters, ag_args, ag_args_ensemble, fit_weighted_ensemble

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
            feature_generator="auto",
            unlabeled_data=None,
            _feature_generator_kwargs=None,
            # learning curves and test data (for logging purposes only)
            learning_curves=False,
            test_data=None,
            raise_on_model_failure=False,
            # experimental
            _experimental_dynamic_hyperparameters=False,
        )
        kwargs, ds_valid_keys = self._sanitize_dynamic_stacking_kwargs(kwargs)
        kwargs = self._validate_fit_extra_kwargs(kwargs, extra_valid_keys=list(fit_kwargs_default.keys()) + ds_valid_keys)
        kwargs_sanitized = fit_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        return kwargs_sanitized

    def _validate_calibrate_decision_threshold(self, calibrate_decision_threshold: bool | str):
        valid_calibrate_decision_threshold_options = [True, False, "auto"]
        if calibrate_decision_threshold not in valid_calibrate_decision_threshold_options:
            raise ValueError(
                f"`calibrate_decision_threshold` must be a value in " f"{valid_calibrate_decision_threshold_options}, but is: {calibrate_decision_threshold}"
            )

    def _validate_num_cpus(self, num_cpus: int | str):
        if num_cpus is None:
            raise ValueError(f"`num_cpus` must be an int or 'auto'. Value: {num_cpus}")
        if isinstance(num_cpus, str):
            if num_cpus != "auto":
                raise ValueError(f"`num_cpus` must be an int or 'auto'. Value: {num_cpus}")
        elif not isinstance(num_cpus, int):
            raise TypeError(f"`num_cpus` must be an int or 'auto'. Found: {type(num_cpus)} | Value: {num_cpus}")
        else:
            if num_cpus < 1:
                raise ValueError(f"`num_cpus` must be greater than or equal to 1. (num_cpus={num_cpus})")

    def _validate_num_gpus(self, num_gpus: int | float | str):
        if num_gpus is None:
            raise ValueError(f"`num_gpus` must be an int, float, or 'auto'. Value: {num_gpus}")
        if isinstance(num_gpus, str):
            if num_gpus != "auto":
                raise ValueError(f"`num_gpus` must be an int, float, or 'auto'. Value: {num_gpus}")
        elif not isinstance(num_gpus, (int, float)):
            raise TypeError(f"`num_gpus` must be an int, float, or 'auto'. Found: {type(num_gpus)} | Value: {num_gpus}")
        else:
            if num_gpus < 0:
                raise ValueError(f"`num_gpus` must be greater than or equal to 0. (num_gpus={num_gpus})")

    def _validate_and_set_memory_limit(self, memory_limit: float | str):
        if memory_limit is None:
            raise ValueError(f"`memory_limit` must be an int, float, or 'auto'. Value: {memory_limit}")
        if isinstance(memory_limit, str):
            if memory_limit != "auto":
                raise ValueError(f"`memory_limit` must be an int, float, or 'auto'. Value: {memory_limit}")
        elif not isinstance(memory_limit, (int, float)):
            raise TypeError("`memory_limit` must be an int, float, or 'auto'." f" Found: {type(memory_limit)} | Value: {memory_limit}")
        else:
            if memory_limit <= 0:
                raise ValueError(f"`memory_limit` must be greater than 0. (memory_limit={memory_limit})")

        if memory_limit != "auto":
            logger.log(20, f"Enforcing custom memory (soft) limit of {memory_limit} GB!")
            os.environ["AG_MEMORY_LIMIT_IN_GB"] = str(memory_limit)

    def _validate_fit_strategy(self, fit_strategy: str):
        valid_values = ["sequential", "parallel"]
        if fit_strategy not in valid_values:
            raise ValueError(f"fit_strategy must be one of {valid_values}. Value: {fit_strategy}")

    def _fit_extra_kwargs_dict(self) -> dict:
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
            delay_bag_sets=False,
            num_stack_levels=None,
            hyperparameter_tune_kwargs=None,
            # core_kwargs -> +1 nest
            ag_args=None,
            ag_args_fit=None,
            ag_args_ensemble=None,
            included_model_types=None,
            excluded_model_types=None,
            # aux_kwargs -> +1 nest
            # post_fit_kwargs -> +1 nest
            set_best_to_refit_full=False,
            keep_only_best=False,
            save_space=False,
            refit_full=False,
            save_bag_folds=None,
            # other
            verbosity=self.verbosity,
            feature_prune_kwargs=None,
            raise_on_no_models_fitted=True,
            # private
            _save_bag_folds=None,
            calibrate="auto",
            # pseudo label
            pseudo_data=None,
            name_suffix=None,
        )

    @staticmethod
    def _sanitize_dynamic_stacking_kwargs(kwargs: dict) -> tuple[dict, list[str]]:
        ds_kwargs_key = "ds_args"
        ds_args = dict(
            validation_procedure="holdout",
            detection_time_frac=1 / 4,
            holdout_frac=1 / 9,
            n_folds=2,
            n_repeats=1,
            memory_safe_fits="auto",
            clean_up_fits=True,
            holdout_data=None,
            enable_ray_logging=True,
            enable_callbacks=True,
        )
        allowed_kes = set(ds_args.keys())

        if ds_kwargs_key in kwargs:
            kwargs[ds_kwargs_key] = copy.deepcopy(kwargs[ds_kwargs_key])
            ds_args.update(kwargs[ds_kwargs_key])

        key_mismatch = set(ds_args.keys()) - allowed_kes
        if key_mismatch:
            raise ValueError(f"Got invalid keys for `ds_args`. Allowed: {allowed_kes}. Got: {key_mismatch}")
        if ("validation_procedure" in ds_args) and (
            (not isinstance(ds_args["validation_procedure"], str)) or (ds_args["validation_procedure"] not in ["holdout", "cv"])
        ):
            raise ValueError("`validation_procedure` in `ds_args` must be str in {'holdout','cv'}. " + f"Got: {ds_args['validation_procedure']}")
        for arg_name in ["clean_up_fits", "enable_ray_logging"]:
            if (arg_name in ds_args) and (not isinstance(ds_args[arg_name], bool)):
                raise ValueError(f"`{arg_name}` in `ds_args` must be bool.  Got: {type(ds_args[arg_name])}")
        if "memory_safe_fits" in ds_args and not isinstance(ds_args["memory_safe_fits"], (bool, str)):
            raise ValueError(f"`memory_safe_fits` in `ds_args` must be bool or 'auto'.  Got: {type(ds_args['memory_safe_fits'])}")
        for arg_name in ["detection_time_frac", "holdout_frac"]:
            if (arg_name in ds_args) and ((not isinstance(ds_args[arg_name], float)) or (ds_args[arg_name] >= 1) or (ds_args[arg_name] <= 0)):
                raise ValueError(f"`{arg_name}` in `ds_args` must be float in (0,1).  Got: {type(ds_args[arg_name])}, {ds_args[arg_name]}")
        if ("n_folds" in ds_args) and ((not isinstance(ds_args["n_folds"], int)) or (ds_args["n_folds"] < 2)):
            raise ValueError(f"`n_folds` in `ds_args` must be int in [2, +inf).  Got: {type(ds_args['n_folds'])}, {ds_args['n_folds']}")
        if ("n_repeats" in ds_args) and ((not isinstance(ds_args["n_repeats"], int)) or (ds_args["n_repeats"] < 1)):
            raise ValueError(f"`n_repeats` in `ds_args` must be int in [1, +inf).  Got: {type(ds_args['n_repeats'])}, {ds_args['n_repeats']}")
        if ("holdout_data" in ds_args) and (not isinstance(ds_args["holdout_data"], (str, pd.DataFrame))) and (ds_args["holdout_data"] is not None):
            raise ValueError(f"`holdout_data` in `ds_args` must be None, str, or pd.DataFrame.  Got: {type(ds_args['holdout_data'])}")
        if (ds_args["validation_procedure"] == "cv") and (ds_args["holdout_data"] is not None):
            raise ValueError(
                "`validation_procedure` in `ds_args` is 'cv' but `holdout_data` in `ds_args` is specified."
                "You must decide for either (repeated) cross-validation or holdout validation."
            )
        kwargs[ds_kwargs_key] = ds_args
        return kwargs, [ds_kwargs_key]

    def _validate_fit_extra_kwargs(self, kwargs: dict, extra_valid_keys: list[str] | None = None):
        fit_extra_kwargs_default = self._fit_extra_kwargs_dict()

        allowed_kwarg_names = list(fit_extra_kwargs_default.keys())
        if extra_valid_keys is not None:
            allowed_kwarg_names += extra_valid_keys
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                public_kwarg_options = [kwarg for kwarg in allowed_kwarg_names if kwarg[0] != "_"]
                public_kwarg_options.sort()
                raise ValueError(f"Unknown `.fit` keyword argument specified: '{kwarg_name}'\nValid kwargs: {public_kwarg_options}")

        kwargs_sanitized = fit_extra_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        # Deepcopy args to avoid altering outer context
        deepcopy_args = ["ag_args", "ag_args_fit", "ag_args_ensemble", "included_model_types", "excluded_model_types"]
        for deepcopy_arg in deepcopy_args:
            kwargs_sanitized[deepcopy_arg] = copy.deepcopy(kwargs_sanitized[deepcopy_arg])

        refit_full = kwargs_sanitized["refit_full"]
        set_best_to_refit_full = kwargs_sanitized["set_best_to_refit_full"]
        if refit_full and not self._learner.cache_data:
            raise ValueError("`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.")
        if set_best_to_refit_full and not refit_full:
            raise ValueError(
                "`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`."
            )
        valid_calibrate_options = [True, False, "auto"]
        calibrate = kwargs_sanitized["calibrate"]
        if calibrate not in valid_calibrate_options:
            raise ValueError(f"`calibrate` must be a value in {valid_calibrate_options}, but is: {calibrate}")

        return kwargs_sanitized

    def _prune_data_features(self, train_features: list, other_features: list, is_labeled: bool) -> tuple[list, list]:
        """
        Removes certain columns from the provided datasets that do not contain predictive features.

        Parameters
        ----------
        train_features : list
            The features/columns for the incoming training data
        other_features : list
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

    def _validate_fit_data(
        self,
        train_data: str | pd.DataFrame,
        tuning_data: str | pd.DataFrame | None = None,
        test_data: str | pd.DataFrame | None = None,
        unlabeled_data: str | pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        if test_data is not None and isinstance(test_data, str):
            test_data = TabularDataset(test_data)
        if unlabeled_data is not None and isinstance(unlabeled_data, str):
            unlabeled_data = TabularDataset(unlabeled_data)

        if not isinstance(train_data, pd.DataFrame):
            raise AssertionError(f"train_data is required to be a pandas DataFrame, but was instead: {type(train_data)}")

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )

        self._validate_single_fit_dataset(train_data=train_data, other_data=tuning_data, name="tuning_data", is_labeled=True)
        self._validate_single_fit_dataset(train_data=train_data, other_data=test_data, name="test_data", is_labeled=True)
        self._validate_single_fit_dataset(train_data=train_data, other_data=unlabeled_data, name="unlabeled_data", is_labeled=False)

        return train_data, tuning_data, test_data, unlabeled_data

    def _validate_single_fit_dataset(self, train_data: pd.DataFrame, other_data: pd.DataFrame, name: str, is_labeled: bool = True):
        """
        Validates additional dataset, ensuring format is consistent with train dataset.

        Parameters:
        -----------
        train_data : DataFrame
            training set dataframe
        other_data : DataFrame
            additional data set
        name : str
            name of additional data set
        is_labeled : bool
            whether the other_data is labeled

        Returns:
        --------
        None
        """
        if other_data is not None:
            if not isinstance(other_data, pd.DataFrame):
                raise AssertionError(f"{name} is required to be a pandas DataFrame, but was instead: {type(other_data)}")
            self._validate_unique_indices(data=other_data, name=name)
            train_features = [column for column in train_data.columns if column != self.label]
            other_features = [column for column in other_data.columns if column != self.label]
            train_features, other_features = self._prune_data_features(train_features=train_features, other_features=other_features, is_labeled=is_labeled)
            train_features = np.array(train_features)
            other_features = np.array(other_features)
            if np.any(train_features != other_features):
                raise ValueError(f"Column names must match between train_data and {name}")

            if self.label in other_data:
                train_label_type = train_data[self.label].dtype
                other_label_type = other_data[self.label].dtype

                if train_label_type != other_label_type:
                    logger.warning(
                        f"WARNING: train_data and {name} have mismatched label column dtypes! "
                        f"train_label_type={train_label_type}, {name}_type={other_label_type}.\n"
                        f"\tYou should ensure the dtypes match to avoid bugs or instability.\n"
                        f"\tAutoGluon will attempt to convert the dtypes to align."
                    )

    def _initialize_learning_curve_params(self, learning_curves: dict | bool | None = None, problem_type: str | None = None) -> dict:
        """
        Convert users learning_curve dict parameters into ag_param format.
        Also, converts all metrics into list of autogluon Scorer objects.

        Parameters:
        -----------
        learning_curves : bool | dict | None
            If bool, whether to generate learning curves.
            If dict, the dictionary of learning_curves parameters passed into predictor from the user.
            If None, will not generate curves.
        problem_type : str | None
            The current problem type.

        Returns:
        --------
        params : dict
            The learning curves parameters in ag_params format.
        """
        if learning_curves is None or learning_curves == False:
            return {}
        elif type(learning_curves) != dict and type(learning_curves) != bool:
            raise ValueError("Learning curves parameter must be a boolean or dict!")

        # metrics defaults to self.eval_metric if not specified
        metrics = None
        use_error = False

        if type(learning_curves) == dict:
            if "metrics" in learning_curves:
                metrics = learning_curves["metrics"]
                if not isinstance(metrics, list):
                    metrics = [metrics]

                names, scorers = [], []
                for metric in metrics:
                    if isinstance(metric, str):
                        names.append(metric)
                    elif isinstance(metric, Scorer):
                        scorers.append(metric)

                names = [get_metric(name, problem_type, "eval_metric") for name in names]
                metrics = names + scorers

                # check for duplicate metrics / aliases
                all_metric_names = [metric.name for metric in metrics]
                if len(set(all_metric_names)) != len(all_metric_names):
                    from collections import Counter

                    counts = {metric: count for metric, count in Counter(all_metric_names).items() if count > 1}
                    raise ValueError(f"The following learning curve metrics appeared more than once: {counts}")

            if "use_error" in learning_curves:
                use_error = learning_curves["use_error"]

        params = {
            "ag.generate_curves": True,
            "ag.use_error_for_curve_metrics": use_error,
        }

        if metrics:
            params["ag.curve_metrics"] = metrics

        return params

    @staticmethod
    def _validate_unique_indices(data: pd.DataFrame, name: str):
        is_duplicate_index = data.index.duplicated(keep=False)
        if is_duplicate_index.any():
            duplicate_count = is_duplicate_index.sum()
            raise AssertionError(
                f"{name} contains {duplicate_count} duplicated indices. "
                "Please ensure DataFrame indices are unique.\n"
                f"\tYou can identify the indices which are duplicated via `{name}.index.duplicated(keep=False)`"
            )

    @staticmethod
    def _validate_infer_limit(infer_limit: float, infer_limit_batch_size: int) -> tuple[float, int]:
        if infer_limit_batch_size is not None:
            if not isinstance(infer_limit_batch_size, int):
                raise ValueError(f"infer_limit_batch_size must be type int, but was instead type {type(infer_limit_batch_size)}")
            elif infer_limit_batch_size < 1:
                raise AssertionError(f"infer_limit_batch_size must be >=1, value: {infer_limit_batch_size}")
        if infer_limit is not None:
            if not isinstance(infer_limit, (int, float)):
                raise ValueError(f"infer_limit must be type int or float, but was instead type {type(infer_limit)}")
            if infer_limit <= 0:
                raise AssertionError(f"infer_limit must be greater than zero! (infer_limit={infer_limit})")
        if infer_limit is not None and infer_limit_batch_size is None:
            infer_limit_batch_size = 10000
            logger.log(20, f"infer_limit specified, but infer_limit_batch_size was not specified. Setting infer_limit_batch_size={infer_limit_batch_size}")
        return infer_limit, infer_limit_batch_size

    def _set_feature_generator(self, feature_generator="auto", feature_metadata=None, init_kwargs=None):
        if self._learner.feature_generator is not None:
            if isinstance(feature_generator, str) and feature_generator == "auto":
                feature_generator = self._learner.feature_generator
            else:
                raise AssertionError("FeatureGenerator already exists!")
        self._learner.feature_generator = get_default_feature_generator(
            feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=init_kwargs
        )

    def _sanitize_stack_args(
        self,
        num_bag_folds: int,
        num_bag_sets: int,
        num_stack_levels: int,
        num_train_rows: int,
        dynamic_stacking: bool | str,
        use_bag_holdout: bool | str,
        use_bag_holdout_was_auto: bool,
        dynamic_stacking_was_auto: bool,
    ):
        if not isinstance(num_bag_folds, int):
            raise ValueError(f"num_bag_folds must be an integer. (num_bag_folds={num_bag_folds})")
        if not isinstance(num_stack_levels, int):
            raise ValueError(f"num_stack_levels must be an integer. (num_stack_levels={num_stack_levels})")
        if num_bag_folds < 2 and num_bag_folds != 0:
            raise ValueError(f"num_bag_folds must be equal to 0 or >=2. (num_bag_folds={num_bag_folds})")
        if num_stack_levels != 0 and num_bag_folds == 0:
            raise ValueError(f"num_stack_levels must be 0 if num_bag_folds is 0. (num_stack_levels={num_stack_levels}, num_bag_folds={num_bag_folds})")
        if not isinstance(num_bag_sets, int):
            raise ValueError(f"num_bag_sets must be an integer. (num_bag_sets={num_bag_sets})")
        if not isinstance(dynamic_stacking, bool):
            raise ValueError(f"dynamic_stacking must be a bool. (dynamic_stacking={dynamic_stacking})")
        if not isinstance(use_bag_holdout, bool):
            raise ValueError(f"use_bag_holdout must be a bool. (use_bag_holdout={use_bag_holdout})")

        if use_bag_holdout_was_auto and num_bag_folds != 0:
            if use_bag_holdout:
                log_extra = f"Reason: num_train_rows >= {USE_BAG_HOLDOUT_AUTO_THRESHOLD}. (num_train_rows={num_train_rows})"
            else:
                log_extra = f"Reason: num_train_rows < {USE_BAG_HOLDOUT_AUTO_THRESHOLD}. (num_train_rows={num_train_rows})"
            logger.log(20, f"Setting use_bag_holdout from 'auto' to {use_bag_holdout}. {log_extra}")

        if dynamic_stacking and num_stack_levels < 1:
            log_extra_ds = f"Reason: Stacking is not enabled. (num_stack_levels={num_stack_levels})"
            if not dynamic_stacking_was_auto:
                logger.log(20, f"Forcing dynamic_stacking to False. {log_extra_ds}")
            dynamic_stacking = False
        elif dynamic_stacking_was_auto:
            if dynamic_stacking:
                log_extra_ds = f"Reason: Enable dynamic_stacking when use_bag_holdout is disabled. (use_bag_holdout={use_bag_holdout})"
            else:
                log_extra_ds = f"Reason: Skip dynamic_stacking when use_bag_holdout is enabled. (use_bag_holdout={use_bag_holdout})"
            logger.log(20, f"Setting dynamic_stacking from 'auto' to {dynamic_stacking}. {log_extra_ds}")

        return num_bag_folds, num_bag_sets, num_stack_levels, dynamic_stacking, use_bag_holdout

    # TODO: Add .delete() method to easily clean-up clones?
    #  Would need to be careful that user doesn't delete important things accidentally.
    # TODO: Add .save_zip() and load_zip() methods to pack and unpack artifacts into a single file to simplify deployment code?
    def clone(self, path: str, *, return_clone: bool = False, dirs_exist_ok: bool = False) -> str | "TabularPredictor":
        """
        Clone the predictor and all of its artifacts to a new location on local disk.
        This is ideal for use-cases where saving a snapshot of the predictor is desired before performing
        more advanced operations (such as fit_extra and refit_full).

        Parameters
        ----------
        path : str
            Directory path the cloned predictor will be saved to.
        return_clone : bool, default = False
            If True, returns the loaded cloned TabularPredictor object.
            If False, returns the local path to the cloned TabularPredictor object.
        dirs_exist_ok : bool, default = False
            If True, will clone the predictor even if the path directory already exists, potentially overwriting unrelated files.
            If False, will raise an exception if the path directory already exists and avoid performing the copy.

        Returns
        -------
        If return_clone == True, returns the loaded cloned TabularPredictor object.
        If return_clone == False, returns the local path to the cloned TabularPredictor object.

        """
        assert path != self.path, f"Cannot clone into the same directory as the original predictor! (path='{path}')"
        path_clone = shutil.copytree(src=self.path, dst=path, dirs_exist_ok=dirs_exist_ok)
        logger.log(
            30,
            f"Cloned {self.__class__.__name__} located in '{self.path}' to '{path_clone}'.\n"
            f'\tTo load the cloned predictor: predictor_clone = {self.__class__.__name__}.load(path="{path_clone}")',
        )
        return self.__class__.load(path=path_clone) if return_clone else path_clone

    def clone_for_deployment(self, path: str, *, model: str = "best", return_clone: bool = False, dirs_exist_ok: bool = False) -> str | "TabularPredictor":
        """
        Clone the predictor and all of its artifacts to a new location on local disk,
        then delete the clones artifacts unnecessary during prediction.
        This is ideal for use-cases where saving a snapshot of the predictor is desired before performing
        more advanced operations (such as fit_extra and refit_full).

        Note that the clone can no longer fit new models,
        and most functionality except for predict and predict_proba will no longer work.

        Identical to performing the following operations in order:

        predictor_clone = predictor.clone(path=path, return_clone=True, dirs_exist_ok=dirs_exist_ok)
        predictor_clone.delete_models(models_to_keep=model)
        predictor_clone.set_model_best(model=model, save_trainer=True)
        predictor_clone.save_space()

        Parameters
        ----------
        path : str
            Directory path the cloned predictor will be saved to.
        model : str, default = 'best'
            The model to use in the optimized predictor clone.
            All other unrelated models will be deleted to save disk space.
            Refer to the `models_to_keep` argument of `predictor.delete_models` for available options.
            Internally calls `predictor_clone.delete_models(models_to_keep=model)`
        return_clone : bool, default = False
            If True, returns the loaded cloned TabularPredictor object.
            If False, returns the local path to the cloned TabularPredictor object.
        dirs_exist_ok : bool, default = False
            If True, will clone the predictor even if the path directory already exists, potentially overwriting unrelated files.
            If False, will raise an exception if the path directory already exists and avoids performing the copy.

        Returns
        -------
        If return_clone == True, returns the loaded cloned TabularPredictor object.
        If return_clone == False, returns the local path to the cloned TabularPredictor object.
        """
        predictor_clone = self.clone(path=path, return_clone=True, dirs_exist_ok=dirs_exist_ok)
        if model == "best":
            model = predictor_clone.model_best
            logger.log(30, f"Clone: Keeping minimum set of models required to predict with best model '{model}'...")
        else:
            logger.log(30, f"Clone: Keeping minimum set of models required to predict with model '{model}'...")
        predictor_clone.delete_models(models_to_keep=model, dry_run=False)
        if isinstance(model, str) and model in predictor_clone.model_names(can_infer=True):
            predictor_clone.set_model_best(model=model, save_trainer=True)
        logger.log(
            30,
            f"Clone: Removing artifacts unnecessary for prediction. "
            f"NOTE: Clone can no longer fit new models, and most functionality except for predict and predict_proba will no longer work",
        )
        predictor_clone.save_space()
        return predictor_clone if return_clone else predictor_clone.path

    def simulation_artifact(self, test_data: pd.DataFrame = None) -> dict:
        """
        [Advanced] Computes and returns the necessary information to perform zeroshot HPO simulation.
        For a usage example, refer to https://github.com/autogluon/tabrepo/blob/main/examples/run_quickstart_from_scratch.py

        Parameters
        ----------
        test_data: pd.DataFrame, default = None
            The test data to predict with.
            If None, the keys `pred_proba_dict_test` and `y_test` will not be present in the output.

        Returns
        -------
        simulation_dict: dict
            The dictionary of information required for zeroshot HPO simulation.
            Keys are as follows:
                pred_proba_dict_val: Dictionary of model name to prediction probabilities (or predictions if regression) on the validation data
                pred_proba_dict_test: Dictionary of model name to prediction probabilities (or predictions if regression) on the test data
                y_val: Pandas Series of ground truth labels for the validation data (internal representation)
                y_test: Pandas Series of ground truth labels for the test data (internal representation)
                eval_metric: The string name of the evaluation metric (obtained via `predictor.eval_metric.name`)
                problem_type: The problem type (obtained via `predictor.problem_type`)
                problem_type_transform: The transformed (internal) problem type (obtained via `predictor._learner.label_cleaner.problem_type_transform,`)
                ordered_class_labels: The original class labels (`predictor._learner.label_cleaner.ordered_class_labels`)
                ordered_Class_labels_transformed: The transformed (internal) class labels (`predictor._learner.label_cleaner.ordered_class_labels_transformed`)
                num_classes: The number of internal classes (`self._learner.label_cleaner.num_classes`)
                label: The label column name (`predictor.label`)
        """
        models = self.model_names(can_infer=True)

        pred_proba_dict_test = None
        if self.can_predict_proba:
            pred_proba_dict_val = self.predict_proba_multi(inverse_transform=False, as_multiclass=False, models=models)
            if test_data is not None:
                pred_proba_dict_test = self.predict_proba_multi(test_data, inverse_transform=False, as_multiclass=False, models=models)
        else:
            pred_proba_dict_val = self.predict_multi(inverse_transform=False, models=models)
            if test_data is not None:
                pred_proba_dict_test = self.predict_multi(test_data, inverse_transform=False, models=models)

        val_data_source = "val" if self.has_val else "train"
        _, y_val = self.load_data_internal(data=val_data_source, return_X=False, return_y=True)
        if test_data is not None:
            y_test = test_data[self.label]
            y_test = self.transform_labels(y_test, inverse=False)
            test_info = dict(
                pred_proba_dict_test=pred_proba_dict_test,
                y_test=y_test,
            )
        else:
            test_info = dict()

        simulation_dict = dict(
            **test_info,
            pred_proba_dict_val=pred_proba_dict_val,
            y_val=y_val,
            eval_metric=self.eval_metric.name,
            problem_type=self.problem_type,
            problem_type_transform=self._learner.label_cleaner.problem_type_transform,
            ordered_class_labels=self._learner.label_cleaner.ordered_class_labels,
            ordered_class_labels_transformed=self._learner.label_cleaner.ordered_class_labels_transformed,
            num_classes=self._learner.label_cleaner.num_classes,
            label=self.label,
        )

        return simulation_dict

    @staticmethod
    def _check_if_hyperparameters_handle_text(hyperparameters: dict) -> bool:
        """Check if hyperparameters contain a model that supports raw text features as input"""
        models_in_hyperparameters = set()
        advanced_hyperparameter_format = is_advanced_hyperparameter_format(hyperparameters=hyperparameters)
        if advanced_hyperparameter_format:
            for key in hyperparameters:
                for m in hyperparameters[key]:
                    models_in_hyperparameters.add(m)
        else:
            for key in hyperparameters:
                models_in_hyperparameters.add(key)
        models_in_hyperparameters_raw_text_compatible = []
        model_key_to_cls_map = ag_model_registry.key_to_cls_map()
        for m in models_in_hyperparameters:
            if isinstance(m, str):
                # TODO: Technically the use of MODEL_TYPES here is a hack since we should derive valid types from trainer,
                #  but this is required prior to trainer existing.
                if m in model_key_to_cls_map:
                    m = model_key_to_cls_map[m]
                else:
                    continue
            if m._get_class_tags().get("handles_text", False):
                models_in_hyperparameters_raw_text_compatible.append(m)

        if models_in_hyperparameters_raw_text_compatible:
            return True
        else:
            return False

    @staticmethod
    def _validate_hyperparameters(hyperparameters: dict):
        """
        Verifies correctness of hyperparameters object.
        """
        valid_types = (list, dict)

        def _validate_hyperparameters_util(params: dict):
            assert isinstance(params, dict), f"`hyperparameters` must be a dict, but found: {type(params)}"
            for model_type in params:
                if not isinstance(params[model_type], valid_types):
                    extra_msg = ""
                    if isinstance(params[model_type], str) and params[model_type] == "GBMLarge":
                        if version.parse(__version__) >= version.parse("1.3.0"):
                            extra_msg = "\n" + get_hyperparameter_str_deprecation_msg()
                        else:
                            # Will log a deprecation warning downstream in trainer
                            continue
                    raise AssertionError(
                        f"Hyperparameters are incorrectly formatted, refer to the documentation for examples. "
                        f"`hyperparameters` key '{model_type}' has an unexpected value type."
                        f"\n\tvalid types: {valid_types}"
                        f"\n\tactual type:  {type(params[model_type])}"
                        f"\n\tactual value: {params[model_type]}"
                        f"{extra_msg}"
                    )

        advanced_hyperparameter_format = is_advanced_hyperparameter_format(hyperparameters=hyperparameters)
        if advanced_hyperparameter_format:
            for stack_level in hyperparameters:
                _validate_hyperparameters_util(params=hyperparameters[stack_level])
        else:
            _validate_hyperparameters_util(params=hyperparameters)

    def _sanitize_pseudo_data(self, pseudo_data: pd.DataFrame, name="pseudo_data") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        assert isinstance(pseudo_data, pd.DataFrame)
        if self.label not in pseudo_data.columns:
            raise ValueError(f"'{name}' does not contain the labeled column.")

        if self.sample_weight is not None:
            raise ValueError(f"Applying 'sample_weight' with {name} is not supported.")

        X_pseudo = pseudo_data.drop(columns=[self.label])
        y_pseudo_og = pseudo_data[self.label]
        X_pseudo = self._learner.transform_features(X_pseudo)
        y_pseudo = self._learner.label_cleaner.transform(y_pseudo_og)

        if np.isnan(y_pseudo.unique()).any():
            raise Exception(f"NaN was found in the label column for {name}." "Please ensure no NaN values in target column")
        return X_pseudo, y_pseudo, y_pseudo_og

    def _assert_is_fit(self, message_suffix: str = None):
        if not self.is_fit:
            error_message = "Predictor is not fit. Call `.fit` before calling"
            if message_suffix is None:
                error_message = f"{error_message} this method."
            else:
                error_message = f"{error_message} `.{message_suffix}`."
            raise AssertionError(error_message)


def _dystack(
    predictor: TabularPredictor,
    train_data: Union[str, pd.DataFrame],
    time_limit: float,
    ds_fit_kwargs: dict,
    ag_fit_kwargs: dict,
    ag_post_fit_kwargs: dict,
    holdout_data=Union[str, pd.DataFrame, None],
):
    """Perform a sub-fit of a TabularPredictor on the provided input data and arguments for a TabularPredictor. To perform the sub-fit, the `self._learner` is
    used for fitting and predicting. After the sub-fit, `self._learner` is reset to its original state and the results of the sub-fit is returned. The sub-fit's
    training and validation data depends on the arguments in ds_fit_kwargs and holdout_data specified at `fit()`."""
    holdout_frac = ds_fit_kwargs.get("holdout_frac", None)
    train_indices = ds_fit_kwargs.get("train_indices", None)
    if holdout_frac is not None:
        train_data, val_data = generate_train_test_split_combined(
            data=train_data,
            label=predictor.label,
            problem_type=predictor.problem_type,
            test_size=holdout_frac,
            random_state=42,
        )
    elif train_indices is not None:
        val_indices = ds_fit_kwargs.get("val_indices", None)
        val_data = train_data.iloc[val_indices]
        train_data = train_data.iloc[train_indices]
    elif holdout_data is not None:
        train_data = train_data
        val_data = holdout_data
    else:
        raise ValueError("Unsupported validation procedure during dynamic stacking!")

    set_logger_verbosity(verbosity=ag_fit_kwargs["verbosity"])
    learner_og = copy.deepcopy(predictor._learner)

    ag_fit_kwargs["X"] = train_data
    ag_fit_kwargs["time_limit"] = time_limit
    ds_fit_context = ds_fit_kwargs.get("ds_fit_context")
    clean_up_fits = ds_fit_kwargs.get("clean_up_fits")

    predictor._learner.set_contexts(path_context=ds_fit_context)
    logger.log(20, f"Running DyStack sub-fit ...")
    try:
        predictor._fit(ag_fit_kwargs=ag_fit_kwargs, ag_post_fit_kwargs=ag_post_fit_kwargs)
    except Exception as e:
        return False, None, e

    if not predictor.model_names():
        logger.log(20, f"Unable to determine stacked overfitting. AutoGluon's sub-fit did not successfully train any models!")
        stacked_overfitting = False
        ho_leaderboard = None
    else:
        leaderboard_kwargs = dict()
        if predictor.model_best in predictor.model_refit_map(inverse=True):
            leaderboard_kwargs = dict(refit_full=True, set_refit_score_to_parent=True)
        # Determine stacked overfitting
        ho_leaderboard = predictor.leaderboard(data=val_data, **leaderboard_kwargs).reset_index(drop=True)
        stacked_overfitting = check_stacked_overfitting_from_leaderboard(ho_leaderboard)

    del predictor._learner
    predictor._learner = learner_og

    if clean_up_fits:
        logger.log(20, f"Deleting DyStack predictor artifacts (clean_up_fits={clean_up_fits}) ...")
        shutil.rmtree(path=ds_fit_context)
    else:
        predictor._sub_fits.append(ds_fit_context)

    return stacked_overfitting, ho_leaderboard, None
