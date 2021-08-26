import copy
import json
import logging
import os
import random
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import classification_report

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, AUTO_WEIGHT, BALANCE_WEIGHT
from autogluon.core.metrics import confusion_matrix, get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.core.utils import get_leaderboard_pareto_frontier, augment_rare_classes, extract_column, compute_weighted_metric
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl
from autogluon.core.utils import get_pred_from_proba, get_pred_from_proba_df, infer_problem_type
from autogluon.features.generators import PipelineFeatureGenerator

from ..trainer.abstract_trainer import AbstractTrainer
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerMulticlassToBinary

logger = logging.getLogger(__name__)


# TODO: - Semi-supervised learning
# TODO: - Minimize memory usage of DataFrames (convert int64 -> uint8 when possible etc.)
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
# TODO: Loading learner from S3 on Windows may cause issues due to os.path.sep
class AbstractLearner:
    learner_file_name = 'learner.pkl'
    learner_info_name = 'info.pkl'
    learner_info_json_name = 'info.json'

    def __init__(self, path_context: str, label: str, feature_generator: PipelineFeatureGenerator, ignored_columns: list = None, label_count_threshold=10,
                 problem_type=None, quantile_levels=None, eval_metric=None, positive_class=None, cache_data=True, is_trainer_present=False,
                 random_state=0, sample_weight=None, weight_evaluation=False, groups=None):
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)
        self.label = label
        self.ignored_columns = ignored_columns
        if self.ignored_columns is None:
            self.ignored_columns = []
        self.threshold = label_count_threshold
        self.problem_type = problem_type
        self.eval_metric = get_metric(eval_metric, self.problem_type, 'eval_metric')

        if self.problem_type == QUANTILE and quantile_levels is None:
            raise ValueError("if `problem_type='quantile'`, `quantile_levels` has to be specified")
        if isinstance(quantile_levels, float):
            quantile_levels = [quantile_levels]
        if isinstance(quantile_levels, Iterable):
            for quantile in quantile_levels:
                if quantile <= 0.0 or quantile >= 1.0:
                    raise ValueError("quantile values have to be non-negative and less than 1.0 (0.0 < q < 1.0). "
                                     "For example, 0.95 quantile = 95 percentile")
            quantile_levels = np.sort(np.array(quantile_levels))
        self.quantile_levels = quantile_levels

        self.cache_data = cache_data
        if not self.cache_data:
            logger.log(30, 'Warning: `cache_data=False` will disable or limit advanced functionality after training such as feature importance calculations. It is recommended to set `cache_data=True` unless you explicitly wish to not have the data saved to disk.')
        self.is_trainer_present = is_trainer_present
        if random_state is None:
            random_state = random.randint(0, 1000000)
        self.random_state = random_state
        self.cleaner = None
        self.label_cleaner: LabelCleaner = None
        self.feature_generator: PipelineFeatureGenerator = feature_generator

        self.trainer: AbstractTrainer = None
        self.trainer_type = None
        self.trainer_path = None
        self.reset_paths = False

        self._pre_X_rows = None
        self._post_X_rows = None
        self._positive_class = positive_class
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation
        self.groups = groups
        if sample_weight is not None and not isinstance(sample_weight, str):
            raise ValueError("sample_weight must be a string indicating the name of the column that contains sample weights. If you have a vector of sample weights, first add these as an extra column to your data.")
        if weight_evaluation and sample_weight is None:
            raise ValueError("Must specify sample_weight column if you specify weight_evaluation=True")
        if groups is not None and not isinstance(groups, str):
            raise ValueError('groups must be a string indicating the name of the column that contains the split groups. If you have a vector of split groups, first add these as an extra column to your data.')
        try:
            from ..version import __version__
            self.version = __version__
        except:
            self.version = None

    # TODO: Possibly rename to features_in or consider refactoring all feature_generators features_in -> features
    @property
    def features(self):
        return self.feature_generator.features_in

    @property
    def feature_metadata_in(self):
        return self.feature_generator.feature_metadata_in

    @property
    def feature_generators(self):
        return [self.feature_generator]

    @property
    def class_labels(self):
        return self.label_cleaner.ordered_class_labels

    @property
    def is_fit(self):
        return self.trainer_path is not None or self.trainer is not None

    @property
    def positive_class(self):
        """
        Returns the positive class name in binary classification. Useful for computing metrics such as F1 which require a positive and negative class.
        In binary classification, :class:`TabularPredictor.predict_proba()` returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        if not self.is_fit:
            if self._positive_class is not None:
                return self._positive_class
            raise AssertionError('Predictor must be fit to return positive_class.')
        if self.problem_type != BINARY:
            logger.warning(f"Warning: Attempted to retrieve positive class label in a non-binary problem. Positive class labels only exist in binary classification. Returning None instead. self.problem_type is '{self.problem_type}' but positive_class only exists for '{BINARY}'.")
            return None
        return self.label_cleaner.cat_mappings_dependent_var[1]

    def set_contexts(self, path_context):
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        model_context = path_context + 'models' + os.path.sep
        save_path = path_context + self.learner_file_name
        return path_context, model_context, save_path

    def fit(self, X: DataFrame, X_val: DataFrame = None, **kwargs):
        if self.is_fit:
            raise AssertionError('Learner is already fit.')
        self._validate_fit_input(X=X, X_val=X_val, **kwargs)
        return self._fit(X=X, X_val=X_val, **kwargs)

    def _fit(self, X: DataFrame, X_val: DataFrame = None, scheduler_options=None, hyperparameter_tune=False,
            feature_prune=False, holdout_frac=0.1, hyperparameters=None, verbosity=2):
        raise NotImplementedError

    def predict_proba(self, X: DataFrame, model=None, as_pandas=True, as_multiclass=True, inverse_transform=True):
        if as_pandas:
            X_index = copy.deepcopy(X.index)
        else:
            X_index = None
        y_pred_proba = self.load_trainer().predict_proba(self.transform_features(X), model=model)
        if inverse_transform:
            y_pred_proba = self.label_cleaner.inverse_transform_proba(y_pred_proba)
        if as_multiclass and (self.problem_type == BINARY):
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)
        if as_pandas:
            if self.problem_type == MULTICLASS or (as_multiclass and self.problem_type == BINARY):
                y_pred_proba = pd.DataFrame(data=y_pred_proba, columns=self.class_labels, index=X_index)
            elif self.problem_type == QUANTILE:
                y_pred_proba = pd.DataFrame(data=y_pred_proba, columns=self.quantile_levels, index=X_index)
            else:
                y_pred_proba = pd.Series(data=y_pred_proba, name=self.label, index=X_index)
        return y_pred_proba

    def predict(self, X: DataFrame, model=None, as_pandas=True):
        if as_pandas:
            X_index = copy.deepcopy(X.index)
        else:
            X_index = None
        y_pred_proba = self.predict_proba(X=X, model=model, as_pandas=False, as_multiclass=False, inverse_transform=False)
        problem_type = self.label_cleaner.problem_type_transform or self.problem_type
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
        if problem_type != QUANTILE:
            y_pred = self.label_cleaner.inverse_transform(pd.Series(y_pred))
            if as_pandas:
                y_pred.index = X_index
                y_pred.name = self.label
            else:
                y_pred = y_pred.values
        else:
            if as_pandas:
                y_pred = pd.DataFrame(data=y_pred, columns=self.quantile_levels, index=X_index)
        return y_pred

    def _validate_fit_input(self, X: DataFrame, **kwargs):
        if self.label not in X.columns:
            raise KeyError(f"Label column '{self.label}' is missing from training data. Training data columns: {list(X.columns)}")
        X_val = kwargs.get('X_val', None)
        self._validate_sample_weight(X, X_val)
        self._validate_groups(X, X_val)

    def _validate_sample_weight(self, X, X_val):
        if self.sample_weight is not None:
            if self.sample_weight in [AUTO_WEIGHT, BALANCE_WEIGHT]:
                prefix = f"Using predefined sample weighting strategy: {self.sample_weight}."
                if self.weight_evaluation:
                    prefix += " Warning: We do not recommend weight_evaluation=True with predefined sample weighting."
            else:
                if self.sample_weight not in X.columns:
                    raise KeyError(f"sample_weight column '{self.sample_weight}' is missing from training data. Training data columns: {list(X.columns)}")
                weight_vals = X[self.sample_weight]
                if weight_vals.isna().sum() > 0:
                    raise ValueError(f"Sample weights in column '{self.sample_weight}' cannot be nan")
                if weight_vals.dtype.kind not in 'biuf':
                    raise ValueError(f"Sample weights in column '{self.sample_weight}' must be numeric values")
                if weight_vals.min() < 0:
                    raise ValueError(f"Sample weights in column '{self.sample_weight}' must be nonnegative")
                if self.weight_evaluation and X_val is not None and self.sample_weight not in X_val.columns:
                    raise KeyError(f"sample_weight column '{self.sample_weight}' cannot be missing from validation data if weight_evaluation=True")
                prefix = f"Values in column '{self.sample_weight}' used as sample weights instead of predictive features."
            if self.weight_evaluation:
                suffix = " Evaluation will report weighted metrics, so ensure same column exists in test data."
            else:
                suffix = " Evaluation metrics will ignore sample weights, specify weight_evaluation=True to instead report weighted metrics."
            logger.log(20, prefix+suffix)

    def _validate_groups(self, X, X_val):
        if self.groups is not None:
            if self.groups not in X.columns:
                raise KeyError(f"groups column '{self.groups}' is missing from training data. Training data columns: {list(X.columns)}")
            groups_vals = X[self.groups]
            if len(groups_vals.unique()) < 2:
                raise ValueError(f"Groups in column '{self.groups}' cannot have fewer than 2 unique values. Values: {list(groups_vals.unique())}")
            if X_val is not None and self.groups in X_val.columns:
                raise KeyError(f"groups column '{self.groups}' cannot be in validation data. Validation data columns: {list(X_val.columns)}")
            logger.log(20, f"Values in column '{self.groups}' used as split folds instead of being automatically set. Bagged models will have {len(groups_vals.unique())} splits.")

    def get_inputs_to_stacker(self, dataset=None, model=None, base_models: list = None, use_orig_features=True):
        if model is not None or base_models is not None:
            if model is not None and base_models is not None:
                raise AssertionError('Only one of `model`, `base_models` is allowed to be set.')

        trainer = self.load_trainer()
        if dataset is None:
            if trainer.bagged_mode:
                dataset_preprocessed = trainer.load_X()
                fit = True
            else:
                dataset_preprocessed = trainer.load_X_val()
                fit = False
        else:
            dataset_preprocessed = self.transform_features(dataset)
            fit = False
        if base_models is not None:
            dataset_preprocessed = trainer.get_inputs_to_stacker(X=dataset_preprocessed, base_models=base_models, fit=fit, use_orig_features=use_orig_features)
        elif model is not None:
            base_models = list(trainer.model_graph.predecessors(model))
            dataset_preprocessed = trainer.get_inputs_to_stacker(X=dataset_preprocessed, base_models=base_models, fit=fit, use_orig_features=use_orig_features)
            # Note: Below doesn't quite work here because weighted_ensemble has unique input format returned that isn't a DataFrame.
            # dataset_preprocessed = trainer.get_inputs_to_model(model=model_to_get_inputs_for, X=dataset_preprocessed, fit=fit)

        return dataset_preprocessed

    # Fits _FULL models and links them in the stack so _FULL models only use other _FULL models as input during stacking
    # If model is specified, will fit all _FULL models that are ancestors of the provided model, automatically linking them.
    # If no model is specified, all models are refit and linked appropriately.
    def refit_ensemble_full(self, model='all'):
        return self.load_trainer().refit_ensemble_full(model=model)

    def fit_transform_features(self, X, y=None, **kwargs):
        if self.label in X:
            X = X.drop(columns=[self.label])
        if self.ignored_columns:
            logger.log(20, f'Dropping user-specified ignored columns: {self.ignored_columns}')
            X = X.drop(columns=self.ignored_columns, errors='ignore')
        for feature_generator in self.feature_generators:
            X = feature_generator.fit_transform(X, y, **kwargs)
        return X

    def transform_features(self, X):
        for feature_generator in self.feature_generators:
            X = feature_generator.transform(X)
        return X

    def score(self, X: DataFrame, y=None, model=None):
        if y is None:
            X, y = self.extract_label(X)
        self._validate_class_labels(y)
        w = None
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)
        if self.eval_metric.needs_pred:
            y_pred = self.predict(X=X, model=model, as_pandas=False)
            if self.problem_type == BINARY:
                # Use 1 and 0, otherwise f1 can crash due to unknown pos_label.
                y_pred = self.label_cleaner.transform(y_pred)
                y = self.label_cleaner.transform(y)
        elif self.eval_metric.needs_quantile:
            y_pred = self.predict(X=X, model=model, as_pandas=False)
        else:
            y_pred = self.predict_proba(X=X, model=model, as_pandas=False, as_multiclass=False)
            y = self.label_cleaner.transform(y)
        return compute_weighted_metric(y, y_pred, self.eval_metric, w, weight_evaluation=self.weight_evaluation, quantile_levels=self.quantile_levels)

    # Scores both learner and all individual models, along with computing the optimal ensemble score + weights (oracle)
    def score_debug(self, X: DataFrame, y=None, extra_info=False, compute_oracle=False, extra_metrics=None, silent=False):
        leaderboard_df = self.leaderboard(extra_info=extra_info, silent=silent)
        if y is None:
            X, y = self.extract_label(X)
        if extra_metrics is None:
            extra_metrics = []
        self._validate_class_labels(y)
        w = None
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)

        X = self.transform_features(X)
        y_internal = self.label_cleaner.transform(y)
        y_internal = y_internal.fillna(-1)

        trainer = self.load_trainer()
        scores = {}
        all_trained_models = trainer.get_model_names()
        all_trained_models_can_infer = trainer.get_model_names(can_infer=True)
        all_trained_models_original = all_trained_models.copy()
        model_pred_proba_dict, pred_time_test_marginal = trainer.get_model_pred_proba_dict(X=X, models=all_trained_models_can_infer, fit=False, record_pred_time=True)

        if compute_oracle:
            pred_probas = list(model_pred_proba_dict.values())
            ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=trainer.problem_type, metric=self.eval_metric, quantile_levels=self.quantile_levels)
            ensemble_selection.fit(predictions=pred_probas, labels=y_internal, identifiers=None, sample_weight=w)  # TODO: Only fit non-nan

            oracle_weights = ensemble_selection.weights_
            oracle_pred_time_start = time.time()
            oracle_pred_proba_norm = [pred * weight for pred, weight in zip(pred_probas, oracle_weights)]
            oracle_pred_proba_ensemble = np.sum(oracle_pred_proba_norm, axis=0)
            oracle_pred_time = time.time() - oracle_pred_time_start
            model_pred_proba_dict['OracleEnsemble'] = oracle_pred_proba_ensemble
            pred_time_test_marginal['OracleEnsemble'] = oracle_pred_time
            all_trained_models.append('OracleEnsemble')

        scoring_args = dict(
            y=y,
            y_internal=y_internal,
            sample_weight=w
        )

        extra_scores = {}
        for model_name, y_pred_proba_internal in model_pred_proba_dict.items():
            scores[model_name] = self._score_with_pred_proba(
                y_pred_proba_internal=y_pred_proba_internal,
                metric=self.eval_metric,
                **scoring_args
            )
            for metric in extra_metrics:
                metric = get_metric(metric, self.problem_type, 'leaderboard_metric')
                if metric.name not in extra_scores:
                    extra_scores[metric.name] = {}
                extra_scores[metric.name][model_name] = self._score_with_pred_proba(
                    y_pred_proba_internal=y_pred_proba_internal,
                    metric=metric,
                    **scoring_args
                )

        if extra_scores:
            series = []
            for metric in extra_scores:
                series.append(pd.Series(extra_scores[metric], name=metric))
            df_extra_scores = pd.concat(series, axis=1)
            extra_metrics_names = list(df_extra_scores.columns)
            df_extra_scores['model'] = df_extra_scores.index
            df_extra_scores = df_extra_scores.reset_index(drop=True)
        else:
            df_extra_scores = None
            extra_metrics_names = None

        pred_time_test = {}
        # TODO: Add support for calculating pred_time_test_full for oracle_ensemble, need to copy graph from trainer and add oracle_ensemble to it with proper edges.
        for model in model_pred_proba_dict.keys():
            if model in all_trained_models_original:
                base_model_set = trainer.get_minimum_model_set(model)
                if len(base_model_set) == 1:
                    pred_time_test[model] = pred_time_test_marginal[base_model_set[0]]
                else:
                    pred_time_test_full_num = 0
                    for base_model in base_model_set:
                        pred_time_test_full_num += pred_time_test_marginal[base_model]
                    pred_time_test[model] = pred_time_test_full_num
            else:
                pred_time_test[model] = None

        scored_models = list(scores.keys())
        for model in all_trained_models:
            if model not in scored_models:
                scores[model] = None
                pred_time_test[model] = None
                pred_time_test_marginal[model] = None

        logger.debug('Model scores:')
        logger.debug(str(scores))
        model_names_final = list(scores.keys())
        df = pd.DataFrame(
            data={
                'model': model_names_final,
                'score_test': list(scores.values()),
                'pred_time_test': [pred_time_test[model] for model in model_names_final],
                'pred_time_test_marginal': [pred_time_test_marginal[model] for model in model_names_final],
            }
        )
        if df_extra_scores is not None:
            df = pd.merge(df, df_extra_scores, on='model', how='left')

        df_merged = pd.merge(df, leaderboard_df, on='model', how='left')
        df_merged = df_merged.sort_values(by=['score_test', 'pred_time_test', 'score_val', 'pred_time_val', 'model'], ascending=[False, True, False, True, False]).reset_index(drop=True)
        df_columns_lst = df_merged.columns.tolist()
        explicit_order = [
            'model',
            'score_test',
        ]
        if extra_metrics_names is not None:
            explicit_order += extra_metrics_names
        explicit_order += [
            'score_val',
            'pred_time_test',
            'pred_time_val',
            'fit_time',
            'pred_time_test_marginal',
            'pred_time_val_marginal',
            'fit_time_marginal',
            'stack_level',
            'can_infer',
            'fit_order',
        ]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_merged = df_merged[df_columns_new]

        return df_merged

    def _score_with_pred_proba(self,
                               y,
                               y_internal,
                               y_pred_proba_internal,
                               metric,
                               sample_weight=None):
        metric = get_metric(metric, self.problem_type, 'leaderboard_metric')
        if metric.needs_pred:
            if self.problem_type == BINARY:
                # Use 1 and 0, otherwise f1 can crash due to unknown pos_label.
                y_pred = get_pred_from_proba(y_pred_proba_internal, problem_type=self.problem_type)
                y_tmp = y_internal
            else:
                y_pred = self.label_cleaner.inverse_transform_proba(y_pred_proba_internal, as_pred=True)
                y_tmp = y
        elif metric.needs_quantile:
            y_pred = self.label_cleaner.inverse_transform_proba(y_pred_proba_internal, as_pred=True)
            y_tmp = y
        else:
            y_pred = self.label_cleaner.inverse_transform_proba(y_pred_proba_internal, as_pred=False)
            y_tmp = y_internal
        return compute_weighted_metric(y_tmp, y_pred, metric, weights=sample_weight, weight_evaluation=self.weight_evaluation, quantile_levels=self.quantile_levels)

    def _score_with_pred(self,
                         y,
                         y_internal,
                         y_pred_internal,
                         metric,
                         sample_weight=None):
        metric = get_metric(metric, self.problem_type, 'leaderboard_metric')
        if self.problem_type == BINARY:
            # Use 1 and 0, otherwise f1 can crash due to unknown pos_label.
            y_pred = y_pred_internal
            y_tmp = y_internal
        else:
            y_pred = self.label_cleaner.inverse_transform(y_pred_internal)
            y_tmp = y
        return compute_weighted_metric(y_tmp, y_pred, metric, weights=sample_weight, weight_evaluation=self.weight_evaluation, quantile_levels=self.quantile_levels)

    def _validate_class_labels(self, y: Series):
        null_count = y.isnull().sum()
        if null_count:
            raise ValueError(f'Labels cannot contain missing (nan) values. Found {null_count} missing label values.')
        if self.problem_type == MULTICLASS and not self.eval_metric.needs_pred:
            y_unique = np.unique(y)
            valid_class_set = set(self.class_labels)
            unknown_classes = []
            for cls in y_unique:
                if cls not in valid_class_set:
                    unknown_classes.append(cls)
            if unknown_classes:
                # log_loss / pac_score
                raise ValueError(f'Multiclass scoring with eval_metric=\'{self.eval_metric.name}\' does not support unknown classes. Unknown classes: {unknown_classes}')

    def evaluate_predictions(self, y_true, y_pred, silent=False, auxiliary_metrics=True, detailed_report=False):
        """ Evaluate predictions. Does not support sample weights since this method reports a variety of metrics.
            Args:
                silent (bool): Should we print which metric is being used as well as performance.
                auxiliary_metrics (bool): Should we compute other (problem_type specific) metrics in addition to the default metric?
                detailed_report (bool): Should we computed more-detailed versions of the auxiliary_metrics? (requires auxiliary_metrics=True).

            Returns single performance-value if auxiliary_metrics=False.
            Otherwise returns dict where keys = metrics, values = performance along each metric.
        """

        if self.weight_evaluation:
            raise AssertionError('evaluate_predictions does not support `weight_evaluation=True`. Use `predictor.leaderboard` instead.')

        is_proba = False
        assert isinstance(y_true, (np.ndarray, pd.Series))
        assert isinstance(y_pred, (np.ndarray, pd.Series, pd.DataFrame))
        self._validate_class_labels(y_true)
        if isinstance(y_pred, np.ndarray):
            if self.problem_type == QUANTILE:
                y_pred = pd.DataFrame(data=y_pred, columns=self.quantile_levels)
            elif len(y_pred.shape) > 1:
                y_pred = pd.DataFrame(data=y_pred, columns=self.class_labels)

        if isinstance(y_pred, pd.DataFrame):
            is_proba = True
        elif not self.eval_metric.needs_pred:
            raise AssertionError(f'`evaluate_predictions` requires y_pred_proba input '
                                 f'when evaluating "{self.eval_metric.name}"... Please generate valid input via `predictor.predict_proba(data)`.\n'
                                 f'This may have occurred if you passed in predict input instead of predict_proba input, '
                                 f'or if you specified `as_multiclass=False` to `predictor.predict_proba(data, as_multiclass=False)`, '
                                 f'which is not supported by `evaluate_predictions`.')
        if is_proba:
            y_pred_proba = y_pred
            y_pred = get_pred_from_proba_df(y_pred_proba, problem_type=self.problem_type)
            if self.problem_type == BINARY:
                # roc_auc crashes if this isn't done
                y_pred_proba = y_pred_proba[self.positive_class]
        else:
            y_pred_proba = None
            y_pred = pd.Series(y_pred)
        if y_pred_proba is not None:
            y_pred_proba_internal = self.label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        else:
            y_pred_proba_internal = None
        y_true_internal = self.label_cleaner.transform(y_true)  # Get labels in numeric order
        y_true_internal = y_true_internal.fillna(-1)
        y_pred_internal = self.label_cleaner.transform(y_pred)  # Get labels in numeric order

        # Compute auxiliary metrics:
        auxiliary_metrics_lst = [self.eval_metric]
        performance_dict = {}

        if auxiliary_metrics:
            if self.problem_type == REGRESSION:  # Adding regression metrics
                auxiliary_metrics_lst += [
                    'root_mean_squared_error',
                    'mean_squared_error',
                    'mean_absolute_error',
                    'r2',
                    'pearsonr',
                    'median_absolute_error',
                ]
            if self.problem_type in [BINARY, MULTICLASS]:  # Adding classification metrics
                auxiliary_metrics_lst += [
                    'accuracy',
                    'balanced_accuracy',
                    # 'log_loss',  # Don't include as it probably adds more confusion to novice users (can be infinite)
                    'mcc',
                ]
            if self.problem_type == BINARY:  # binary-specific metrics
                auxiliary_metrics_lst += [
                    'roc_auc',
                    'f1',
                    'precision',
                    'recall',
                ]

        scoring_args = dict(
            y=y_true,
            y_internal=y_true_internal,
            # sample_weight=sample_weight,  # TODO: add sample_weight support
        )

        for aux_metric in auxiliary_metrics_lst:
            if isinstance(aux_metric, str):
                aux_metric = get_metric(metric=aux_metric, problem_type=self.problem_type, metric_type='aux_metric')
            if not aux_metric.needs_pred and y_pred_proba_internal is None:
                logger.log(15, f'Skipping {aux_metric.name} because no prediction probabilities are available to score.')
                continue

            if aux_metric.name not in performance_dict:
                if y_pred_proba_internal is not None:
                    score = self._score_with_pred_proba(
                        y_pred_proba_internal=y_pred_proba_internal,
                        metric=aux_metric,
                        **scoring_args
                    )
                else:
                    score = self._score_with_pred(
                        y_pred_internal=y_pred_internal,
                        metric=aux_metric,
                        **scoring_args
                    )
                performance_dict[aux_metric.name] = score

        if self.eval_metric.name in performance_dict:
            score_eval = performance_dict[self.eval_metric.name]
            score_eval_flipped = self.eval_metric.convert_score_to_sklearn_val(score_eval)  # flip negative once again back to positive (so higher is no longer necessarily better)
            if score_eval_flipped != score_eval:
                flipped = True
            else:
                flipped = False
            if not silent:
                logger.log(20, f"Evaluation: {self.eval_metric.name} on test data: {score_eval}")
                if flipped:
                    logger.log(20, f"\tNote: Scores are always higher_is_better. This metric score can be multiplied by -1 to get the metric value.")

        if not silent:
            logger.log(20, "Evaluations on test data:")
            logger.log(20, json.dumps(performance_dict, indent=4))

        if detailed_report and (self.problem_type != REGRESSION):
            # Construct confusion matrix
            try:
                performance_dict['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=self.label_cleaner.ordered_class_labels, output_format='pandas_dataframe')
            except ValueError:
                pass
            # One final set of metrics to report
            cl_metric = lambda y_true, y_pred: classification_report(y_true, y_pred, output_dict=True)
            metric_name = 'classification_report'
            if metric_name not in performance_dict:
                try:  # only compute auxiliary metrics which do not error (y_pred = class-probabilities may cause some metrics to error)
                    performance_dict[metric_name] = cl_metric(y_true, y_pred)
                except ValueError:
                    pass
                if not silent and metric_name in performance_dict:
                    logger.log(20, "Detailed (per-class) classification report:")
                    logger.log(20, json.dumps(performance_dict[metric_name], indent=4))
        return performance_dict

    def extract_label(self, X):
        if self.label not in list(X.columns):
            raise ValueError(f"Provided DataFrame does not contain label column: {self.label}")
        y = X[self.label].copy()
        X = X.drop(self.label, axis=1)
        return X, y

    def leaderboard(self, X=None, y=None, extra_info=False, extra_metrics=None, only_pareto_frontier=False, silent=False):
        if X is not None:
            leaderboard = self.score_debug(X=X, y=y, extra_info=extra_info, extra_metrics=extra_metrics, silent=True)
        else:
            if extra_metrics:
                raise AssertionError('`extra_metrics` is only valid when data is specified.')
            trainer = self.load_trainer()
            leaderboard = trainer.leaderboard(extra_info=extra_info)
        if only_pareto_frontier:
            if 'score_test' in leaderboard.columns and 'pred_time_test' in leaderboard.columns:
                score_col = 'score_test'
                inference_time_col = 'pred_time_test'
            else:
                score_col = 'score_val'
                inference_time_col = 'pred_time_val'
            leaderboard = get_leaderboard_pareto_frontier(leaderboard=leaderboard, score_col=score_col, inference_time_col=inference_time_col)
        if not silent:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(leaderboard)
        return leaderboard

    # TODO: cache_data must be set to True to be able to pass X and y as None in this function, otherwise it will error.
    # Warning: This can take a very, very long time to compute if the data is large and the model is complex.
    # A value of 0.01 means that the objective metric error would be expected to increase by 0.01 if the feature were removed.
    # Negative values mean the feature is likely harmful.
    # model: model (str) to get feature importances for, if None will choose best model.
    # features: list of feature names that feature importances are calculated for and returned, specify None to get all feature importances.
    # feature_stage: Whether to compute feature importance on raw original features ('original'), transformed features ('transformed') or on the features used by the particular model ('transformed_model').
    def get_feature_importance(self, model=None, X=None, y=None, features: list = None, feature_stage='original', subsample_size=1000, silent=False, **kwargs) -> DataFrame:
        valid_feature_stages = ['original', 'transformed', 'transformed_model']
        if feature_stage not in valid_feature_stages:
            raise ValueError(f'feature_stage must be one of: {valid_feature_stages}, but was {feature_stage}.')
        trainer = self.load_trainer()
        if X is not None:
            if y is None:
                X, y = self.extract_label(X)
            y = self.label_cleaner.transform(y)
            X, y = self._remove_nan_label_rows(X, y)
            if self.ignored_columns:
                X = X.drop(columns=self.ignored_columns, errors='ignore')

            if feature_stage == 'original':
                return trainer._get_feature_importance_raw(model=model, X=X, y=y, features=features, subsample_size=subsample_size, transform_func=self.transform_features, silent=silent, **kwargs)
            X = self.transform_features(X)
        else:
            if feature_stage == 'original':
                raise AssertionError('Feature importance `dataset` cannot be None if `feature_stage==\'original\'`. A test dataset must be specified.')
            y = None
        raw = feature_stage == 'transformed'
        return trainer.get_feature_importance(X=X, y=y, model=model, features=features, raw=raw, subsample_size=subsample_size, silent=silent, **kwargs)

    @staticmethod
    def _remove_nan_label_rows(X, y):
        if y.isnull().any():
            y = y.dropna()
            X = X.loc[y.index]
        return X, y

    @staticmethod
    def infer_problem_type(y: Series):
        return infer_problem_type(y=y)

    def save(self):
        trainer = None
        if self.trainer is not None:
            if not self.is_trainer_present:
                self.trainer.save()
                trainer = self.trainer
                self.trainer = None
        save_pkl.save(path=self.save_path, object=self)
        self.trainer = trainer

    # reset_paths=True if the learner files have changed location since fitting.
    # TODO: Potentially set reset_paths=False inside load function if it is the same path to avoid re-computing paths on all models
    # TODO: path_context -> path for v0.1
    @classmethod
    def load(cls, path_context, reset_paths=True):
        load_path = path_context + cls.learner_file_name
        obj = load_pkl.load(path=load_path)
        if reset_paths:
            obj.set_contexts(path_context)
            if obj.trainer_path is not None:
                obj.trainer_path = obj.model_context
            obj.reset_paths = reset_paths
            # TODO: Still have to change paths of models in trainer + trainer object path variables
            return obj
        else:
            obj.set_contexts(obj.path_context)
            return obj

    def save_trainer(self, trainer):
        if self.is_trainer_present:
            self.trainer = trainer
            self.save()
        else:
            self.trainer_path = trainer.path
            trainer.save()

    def load_trainer(self) -> AbstractTrainer:
        if self.trainer is not None:
            return self.trainer
        else:
            if self.trainer_path is None:
                raise AssertionError('Trainer does not exist.')
            return self.trainer_type.load(path=self.trainer_path, reset_paths=self.reset_paths)

    # Loads models in memory so that they don't have to be loaded during predictions
    def persist_trainer(self, low_memory=False, models='all', with_ancestors=False, max_memory=None) -> list:
        self.trainer = self.load_trainer()
        if not low_memory:
            return self.trainer.persist_models(models, with_ancestors=with_ancestors, max_memory=max_memory)
            # Warning: After calling this, it is not necessarily safe to save learner or trainer anymore
            #  If neural network is persisted and then trainer or learner is saved, there will be an exception thrown
        else:
            return []

    def distill(self, X=None, y=None, X_val=None, y_val=None, time_limit=None, hyperparameters=None, holdout_frac=None,
                verbosity=None, models_name_suffix=None, teacher_preds='soft',
                augmentation_data=None, augment_method='spunge', augment_args={'size_factor':5,'max_size':int(1e5)}):
        """ See abstract_trainer.distill() for details. """
        if X is not None:
            if (self.eval_metric is not None) and (self.eval_metric.name == 'log_loss') and (self.problem_type == MULTICLASS):
                X = augment_rare_classes(X, self.label, self.threshold)
            if y is None:
                X, y = self.extract_label(X)
            X = self.transform_features(X)
            y = self.label_cleaner.transform(y)
            if self.problem_type == MULTICLASS:
                y = y.fillna(-1)
        else:
            y = None

        if X_val is not None:
            if X is None:
                raise ValueError("Cannot specify X_val without specifying X")
            if y_val is None:
                X_val, y_val = self.extract_label(X_val)
            X_val = self.transform_features(X_val)
            y_val = self.label_cleaner.transform(y_val)

        if augmentation_data is not None:
            augmentation_data = self.transform_features(augmentation_data)

        trainer = self.load_trainer()
        distilled_model_names = trainer.distill(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit, hyperparameters=hyperparameters,
                                                holdout_frac=holdout_frac, verbosity=verbosity, teacher_preds=teacher_preds, models_name_suffix=models_name_suffix,
                                                augmentation_data=augmentation_data, augment_method=augment_method, augment_args=augment_args)
        self.save_trainer(trainer=trainer)
        return distilled_model_names

    @classmethod
    def load_info(cls, path, reset_paths=True, load_model_if_required=True):
        load_path = path + cls.learner_info_name
        try:
            return load_pkl.load(path=load_path)
        except Exception as e:
            if load_model_if_required:
                learner = cls.load(path_context=path, reset_paths=reset_paths)
                return learner.get_info()
            else:
                raise e

    def save_info(self, include_model_info=False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.learner_info_name, object=info)
        save_json.save(path=self.path + self.learner_info_json_name, obj=info)
        return info

    # TODO: Add data info gathering at beginning of .fit() that is used by all learners to add to get_info output
    # TODO: Add feature inference / feature engineering info to get_info output
    def get_info(self, **kwargs):
        learner_info = {
            'path': self.path,
            'label': self.label,
            'random_state': self.random_state,
            'version': self.version,
            'features': self.features,
            'feature_metadata_in': self.feature_metadata_in,
        }

        return learner_info
