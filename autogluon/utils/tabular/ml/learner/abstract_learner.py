import datetime
import json
import logging
import os
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy import corrcoef
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, classification_report  # , roc_curve, auc
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error, median_absolute_error  # , max_error

from ..constants import BINARY, MULTICLASS, REGRESSION
from ..trainer.abstract_trainer import AbstractTrainer
from ..tuning.ensemble_selection import EnsembleSelection
from ..utils import get_pred_from_proba, get_leaderboard_pareto_frontier, infer_problem_type, augment_rare_classes
from ...data.label_cleaner import LabelCleaner, LabelCleanerMulticlassToBinary
from ...features.abstract_feature_generator import AbstractFeatureGenerator
from ...utils.loaders import load_pkl, load_pd
from ...utils.savers import save_pkl, save_pd, save_json
from ...metrics.classification_metrics import confusion_matrix


logger = logging.getLogger(__name__)


# TODO: - Semi-supervised learning
# TODO: - Minimize memory usage of DataFrames (convert int64 -> uint8 when possible etc.)
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
# TODO: Loading learner from S3 on Windows may cause issues due to os.path.sep
class AbstractLearner:
    learner_file_name = 'learner.pkl'
    learner_info_name = 'info.pkl'
    learner_info_json_name = 'info.json'

    def __init__(self, path_context: str, label: str, id_columns: list, feature_generator: AbstractFeatureGenerator, label_count_threshold=10,
                 problem_type=None, eval_metric=None, stopping_metric=None, is_trainer_present=False, random_seed=0):
        self.path, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)
        self.label = label
        self.submission_columns = id_columns
        self.threshold = label_count_threshold
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.stopping_metric = stopping_metric
        self.is_trainer_present = is_trainer_present
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
        self.random_seed = random_seed
        self.cleaner = None
        self.label_cleaner: LabelCleaner = None
        self.feature_generator: AbstractFeatureGenerator = feature_generator
        self.feature_generators = [self.feature_generator]

        self.trainer: AbstractTrainer = None
        self.trainer_type = None
        self.trainer_path = None
        self.reset_paths = False

        self.time_fit_total = None
        self.time_fit_preprocessing = None
        self.time_fit_training = None
        self.time_limit = None

        try:
            from .....version import __version__
            self.version = __version__
        except:
            self.version = None

    @property
    def class_labels(self):
        return self.label_cleaner.ordered_class_labels

    def set_contexts(self, path_context):
        self.path, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        model_context = path_context + 'models' + os.path.sep
        latest_model_checkpoint = model_context + 'model_checkpoint_latest.pointer'
        eval_result_path = model_context + 'eval_result.pkl'
        predictions_path = path_context + 'predictions.csv'
        save_path = path_context + self.learner_file_name
        return path_context, model_context, latest_model_checkpoint, eval_result_path, predictions_path, save_path

    def fit(self, X: DataFrame, X_val: DataFrame = None, **kwargs):
        self._validate_fit_input(X=X, X_val=X_val, **kwargs)
        return self._fit(X=X, X_val=X_val, **kwargs)

    def _fit(self, X: DataFrame, X_val: DataFrame = None, scheduler_options=None, hyperparameter_tune=False,
            feature_prune=False, holdout_frac=0.1, hyperparameters=None, verbosity=2):
        raise NotImplementedError

    # TODO: Add pred_proba_cache functionality as in predict()
    def predict_proba(self, X: DataFrame, model=None, as_pandas=False, as_multiclass=False, inverse_transform=True):
        X = self.transform_features(X)
        y_pred_proba = self.load_trainer().predict_proba(X, model=model)
        if inverse_transform:
            y_pred_proba = self.label_cleaner.inverse_transform_proba(y_pred_proba)
        if as_multiclass and (self.problem_type == BINARY):
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)
        if as_pandas:
            if self.problem_type == MULTICLASS or (as_multiclass and self.problem_type == BINARY):
                y_pred_proba = pd.DataFrame(data=y_pred_proba, columns=self.class_labels)
            else:
                y_pred_proba = pd.Series(data=y_pred_proba, name=self.label)
        return y_pred_proba

    # TODO: Add decorators for cache functionality, return core code to previous state
    # use_pred_cache to check for a cached prediction of rows, can dramatically speedup repeated runs
    # add_to_pred_cache will update pred_cache with new predictions
    def predict(self, X: DataFrame, model=None, as_pandas=False, use_pred_cache=False, add_to_pred_cache=False):
        pred_cache = None
        if use_pred_cache or add_to_pred_cache:
            try:
                pred_cache = load_pd.load(path=self.pred_cache_path, dtype=X[self.submission_columns].dtypes.to_dict())
            except Exception:
                pass

        if use_pred_cache and (pred_cache is not None):
            X_id = X[self.submission_columns]
            X_in_cache_with_pred = pd.merge(left=X_id.reset_index(), right=pred_cache, on=self.submission_columns).set_index('index')  # Will break if 'index' == self.label or 'index' in self.submission_columns
            X_cache_miss = X[~X.index.isin(X_in_cache_with_pred.index)]
            logger.log(20, f'Using cached predictions for {len(X_in_cache_with_pred)} out of {len(X)} rows, '
                           f'which have already been predicted previously. To make new predictions, set use_pred_cache=False')
        else:
            X_in_cache_with_pred = pd.DataFrame(data=None, columns=self.submission_columns + [self.label])
            X_cache_miss = X

        if len(X_cache_miss) > 0:
            y_pred_proba = self.predict_proba(X=X_cache_miss, model=model, inverse_transform=False)
            problem_type = self.label_cleaner.problem_type_transform or self.problem_type
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
            y_pred = self.label_cleaner.inverse_transform(pd.Series(y_pred))
            y_pred.index = X_cache_miss.index
        else:
            logger.debug('All rows found in cache, no need to load model')
            y_pred = X_in_cache_with_pred[self.label].values
            if as_pandas:
                y_pred = pd.Series(data=y_pred, name=self.label)
            return y_pred

        if add_to_pred_cache:
            X_id_with_y_pred = X_cache_miss[self.submission_columns].copy()
            X_id_with_y_pred[self.label] = y_pred
            if pred_cache is None:
                pred_cache = X_id_with_y_pred.drop_duplicates(subset=self.submission_columns).reset_index(drop=True)
            else:
                pred_cache = pd.concat([X_id_with_y_pred, pred_cache]).drop_duplicates(subset=self.submission_columns).reset_index(drop=True)
            save_pd.save(path=self.pred_cache_path, df=pred_cache)

        if len(X_in_cache_with_pred) > 0:
            y_pred = pd.concat([y_pred, X_in_cache_with_pred[self.label]]).reindex(X.index)

        y_pred = y_pred.values
        if as_pandas:
            y_pred = pd.Series(data=y_pred, name=self.label)
        return y_pred

    def _validate_fit_input(self, X: DataFrame, **kwargs):
        if self.label not in X.columns:
            raise KeyError(f"Label column '{self.label}' is missing from training data. Training data columns: {list(X.columns)}")

    def get_inputs_to_stacker(self, dataset=None, model=None, base_models: list = None, use_orig_features=True):
        if model is not None or base_models is not None:
            if model is not None and base_models is not None:
                raise AssertionError('Only one of `model`, `base_models` is allowed to be set.')

        trainer = self.load_trainer()
        if dataset is None:
            if trainer.bagged_mode:
                dataset_preprocessed = trainer.load_X_train()
                fit = True
            else:
                dataset_preprocessed = trainer.load_X_val()
                fit = False
        else:
            dataset_preprocessed = self.transform_features(dataset)
            fit = False
        if base_models is not None:
            dataset_preprocessed = trainer.get_inputs_to_stacker_v2(X=dataset_preprocessed, base_models=base_models, fit=fit, use_orig_features=use_orig_features)
        elif model is not None:
            base_models = list(trainer.model_graph.predecessors(model))
            dataset_preprocessed = trainer.get_inputs_to_stacker_v2(X=dataset_preprocessed, base_models=base_models, fit=fit, use_orig_features=use_orig_features)
            # Note: Below doesn't quite work here because weighted_ensemble has unique input format returned that isn't a DataFrame.
            # dataset_preprocessed = trainer.get_inputs_to_model(model=model_to_get_inputs_for, X=dataset_preprocessed, fit=fit)

        return dataset_preprocessed

    # TODO: Experimental, not integrated with core code, highly subject to change
    # TODO: Add X, y parameters -> Requires proper preprocessing of train data
    # X should be X_train from original fit call, if None then load saved X_train in trainer (if save_data=True)
    # y should be y_train from original fit call, if None then load saved y_train in trainer (if save_data=True)
    # Compresses bagged ensembles to a single model fit on 100% of the data.
    # Results in worse model quality (-), but much faster inference times (+++), reduced memory usage (+++), and reduced space usage (+++).
    # You must have previously called fit() with cache_data=True.
    def refit_single_full(self, models=None):
        X = None
        y = None
        if X is not None:
            if y is None:
                X, y = self.extract_label(X)
            X = self.transform_features(X)
            y = self.label_cleaner.transform(y)
        else:
            y = None
        trainer = self.load_trainer()
        return trainer.refit_single_full(X=X, y=y, models=models)

    # Fits _FULL models and links them in the stack so _FULL models only use other _FULL models as input during stacking
    # If model is specified, will fit all _FULL models that are ancestors of the provided model, automatically linking them.
    # If no model is specified, all models are refit and linked appropriately.
    def refit_ensemble_full(self, model='all'):
        trainer = self.load_trainer()
        return trainer.refit_ensemble_full(model=model)

    def fit_transform_features(self, X, y=None):
        for feature_generator in self.feature_generators:
            X = feature_generator.fit_transform(X, y)
        return X

    def transform_features(self, X):
        for feature_generator in self.feature_generators:
            X = feature_generator.transform(X)
        return X

    def score(self, X: DataFrame, y=None, model=None):
        if y is None:
            X, y = self.extract_label(X)
        X = self.transform_features(X)
        y = self.label_cleaner.transform(y)
        trainer = self.load_trainer()
        if self.problem_type == MULTICLASS:
            y = y.fillna(-1)
            if (not trainer.eval_metric_expects_y_pred) and (-1 in y.unique()):
                # log_loss / pac_score
                raise ValueError(f'Multiclass scoring with eval_metric=\'{self.eval_metric.name}\' does not support unknown classes.')
        return trainer.score(X=X, y=y, model=model)

    # Scores both learner and all individual models, along with computing the optimal ensemble score + weights (oracle)
    def score_debug(self, X: DataFrame, y=None, extra_info=False, compute_oracle=False, silent=False):
        if y is None:
            X, y = self.extract_label(X)
        X = self.transform_features(X)
        y = self.label_cleaner.transform(y)
        trainer = self.load_trainer()
        if self.problem_type == MULTICLASS:
            y = y.fillna(-1)
            if (not trainer.eval_metric_expects_y_pred) and (-1 in y.unique()):
                # log_loss / pac_score
                raise ValueError(f'Multiclass scoring with eval_metric=\'{self.eval_metric.name}\' does not support unknown classes.')

        scores = {}
        all_trained_models = trainer.get_model_names_all()
        all_trained_models_can_infer = trainer.get_model_names_all(can_infer=True)
        all_trained_models_original = all_trained_models.copy()
        model_pred_proba_dict, pred_time_test_marginal = trainer.get_model_pred_proba_dict(X=X, models=all_trained_models_can_infer, fit=False, record_pred_time=True)

        if compute_oracle:
            pred_probas = list(model_pred_proba_dict.values())
            ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=trainer.problem_type, metric=self.eval_metric)
            ensemble_selection.fit(predictions=pred_probas, labels=y, identifiers=None)
            oracle_weights = ensemble_selection.weights_
            oracle_pred_time_start = time.time()
            oracle_pred_proba_norm = [pred * weight for pred, weight in zip(pred_probas, oracle_weights)]
            oracle_pred_proba_ensemble = np.sum(oracle_pred_proba_norm, axis=0)
            oracle_pred_time = time.time() - oracle_pred_time_start
            model_pred_proba_dict['oracle_ensemble'] = oracle_pred_proba_ensemble
            pred_time_test_marginal['oracle_ensemble'] = oracle_pred_time
            all_trained_models.append('oracle_ensemble')

        for model_name, pred_proba in model_pred_proba_dict.items():
            if (trainer.problem_type == BINARY) and (self.problem_type == MULTICLASS):
                pred_proba = self.label_cleaner.inverse_transform_proba(pred_proba)  # FIXME: I think this doesn't work correctly, must use original y as well!
            if trainer.eval_metric_expects_y_pred:
                pred = get_pred_from_proba(y_pred_proba=pred_proba, problem_type=self.problem_type)
                scores[model_name] = self.eval_metric(y, pred)
            else:
                scores[model_name] = self.eval_metric(y, pred_proba)

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

        leaderboard_df = self.leaderboard(extra_info=extra_info, silent=silent)

        df_merged = pd.merge(df, leaderboard_df, on='model', how='left')
        df_merged = df_merged.sort_values(by=['score_test', 'pred_time_test', 'score_val', 'pred_time_val', 'model'], ascending=[False, True, False, True, False]).reset_index(drop=True)
        df_columns_lst = df_merged.columns.tolist()
        explicit_order = [
            'model',
            'score_test',
            'score_val',
            'pred_time_test',
            'pred_time_val',
            'fit_time',
            'pred_time_test_marginal',
            'pred_time_val_marginal',
            'fit_time_marginal',
            'stack_level',
            'can_infer',
        ]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_merged = df_merged[df_columns_new]

        return df_merged

    def get_pred_probas_models_and_time(self, X, trainer: AbstractTrainer, model_names):
        pred_probas_lst = []
        pred_probas_time_lst = []
        for model_name in model_names:
            model = trainer.load_model(model_name)
            time_start = time.time()
            pred_probas = trainer.pred_proba_predictions(models=[model], X=X)
            if (self.problem_type == MULTICLASS) and (not trainer.eval_metric_expects_y_pred):
                # Handles case where we need to add empty columns to represent classes that were not used for training
                pred_probas = [self.label_cleaner.inverse_transform_proba(pred_proba) for pred_proba in pred_probas]
            time_diff = time.time() - time_start
            pred_probas_lst += pred_probas
            pred_probas_time_lst.append(time_diff)
        return pred_probas_lst, pred_probas_time_lst

    def _remove_missing_labels(self, y_true, y_pred):
        """Removes missing labels and produces warning if any are found."""
        if self.problem_type == REGRESSION:
            non_missing_boolean_mask = [y is not None and not pd.isnull(y) for y in y_true]
        else:
            non_missing_boolean_mask = [y is not None and not pd.isnull(y) and y != '' for y in y_true]

        n_missing = len(non_missing_boolean_mask) - sum(non_missing_boolean_mask)
        if n_missing > 0:
            y_true = y_true[non_missing_boolean_mask]
            y_pred = y_pred[non_missing_boolean_mask]
            warnings.warn(f"There are {n_missing} (out of {len(y_true)}) evaluation datapoints for which the label is missing. "
                          f"AutoGluon removed these points from the evaluation, which thus may not be entirely representative. "
                          f"You should carefully study why there are missing labels in your evaluation data.")
        return y_true, y_pred

    # TODO: Refactor to be less brittle.
    # TODO: Instead take y_pred_proba as input, convert to y_pred when necessary. Treat y_pred_proba as y_pred for regression.
    #  This makes this function behave much nicer, currently confusion matrix is only able to be constructed when y_pred is supplied, and crashes when y_pred_proba is supplied.
    #  Therefore, it is impossible to get confusion matrix when eval_metric is log_loss. This shouldn't be the case.
    def evaluate(self, y_true, y_pred, silent=False, auxiliary_metrics=False, detailed_report=True, high_always_good=False):
        """ Evaluate predictions.
            Args:
                silent (bool): Should we print which metric is being used as well as performance.
                auxiliary_metrics (bool): Should we compute other (problem_type specific) metrics in addition to the default metric?
                detailed_report (bool): Should we computed more-detailed versions of the auxiliary_metrics? (requires auxiliary_metrics=True).
                high_always_good (bool): If True, this means higher values of returned metric are ALWAYS superior (so metrics like MSE should be returned negated)

            Returns single performance-value if auxiliary_metrics=False.
            Otherwise returns dict where keys = metrics, values = performance along each metric.
        """
        assert isinstance(y_true, (np.ndarray, pd.Series))
        assert isinstance(y_pred, (np.ndarray, pd.Series))  # TODO: Enable DataFrame for y_pred_proba

        # TODO: Consider removing _remove_missing_labels, this creates an inconsistency between how .score, .score_debug, and .evaluate compute scores.
        y_true, y_pred = self._remove_missing_labels(y_true, y_pred)
        performance = self.eval_metric(y_true, y_pred)

        metric = self.eval_metric.name

        if not high_always_good:
            performance = performance * self.eval_metric._sign  # flip negative once again back to positive (so higher is no longer necessarily better)

        if not silent:
            logger.log(20, f"Evaluation: {metric} on test data: {performance}")

        if not auxiliary_metrics:
            return performance

        # Otherwise compute auxiliary metrics:
        auxiliary_metrics = []
        if self.problem_type == REGRESSION:  # Adding regression metrics
            pearson_corr = lambda x, y: corrcoef(x, y)[0][1]
            pearson_corr.__name__ = 'pearson_correlation'
            auxiliary_metrics += [
                mean_absolute_error, explained_variance_score, r2_score, pearson_corr, mean_squared_error, median_absolute_error,
                # max_error
            ]
        else:  # Adding classification metrics
            auxiliary_metrics += [accuracy_score, balanced_accuracy_score, matthews_corrcoef]
            if self.problem_type == BINARY:  # binary-specific metrics
                # def auc_score(y_true, y_pred): # TODO: this requires y_pred to be probability-scores
                #     fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label)
                #   return auc(fpr, tpr)
                f1micro_score = lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
                f1micro_score.__name__ = f1_score.__name__
                auxiliary_metrics += [f1micro_score]  # TODO: add auc?
            # elif self.problem_type == MULTICLASS:  # multiclass metrics
            #     auxiliary_metrics += []  # TODO: No multi-class specific metrics for now. Include top-5, top-10 accuracy here.

        performance_dict = OrderedDict({metric: performance})
        for metric_function in auxiliary_metrics:
            if isinstance(metric_function, tuple):
                metric_function, metric_kwargs = metric_function
            else:
                metric_kwargs = None
            metric_name = metric_function.__name__
            if metric_name not in performance_dict:
                try:  # only compute auxiliary metrics which do not error (y_pred = class-probabilities may cause some metrics to error)
                    if metric_kwargs:
                        performance_dict[metric_name] = metric_function(y_true, y_pred, **metric_kwargs)
                    else:
                        performance_dict[metric_name] = metric_function(y_true, y_pred)
                except ValueError:
                    pass

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

    def submit_from_preds(self, X: DataFrame, y_pred_proba, save=True, save_proba=False):
        submission = X[self.submission_columns].copy()
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

        submission[self.label] = y_pred
        submission[self.label] = self.label_cleaner.inverse_transform(submission[self.label])

        if save:
            utcnow = datetime.datetime.utcnow()
            timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
            path_submission = self.model_context + 'submissions' + os.path.sep + 'submission_' + timestamp_str_now + '.csv'
            path_submission_proba = self.model_context + 'submissions' + os.path.sep + 'submission_proba_' + timestamp_str_now + '.csv'
            save_pd.save(path=path_submission, df=submission)
            if save_proba:
                submission_proba = pd.DataFrame(y_pred_proba)  # TODO: Fix for multiclass
                save_pd.save(path=path_submission_proba, df=submission_proba)

        return submission

    def predict_and_submit(self, X: DataFrame, save=True, save_proba=False):
        y_pred_proba = self.predict_proba(X=X, inverse_transform=False)
        return self.submit_from_preds(X=X, y_pred_proba=y_pred_proba, save=save, save_proba=save_proba)

    def leaderboard(self, X=None, y=None, extra_info=False, only_pareto_frontier=False, silent=False):
        if X is not None:
            leaderboard = self.score_debug(X=X, y=y, extra_info=extra_info, silent=True)
        else:
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
    def get_feature_importance(self, model=None, X=None, y=None, features: list = None, feature_stage='original', subsample_size=1000, silent=False) -> Series:
        valid_feature_stages = ['original', 'transformed', 'transformed_model']
        if feature_stage not in valid_feature_stages:
            raise ValueError(f'feature_stage must be one of: {valid_feature_stages}, but was {feature_stage}.')
        trainer = self.load_trainer()
        if X is not None:
            if y is None:
                X, y = self.extract_label(X)
            y = self.label_cleaner.transform(y)
            X, y = self._remove_nan_label_rows(X, y)

            if feature_stage == 'original':
                return trainer._get_feature_importance_raw(model=model, X=X, y=y, features_to_use=features, subsample_size=subsample_size, transform_func=self.transform_features, silent=silent)
            X = self.transform_features(X)
        else:
            if feature_stage == 'original':
                raise AssertionError('Feature importance `dataset` cannot be None if `feature_stage==\'original\'`. A test dataset must be specified.')
            y = None
        raw = feature_stage == 'transformed'
        return trainer.get_feature_importance(X=X, y=y, model=model, features=features, raw=raw, subsample_size=subsample_size, silent=silent)

    @staticmethod
    def _remove_nan_label_rows(X, y):
        if y.isnull().any():
            y = y.dropna()
            X = X.loc[y.index]
        return X, y

    # TODO: Add data info gathering at beginning of .fit() that is used by all learners to add to get_info output
    # TODO: Add feature inference / feature engineering info to get_info output
    def get_info(self, include_model_info=False):
        trainer = self.load_trainer()
        trainer_info = trainer.get_info(include_model_info=include_model_info)
        learner_info = {
            'path': self.path,
            'label': self.label,
            'time_fit_preprocessing': self.time_fit_preprocessing,
            'time_fit_training': self.time_fit_training,
            'time_fit_total': self.time_fit_total,
            'time_limit': self.time_limit,
            'random_seed': self.random_seed,
            'version': self.version,
        }

        learner_info.update(trainer_info)
        return learner_info

    @staticmethod
    def infer_problem_type(y: Series):
        return infer_problem_type(y=y)

    def save(self):
        save_pkl.save(path=self.save_path, object=self)

    # reset_paths=True if the learner files have changed location since fitting.
    # TODO: Potentially set reset_paths=False inside load function if it is the same path to avoid re-computing paths on all models
    # TODO: path_context -> path
    @classmethod
    def load(cls, path_context, reset_paths=True):
        load_path = path_context + cls.learner_file_name
        obj = load_pkl.load(path=load_path)
        if reset_paths:
            obj.set_contexts(path_context)
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
        if self.is_trainer_present:
            return self.trainer
        else:
            return self.trainer_type.load(path=self.trainer_path, reset_paths=self.reset_paths)

    # TODO: Add to predictor
    # TODO: Make this safe in large ensemble situations that would result in OOM
    # Loads all models in memory so that they don't have to loaded during predictions
    def persist_trainer(self, low_memory=False):
        self.trainer = self.load_trainer()
        self.is_trainer_present = True
        if not low_memory:
            self.trainer.load_models_into_memory()
            # Warning: After calling this, it is not necessarily safe to save learner or trainer anymore
            #  If neural network is persisted and then trainer or learner is saved, there will be an exception thrown

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

    def distill(self, X=None, y=None, X_val=None, y_val=None, time_limits=None, hyperparameters=None, holdout_frac=None,
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
        distilled_model_names = trainer.distill(X_train=X, y_train=y, X_val=X_val, y_val=y_val, time_limits=time_limits, hyperparameters=hyperparameters,
                                                holdout_frac=holdout_frac, verbosity=verbosity, teacher_preds=teacher_preds, models_name_suffix=models_name_suffix,
                                                augmentation_data=augmentation_data, augment_method=augment_method, augment_args=augment_args)
        self.save_trainer(trainer=trainer)
        return distilled_model_names
