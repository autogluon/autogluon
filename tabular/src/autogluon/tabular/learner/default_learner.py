import copy
import logging
import math
import platform
import time

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, AUTO_WEIGHT, BALANCE_WEIGHT
from autogluon.core.data import LabelCleaner
from autogluon.core.data.cleaner import Cleaner
from autogluon.core.utils.utils import augment_rare_classes, extract_column, time_func

from .abstract_learner import AbstractLearner
from ..trainer import AutoTrainer

logger = logging.getLogger(__name__)


# TODO: Add functionality for advanced feature generators such as gl_code_matrix_generator (inter-row dependencies, apply to train differently than test, etc., can only run after train/test split, rerun for each cv fold)
# TODO: - Differentiate between advanced generators that require fit (stateful, gl_code_matrix) and those that do not (bucket label averaging in SCOT GC 2019)
# TODO: - Those that do not could be added to preprocessing function of model, but would then have to be recomputed on each model.
# TODO: Add cv / OOF generator option, so that AutoGluon can be used as a base model in an ensemble stacker
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class DefaultLearner(AbstractLearner):
    def __init__(self, trainer_type=AutoTrainer, **kwargs):
        super().__init__(**kwargs)
        self.trainer_type = trainer_type
        self.class_weights = None
        self._time_fit_total = None
        self._time_fit_preprocessing = None
        self._time_fit_training = None
        self._time_limit = None
        self.preprocess_1_time = None  # Time required to preprocess 1 row of data

    # TODO: v0.1 Document trainer_fit_kwargs
    def _fit(self, X: DataFrame, X_val: DataFrame = None, X_unlabeled: DataFrame = None, holdout_frac=0.1,
             num_bag_folds=0, num_bag_sets=1, time_limit=None,
             infer_limit=None, infer_limit_batch_size=None,
             verbosity=2, **trainer_fit_kwargs):
        """ Arguments:
                X (DataFrame): training data
                X_val (DataFrame): data used for hyperparameter tuning. Note: final model may be trained using this data as well as training data
                X_unlabeled (DataFrame): data used for pretraining a model. This is same data format as X, without label-column. This data is used for semi-supervised learning.
                holdout_frac (float): Fraction of data to hold out for evaluating validation performance (ignored if X_val != None, ignored if kfolds != 0)
                num_bag_folds (int): kfolds used for bagging of models, roughly increases model training time by a factor of k (0: disabled)
                num_bag_sets (int): number of repeats of kfold bagging to perform (values must be >= 1),
                    total number of models trained during bagging = num_bag_folds * num_bag_sets
        """
        # TODO: if provided, feature_types in X, X_val are ignored right now, need to pass to Learner/trainer and update this documentation.
        self._time_limit = time_limit
        if time_limit:
            logger.log(20, f'Beginning AutoGluon training ... Time limit = {time_limit}s')
        else:
            logger.log(20, 'Beginning AutoGluon training ...')
        logger.log(20, f'AutoGluon will save models to "{self.path}"')
        logger.log(20, f'AutoGluon Version:  {self.version}')
        logger.log(20, f'Python Version:     {self._python_version}')
        logger.log(20, f'Operating System:   {platform.system()}')
        logger.log(20, f'Train Data Rows:    {len(X)}')
        logger.log(20, f'Train Data Columns: {len([column for column in X.columns if column != self.label])}')
        if X_val is not None:
            logger.log(20, f'Tuning Data Rows:    {len(X_val)}')
            logger.log(20, f'Tuning Data Columns: {len([column for column in X_val.columns if column != self.label])}')
        time_preprocessing_start = time.time()
        logger.log(20, 'Preprocessing data ...')
        self._pre_X_rows = len(X)
        if self.groups is not None:
            num_bag_sets = 1
            num_bag_folds = len(X[self.groups].unique())
        if infer_limit_batch_size is not None:
            X_og = X
        else:
            X_og = None
        X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups = self.general_data_processing(X, X_val, X_unlabeled, holdout_frac, num_bag_folds)
        if infer_limit_batch_size is not None:
            if infer_limit_batch_size >= self._pre_X_rows:
                infer_limit_batch_size_actual = self._pre_X_rows
                X_og_1 = X_og
            else:
                infer_limit_batch_size_actual = infer_limit_batch_size
                X_og_1 = X_og.head(infer_limit_batch_size_actual)
            self.preprocess_1_time = time_func(f=self.transform_features, args=[X_og_1]) / infer_limit_batch_size_actual
            logger.log(20, f'\t{round(self.preprocess_1_time, 4)}s\t= Feature Preprocessing Time (1 row | {infer_limit_batch_size} batch size)')

            if infer_limit is not None:
                infer_limit_new = infer_limit - self.preprocess_1_time
                logger.log(20, f'\t\tFeature Preprocessing requires {round(self.preprocess_1_time/infer_limit*100, 2)}% '
                               f'of the overall inference constraint ({infer_limit}s)\n'
                               f'\t\t{round(infer_limit_new, 4)}s inference time budget remaining for models...')
                if infer_limit_new <= 0:
                    raise AssertionError('Impossible to satisfy inference constraint, budget is exceeded during data preprocessing!\n'
                                         'Consider using fewer features, relaxing the inference constraint, or simplifying the feature generator.')
                infer_limit = infer_limit_new

        self._post_X_rows = len(X)
        time_preprocessing_end = time.time()
        self._time_fit_preprocessing = time_preprocessing_end - time_preprocessing_start
        logger.log(20, f'Data preprocessing and feature engineering runtime = {round(self._time_fit_preprocessing, 2)}s ...')
        if time_limit:
            time_limit_trainer = time_limit - self._time_fit_preprocessing
        else:
            time_limit_trainer = None

        trainer = self.trainer_type(
            path=self.model_context,
            problem_type=self.label_cleaner.problem_type_transform,
            eval_metric=self.eval_metric,
            num_classes=self.label_cleaner.num_classes,
            quantile_levels=self.quantile_levels,
            feature_metadata=self.feature_generator.feature_metadata,
            low_memory=True,
            k_fold=num_bag_folds,  # TODO: Consider moving to fit call
            n_repeats=num_bag_sets,  # TODO: Consider moving to fit call
            sample_weight=self.sample_weight,
            weight_evaluation=self.weight_evaluation,
            save_data=self.cache_data,
            random_state=self.random_state,
            verbosity=verbosity
        )

        self.trainer_path = trainer.path
        if self.eval_metric is None:
            self.eval_metric = trainer.eval_metric

        self.save()
        trainer.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            holdout_frac=holdout_frac,
            time_limit=time_limit_trainer,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            groups=groups,
            **trainer_fit_kwargs
        )
        self.save_trainer(trainer=trainer)
        time_end = time.time()
        self._time_fit_training = time_end - time_preprocessing_end
        self._time_fit_total = time_end - time_preprocessing_start
        logger.log(20, f'AutoGluon training complete, total runtime = {round(self._time_fit_total, 2)}s ... Best model: "{trainer.model_best}"')

    # TODO: Add default values to X_val, X_unlabeled, holdout_frac, and num_bag_folds
    def general_data_processing(self, X: DataFrame, X_val: DataFrame, X_unlabeled: DataFrame, holdout_frac: float, num_bag_folds: int):
        """ General data processing steps used for all models. """
        X = copy.deepcopy(X)

        # TODO: We should probably uncomment the below lines, NaN label should be treated as just another value in multiclass classification -> We will have to remove missing, compute problem type, and add back missing if multiclass
        # if self.problem_type == MULTICLASS:
        #     X[self.label] = X[self.label].fillna('')

        # Remove all examples with missing labels from this dataset:
        missinglabel_inds = [index for index, x in X[self.label].isna().iteritems() if x]
        if len(missinglabel_inds) > 0:
            logger.warning(f"Warning: Ignoring {len(missinglabel_inds)} (out of {len(X)}) training examples for which the label value in column '{self.label}' is missing")
            X = X.drop(missinglabel_inds, axis=0)

        if self.problem_type is None:
            self.problem_type = self.infer_problem_type(X[self.label])
            if self.quantile_levels is not None:
                if self.problem_type == REGRESSION:
                    self.problem_type = QUANTILE
                else:
                    raise ValueError("autogluon infers this to be classification problem for which quantile_levels "
                                     "cannot be specified. If it is truly a quantile regression problem, "
                                     "please specify:problem_type='quantile'")

        if X_val is not None and self.label in X_val.columns:
            holdout_frac = 1

        if (self.eval_metric is not None) and (self.eval_metric.name in ['log_loss', 'pac_score']) and (self.problem_type == MULTICLASS):
            if num_bag_folds > 0:
                self.threshold = 2
                if self.groups is None:
                    X = augment_rare_classes(X, self.label, threshold=2)
            else:
                self.threshold = 1

        self.threshold, holdout_frac, num_bag_folds = self.adjust_threshold_if_necessary(X[self.label], threshold=self.threshold, holdout_frac=holdout_frac, num_bag_folds=num_bag_folds)

        # Gets labels prior to removal of infrequent classes
        y_uncleaned = X[self.label].copy()

        self.cleaner = Cleaner.construct(problem_type=self.problem_type, label=self.label, threshold=self.threshold)
        X = self.cleaner.fit_transform(X)  # TODO: Consider merging cleaner into label_cleaner
        X, y = self.extract_label(X)
        self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y, y_uncleaned=y_uncleaned, positive_class=self._positive_class)
        y = self.label_cleaner.transform(y)
        X = self.set_predefined_weights(X, y)
        X, w = extract_column(X, self.sample_weight)
        X, groups = extract_column(X, self.groups)
        if self.label_cleaner.num_classes is not None and self.problem_type != BINARY:
            logger.log(20, f'Train Data Class Count: {self.label_cleaner.num_classes}')

        if X_val is not None and self.label in X_val.columns:
            X_val = self.cleaner.transform(X_val)
            if len(X_val) == 0:
                logger.warning('All X_val data contained low frequency classes, ignoring X_val and generating from subset of X')
                X_val = None
                y_val = None
                w_val = None
            else:
                X_val, y_val = self.extract_label(X_val)
                y_val = self.label_cleaner.transform(y_val)
                X_val = self.set_predefined_weights(X_val, y_val)
                X_val, w_val = extract_column(X_val, self.sample_weight)
        else:
            y_val = None
            w_val = None

        # TODO: Move this up to top of data before removing data, this way our feature generator is better
        logger.log(20, f'Using Feature Generators to preprocess the data ...')
        if X_val is not None:
            # Do this if working with SKLearn models, otherwise categorical features may perform very badly on the test set
            logger.log(15, 'Performing general data preprocessing with merged train & validation data, so validation performance may not accurately reflect performance on new test data')
            X_super = pd.concat([X, X_val, X_unlabeled], ignore_index=True)
            if self.feature_generator.is_fit():
                logger.log(20, f'{self.feature_generator.__class__.__name__} is already fit, so the training data will be processed via .transform() instead of .fit_transform().')
                X_super = self.feature_generator.transform(X_super)
                self.feature_generator.print_feature_metadata_info()
            else:
                if X_unlabeled is None:
                    y_super = pd.concat([y, y_val], ignore_index=True)
                else:
                    y_unlabeled = pd.Series(np.nan, index=X_unlabeled.index)
                    y_super = pd.concat([y, y_val, y_unlabeled], ignore_index=True)
                X_super = self.fit_transform_features(X_super, y_super, problem_type=self.label_cleaner.problem_type_transform, eval_metric=self.eval_metric)
            X = X_super.head(len(X)).set_index(X.index)

            X_val = X_super.head(len(X)+len(X_val)).tail(len(X_val)).set_index(X_val.index)

            if X_unlabeled is not None:
                X_unlabeled = X_super.tail(len(X_unlabeled)).set_index(X_unlabeled.index)
            del X_super
        else:
            X_super = pd.concat([X, X_unlabeled], ignore_index=True)
            if self.feature_generator.is_fit():
                logger.log(20, f'{self.feature_generator.__class__.__name__} is already fit, so the training data will be processed via .transform() instead of .fit_transform().')
                X_super = self.feature_generator.transform(X_super)
                self.feature_generator.print_feature_metadata_info()
            else:
                if X_unlabeled is None:
                    y_super = y.reset_index(drop=True)
                else:
                    y_unlabeled = pd.Series(np.nan, index=X_unlabeled.index)
                    y_super = pd.concat([y, y_unlabeled], ignore_index=True)
                X_super = self.fit_transform_features(X_super, y_super, problem_type=self.label_cleaner.problem_type_transform, eval_metric=self.eval_metric)

            X = X_super.head(len(X)).set_index(X.index)
            if X_unlabeled is not None:
                X_unlabeled = X_super.tail(len(X_unlabeled)).set_index(X_unlabeled.index)
            del X_super
        X, X_val = self.bundle_weights(X, w, X_val, w_val)  # TODO: consider not bundling sample-weights inside X, X_val
        return X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups

    def bundle_weights(self, X, w, X_val, w_val):
        if w is not None:
            X[self.sample_weight] = w
            if X_val is not None:
                if w_val is not None:
                    X_val[self.sample_weight] = w_val
                elif not self.weight_evaluation:
                    nan_vals = np.empty((len(X_val),))
                    nan_vals[:] = np.nan
                    X_val[self.sample_weight] = nan_vals
                else:
                    raise ValueError(f"sample_weight column '{self.sample_weight}' cannot be missing from X_val if weight_evaluation=True")
        return X, X_val

    def set_predefined_weights(self, X, y):
        if self.sample_weight not in [AUTO_WEIGHT,BALANCE_WEIGHT] or self.problem_type not in [BINARY,MULTICLASS]:
            return X
        if self.sample_weight in X.columns:
            raise ValueError(f"Column name '{self.sample_weight}' cannot appear in your dataset with predefined weighting strategy. Please change it and try again.")
        if self.sample_weight == BALANCE_WEIGHT:
            if self.class_weights is None:
                class_counts = y.value_counts()
                n = len(y)
                k = len(class_counts)
                self.class_weights = {c : n/(class_counts[c]*k) for c in class_counts.index}
                logger.log(20, "Assigning sample weights to balance differences in frequency of classes.")
                logger.log(15, f"Balancing classes via the following weights: {self.class_weights}")
            w = y.map(self.class_weights)
        elif self.sample_weight == AUTO_WEIGHT:  # TODO: support more sophisticated auto_weight strategy
            raise NotImplementedError(f"{AUTO_WEIGHT} strategy not yet supported.")
        X[self.sample_weight] = w  # TODO: consider not bundling sample weights inside X
        return X

    def adjust_threshold_if_necessary(self, y, threshold, holdout_frac, num_bag_folds):
        new_threshold, new_holdout_frac, new_num_bag_folds = self._adjust_threshold_if_necessary(y, threshold, holdout_frac, num_bag_folds)
        if new_threshold != threshold:
            if new_threshold < threshold:
                logger.warning(f'Warning: Updated label_count_threshold from {threshold} to {new_threshold} to avoid cutting too many classes.')
        if new_holdout_frac != holdout_frac:
            if new_holdout_frac > holdout_frac:
                logger.warning(f'Warning: Updated holdout_frac from {holdout_frac} to {new_holdout_frac} to avoid cutting too many classes.')
        if new_num_bag_folds != num_bag_folds:
            logger.warning(f'Warning: Updated num_bag_folds from {num_bag_folds} to {new_num_bag_folds} to avoid cutting too many classes.')
        return new_threshold, new_holdout_frac, new_num_bag_folds

    def _adjust_threshold_if_necessary(self, y, threshold, holdout_frac, num_bag_folds):
        new_threshold = threshold
        num_rows = len(y)
        holdout_frac = max(holdout_frac, 1 / num_rows + 0.001)
        num_bag_folds = min(num_bag_folds, num_rows)

        if num_bag_folds < 2:
            minimum_safe_threshold = 1
        else:
            minimum_safe_threshold = 2

        if minimum_safe_threshold > new_threshold:
            new_threshold = minimum_safe_threshold

        if self.problem_type in [REGRESSION, QUANTILE]:
            return new_threshold, holdout_frac, num_bag_folds

        class_counts = y.value_counts()
        total_rows = class_counts.sum()
        minimum_percent_to_keep = 0.975
        minimum_rows_to_keep = math.ceil(total_rows * minimum_percent_to_keep)
        minimum_class_to_keep = 2

        num_classes = len(class_counts)
        class_counts_valid = class_counts[class_counts >= new_threshold]
        num_rows_valid = class_counts_valid.sum()
        num_classes_valid = len(class_counts_valid)

        if (num_rows_valid >= minimum_rows_to_keep) and (num_classes_valid >= minimum_class_to_keep):
            return new_threshold, holdout_frac, num_bag_folds

        num_classes_valid = 0
        num_rows_valid = 0
        new_threshold = None
        for i in range(num_classes):
            num_classes_valid += 1
            num_rows_valid += class_counts.iloc[i]
            new_threshold = class_counts.iloc[i]
            if (num_rows_valid >= minimum_rows_to_keep) and (num_classes_valid >= minimum_class_to_keep):
                break

        return new_threshold, holdout_frac, num_bag_folds

    def get_info(self, include_model_info=False, **kwargs):
        learner_info = super().get_info(**kwargs)
        trainer = self.load_trainer()
        trainer_info = trainer.get_info(include_model_info=include_model_info)
        learner_info.update({
            'time_fit_preprocessing': self._time_fit_preprocessing,
            'time_fit_training': self._time_fit_training,
            'time_fit_total': self._time_fit_total,
            'time_limit': self._time_limit,
        })

        learner_info.update(trainer_info)
        return learner_info
