import copy, logging
from pandas import DataFrame
import pandas as pd

from .abstract_learner import AbstractLearner
from ...data.cleaner import Cleaner
from ...data.label_cleaner import LabelCleaner
from ..trainer.auto_trainer import AutoTrainer
from ..constants import BINARY

logger = logging.getLogger(__name__)


# TODO: Add functionality for advanced feature generators such as gl_code_matrix_generator (inter-row dependencies, apply to train differently than test, etc., can only run after train/test split, rerun for each cv fold)
# TODO: - Differentiate between advanced generators that require fit (stateful, gl_code_matrix) and those that do not (bucket label averaging in SCOT GC 2019)
# TODO: - Those that do not could be added to preprocessing function of model, but would then have to be recomputed on each model.
# TODO: Add cv / OOF generator option, so that AutoGluon can be used as a base model in an ensemble stacker
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class DefaultLearner(AbstractLearner):
    def __init__(self, path_context: str, label: str, id_columns: list, feature_generator, label_count_threshold=10,
                 problem_type=None, objective_func=None, is_trainer_present=False, trainer_type=AutoTrainer):
        super().__init__(path_context=path_context, label=label, id_columns=id_columns, feature_generator=feature_generator, label_count_threshold=label_count_threshold, 
            problem_type=problem_type, objective_func=objective_func, is_trainer_present=is_trainer_present)
        self.random_state = 0  # TODO: Add as input param
        self.trainer_type = trainer_type

    def fit(self, X: DataFrame, X_test: DataFrame = None, scheduler_options=None, hyperparameter_tune=True,
            feature_prune=False, holdout_frac=0.1, num_bagging_folds=0, stack_ensemble_levels=0, 
            hyperparameters= {'NN': {'num_epochs': 300}, 'GBM': {'num_boost_round': 10000}}, verbosity=2):
        """ Arguments:
                X (DataFrame): training data
                X_test (DataFrame): data used for hyperparameter tuning. Note: final model may be trained using this data as well as training data
                hyperparameter_tune (bool): whether to tune hyperparameters or simply use default values
                feature_prune (bool): whether to perform feature selection
                scheduler_options (tuple: (search_strategy, dict): Options for scheduler
                holdout_frac (float): Fraction of data to hold out for evaluating validation performance (ignored if X_test != None, ignored if kfolds != 0)
                num_bagging_folds (int): kfolds used for bagging of models, roughly increases model training time by a factor of k (0: disabled)
                stack_ensemble_levels : (int) Number of stacking levels to use in ensemble stacking. Roughly increases model training time by factor of stack_levels+1 (0: disabled)
                    Default is 0 (disabled). Use values between 1-3 to improve model quality.
                    Ignored unless kfolds is also set >= 2
                hyperparameters (dict): keys = hyperparameters + search-spaces for each type of model we should train.
        """
        # TODO: if provided, feature_types in X, X_test are ignored right now, need to pass to Learner/trainer and update this documentation.
        X, y, X_test, y_test = self.general_data_processing(X, X_test)

        trainer = self.trainer_type(
            path=self.model_context,
            problem_type=self.trainer_problem_type,
            objective_func=self.objective_func,
            num_classes=self.label_cleaner.num_classes,
            feature_types_metadata=self.feature_generator.feature_types_metadata,
            low_memory=True,
            kfolds=num_bagging_folds,
            stack_ensemble_levels=stack_ensemble_levels,
            scheduler_options=scheduler_options,
            verbosity = verbosity
        )

        self.trainer_path = trainer.path
        if self.objective_func is None:
            self.objective_func = trainer.objective_func

        self.save()
        trainer.train(X, y, X_test, y_test, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, holdout_frac=holdout_frac,
                      hyperparameters=hyperparameters)
        self.save_trainer(trainer=trainer)

    def general_data_processing(self, X: DataFrame, X_test: DataFrame = None):
        """ General data processing steps used for all models. """
        X = copy.deepcopy(X)
        # TODO: We should probably uncomment the below lines, NaN label should be treated as just another value in multiclass classification -> We will have to remove missing, compute problem type, and add back missing if multiclass
        # if self.problem_type == MULTICLASS:
        #     X[self.label] = X[self.label].fillna('')
        # TODO(Nick): from original Grail code (it had an error for Regression tasks). I have replaced this by dropping all examples will missing labels below.  If this is no longer needed, delete.

        # Remove all examples with missing labels from this dataset:
        n = len(X)
        missinglabel_indicators = X[self.label].isna().tolist()
        missinglabel_inds = [i for i,j in enumerate(missinglabel_indicators) if j]
        if len(missinglabel_inds) > 0:
            logger.warning("Warning: Ignoring %s (out of %s) training examples for which the label value in column '%s' is missing" % (len(missinglabel_inds),n, self.label))
        X = X.drop(missinglabel_inds, axis=0)

        if self.problem_type is None:
            self.problem_type = self.get_problem_type(X[self.label])

        # Gets labels prior to removal of infrequent classes
        y_uncleaned = X[self.label].copy()  # .astype('category').cat.categories

        self.cleaner = Cleaner.construct(problem_type=self.problem_type, label=self.label, threshold=self.threshold)
        # TODO: Most models crash if it is a multiclass problem with only two labels after thresholding, switch to being binary if this happens. Convert output from trainer to multiclass output preds in learner
        # TODO: What if all classes in X are low frequency in multiclass? Currently we would crash. Not certain how many problems actually have this property
        X = self.cleaner.fit_transform(X)  # TODO: Consider merging cleaner into label_cleaner
        self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=X[self.label], y_uncleaned=y_uncleaned)
        if (self.label_cleaner.num_classes is not None) and (self.label_cleaner.num_classes == 2):
            self.trainer_problem_type = BINARY
        else:
            self.trainer_problem_type = self.problem_type

        X, y = self.extract_label(X)
        y = self.label_cleaner.transform(y)

        if X_test is not None and self.label in X_test.columns:
            X_test = self.cleaner.transform(X_test)
            if len(X_test) == 0:
                logger.debug('All X_test data contained low frequency classes, ignoring X_test and generating from subset of X')
                X_test = None
                y_test = None
            else:
                X_test, y_test = self.extract_label(X_test)
                y_test = self.label_cleaner.transform(y_test)
        else:
            y_test = None

        # TODO: Move this up to top of data before removing data, this way our feature generator is better
        if X_test is not None:
            # Do this if working with SKLearn models, otherwise categorical features may perform very badly on the test set
            logger.log(15, 'Performing general data preprocessing with merged train & valiation data, so validation performance may not accurately reflect performance on new test data')
            X_super = pd.concat([X, X_test], ignore_index=True)
            X_super = self.feature_generator.fit_transform(X_super, banned_features=self.submission_columns, drop_duplicates=False)
            X = X_super.head(len(X)).set_index(X.index)
            X_test = X_super.tail(len(X_test)).set_index(X_test.index)
            del X_super
        else:
            X = self.feature_generator.fit_transform(X, banned_features=self.submission_columns, drop_duplicates=False)

        return X, y, X_test, y_test
