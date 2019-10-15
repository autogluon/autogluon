
from pandas import DataFrame
import pandas as pd
import copy

from tabular.ml.learner.abstract_learner import AbstractLearner
from tabular.ml.constants import LANGUAGE_MODEL
from tabular.ml.cleaner import Cleaner
from tabular.ml.label_cleaner import LabelCleaner
from tabular.ml.trainer.lm_trainer import LMTrainer
from tabular.ml.trainer.auto_trainer import AutoTrainer


# TODO: Take as input objective function, function takes y_test, y_pred_proba as inputs and outputs score
# TODO: Add functionality for advanced feature generators such as gl_code_matrix_generator (inter-row dependencies, apply to train differently than test, etc., can only run after train/test split, rerun for each cv fold)
# TODO: - Differentiate between advanced generators that require fit (stateful, gl_code_matrix) and those that do not (bucket label averaging in SCOT GC 2019)
# TODO: - Those that do not could be added to preprocessing function of model, but would then have to be recomputed on each model.
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class DefaultLearner(AbstractLearner):
    def __init__(self, path_context: str, label: str, submission_columns: list, feature_generator, threshold=100, problem_type=None, 
                 objective_func=None, is_trainer_present=False):
        super().__init__(path_context=path_context, label=label, submission_columns=submission_columns, feature_generator=feature_generator, 
                         threshold=threshold, problem_type=problem_type, objective_func=objective_func, is_trainer_present=is_trainer_present)
    
    def fit(self, X: DataFrame, X_test: DataFrame = None, sample=None, hyperparameter_tune=False, feature_prune=False, searcher=None, scheduler=None):
        """ Arguments:
            sample (int): subsample data by just using the first SAMPLE datapoints. 
        """
        X, y, X_test, y_test = self.general_data_processing(X, X_test, sample)
        feature_types_metadata = self.feature_generator.feature_types_metadata
        
        # TODO rework this when AbstractLearner is implemented
        if self.problem_type == LANGUAGE_MODEL:
            trainer = LMTrainer(path=self.model_context, problem_type=self.problem_type, feature_types_metadata=feature_types_metadata)
        else:
            trainer = AutoTrainer(path=self.model_context, problem_type=self.problem_type, objective_func=self.objective_func, 
                                  num_classes=self.label_cleaner.num_classes, feature_types_metadata=feature_types_metadata, low_memory=True,
                                  searcher=searcher,scheduler=scheduler)
        
        # TODO: Move generate_train_test_split into trainer itself?
        # TODO: Holdout set?
        if y_test is None:
            X_train, X_test, y_train, y_test = trainer.generate_train_test_split(X, y) # all pandas dataframes
        else:
            X_train = X
            y_train = y
        
        ##########
        # Uncomment below and comment original fit transform to test categorical feature data leakage in unseen data
        # X_train = self.feature_generator.fit_transform(X_train, banned_features=self.submission_columns)
        # X_test = self.feature_generator.transform(X_test)
        ##########
        
        self.trainer_path = trainer.path
        self.trainer_type = type(trainer)
        if self.objective_func is None:
            self.objective_func = trainer.objective_func
        
        self.save()
        trainer.train(X_train, X_test, y_train, y_test, hyperparameter_tune, feature_prune)
        self.save_trainer(trainer=trainer)
    
    def general_data_processing(self, X: DataFrame, X_test: DataFrame = None, sample=None):
        """ General data processing steps used for all models. """
        X = copy.deepcopy(X)
        # if self.problem_type != REGRESSION:
        #     X[self.label] = X[self.label].fillna('')
        # TODO (Nick): from original Grail code (it had an error for Regression tasks). I have replaced this by dropping all examples will missing labels below.  If this is no longer needed, delete.
        
        # Remove all examples with missing labels from this dataset:
        n = len(X)
        missinglabel_indicators = X[self.label].isna().tolist()
        missinglabel_inds = [i for i,j in enumerate(missinglabel_indicators) if j]
        if len(missinglabel_inds) > 0:
            print("Dropping %s (out of %s) training examples for which the label value in column '%s' is missing" % (len(missinglabel_inds),n, self.label))
        X = X.drop(missinglabel_inds, axis=0)
        
        if self.problem_type is None:
            self.problem_type = self.get_problem_type(X[self.label])
        self.cleaner = Cleaner.construct(problem_type=self.problem_type, label=self.label, threshold=self.threshold)
        X = self.cleaner.clean(X)
        
        ##########
        # Enable below for local testing
        # TODO: Random (stratified?) sample
        if sample is not None:
            X = X.head(sample)
            X = Cleaner.construct(problem_type=self.problem_type, label=self.label, threshold=10).clean(X=X)
            # if X_test is not None:
            #     X_test = X_test.head(sample)
        ##########
        
        self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=X[self.label])
        
        X, y = self.extract_label(X)
        if X_test is not None and self.label in X_test.columns:
            X_test, y_test = self.extract_label(X_test)
        else:
            y_test = None
            if X_test is not None:
                raise ValueError("Test data provided to fit() is only intended for validation and must have label column named: %s. Instead use predict() to get predictions for unlabeled test examples." % self.label)
        
        if X_test is not None:
            # Do this if working with SKLearn models, otherwise categorical features may perform very badly on the test set
            print('Performing general data processing with merged train & test data. Validation performance may not accurately reflect performance on new test data.')
            X_super = pd.concat([X, X_test], ignore_index=True)
            X_super = self.feature_generator.fit_transform(X_super, banned_features=self.submission_columns, drop_duplicates=False)
            X = X_super.head(len(X)).set_index(X.index)
            X_test = X_super.tail(len(X_test)).set_index(X_test.index)
            del X_super
        else:
            X = self.feature_generator.fit_transform(X, banned_features=self.submission_columns,drop_duplicates=False)
        
        return (X, y, X_test, y_test)

# TODO: issue can only call learner.fit() or learner.general_data_processing() once. Second time always produces:  AttributeError: 'DataFrame' object has no attribute 'unique'

