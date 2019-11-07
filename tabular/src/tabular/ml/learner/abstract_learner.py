from collections import OrderedDict 
import datetime, json, copy, warnings
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, classification_report # , roc_curve, auc
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error, median_absolute_error # , max_error
import numpy as np
from numpy import corrcoef

from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.ml.label_cleaner import LabelCleaner
from tabular.ml.utils import get_pred_from_proba
from tabular.utils.loaders import load_pkl, load_pd
from tabular.utils.savers import save_pkl, save_pd
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.tuning.ensemble_selection import EnsembleSelection


# TODO: Take as input objective function, function takes y_test, y_pred_proba as inputs and outputs score
# TODO: Add functionality for advanced feature generators such as gl_code_matrix_generator (inter-row dependencies, apply to train differently than test, etc., can only run after train/test split, rerun for each cv fold)
# TODO: - Differentiate between advanced generators that require fit (stateful, gl_code_matrix) and those that do not (bucket label averaging in SCOT GC 2019)
# TODO: - Those that do not could be added to preprocessing function of model, but would then have to be recomputed on each model.
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class AbstractLearner:
    save_file_name = 'learner.pkl'

    def __init__(self, path_context: str, label: str, submission_columns: list, feature_generator, threshold=10, problem_type=None, objective_func=None, is_trainer_present=False, compute_feature_importance=False):
        self.path_context, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)

        self.label = label
        self.submission_columns = submission_columns

        self.threshold = threshold
        self.problem_type = problem_type
        self.objective_func = objective_func
        self.is_trainer_present = is_trainer_present
        self.cleaner = None
        self.label_cleaner: LabelCleaner = None
        self.feature_generator = feature_generator

        self.trainer: AbstractTrainer = None
        self.trainer_type = None
        self.trainer_path = None
        self.reset_paths = False

        self.compute_feature_importance = compute_feature_importance

    def set_contexts(self, path_context):
        self.path_context, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        model_context = path_context + 'models/'
        latest_model_checkpoint = model_context + 'model_checkpoint_latest.pointer'
        eval_result_path = model_context + 'eval_result.pkl'
        predictions_path = path_context + 'predictions.csv'
        save_path = path_context + self.save_file_name
        return path_context, model_context, latest_model_checkpoint, eval_result_path, predictions_path, save_path

    def fit(self, X: DataFrame, X_test: DataFrame=None, sample=None):
        raise NotImplementedError

    # TODO: Add pred_proba_cache functionality as in predict()
    def predict_proba(self, X_test: DataFrame, inverse_transform=True, sample=None):
        ##########
        # Enable below for local testing
        if sample is not None:
            X_test = X_test.head(sample)
        ##########
        trainer = self.load_trainer()

        X_test = self.feature_generator.transform(X_test)
        y_pred_proba = trainer.predict_proba(X_test)
        if inverse_transform:
            y_pred_proba = self.label_cleaner.inverse_transform_proba(y_pred_proba)
        return y_pred_proba

    # TODO: Add decorators for cache functionality, return core code to previous state
    # use_pred_cache to check for a cached prediction of rows, can dramatically speedup repeated runs
    # add_to_pred_cache will update pred_cache with new predictions
    def predict(self, X_test: DataFrame, sample=None, use_pred_cache=False, add_to_pred_cache=False):
        pred_cache = None
        if use_pred_cache or add_to_pred_cache:
            try:
                pred_cache = load_pd.load(path=self.pred_cache_path, dtype=X_test[self.submission_columns].dtypes.to_dict())
            except Exception:
                pass
        if use_pred_cache and (pred_cache is not None):
            X_id = X_test[self.submission_columns]
            X_in_cache_with_pred = pd.merge(left=X_id.reset_index(), right=pred_cache, on=self.submission_columns).set_index('index')  # Will break if 'index' == self.label or 'index' in self.submission_columns
            X_test_cache_miss = X_test[~X_test.index.isin(X_in_cache_with_pred.index)]
            print('found', len(X_in_cache_with_pred), '/', len(X_test), 'rows with cached prediction')
        else:
            X_in_cache_with_pred = pd.DataFrame(data=None, columns=self.submission_columns + [self.label])
            X_test_cache_miss = X_test

        if len(X_test_cache_miss) > 0:
            y_pred_proba = self.predict_proba(X_test=X_test_cache_miss, inverse_transform=False, sample=sample)
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            y_pred = self.label_cleaner.inverse_transform(pd.Series(y_pred))
            y_pred.index = X_test_cache_miss.index
        else:
            print('all X_test rows found in cache, no need to load model...')
            return X_in_cache_with_pred[self.label].values
            
        if add_to_pred_cache:
            X_id_with_y_pred = X_test_cache_miss[self.submission_columns].copy()
            X_id_with_y_pred[self.label] = y_pred
            if pred_cache is None:
                pred_cache = X_id_with_y_pred.drop_duplicates(subset=self.submission_columns).reset_index(drop=True)
            else:
                pred_cache = pd.concat([X_id_with_y_pred, pred_cache]).drop_duplicates(subset=self.submission_columns).reset_index(drop=True)
            save_pd.save(path=self.pred_cache_path, df=pred_cache)
        
        if len(X_in_cache_with_pred) > 0:
            y_pred = pd.concat([y_pred, X_in_cache_with_pred[self.label]]).reindex(X_test.index)

        return y_pred.values

    def score(self, X: DataFrame, y=None):
        if y is None:
            X, y = self.extract_label(X)
        X = self.feature_generator.transform(X)
        y = self.label_cleaner.transform(y)
        trainer = self.load_trainer()
        if self.problem_type == MULTICLASS:
            y = y.fillna(-1)
            if trainer.objective_func_expects_y_pred:
                return trainer.score(X=X, y=y)
            else:
                # Log loss
                if -1 in y.unique():
                    raise ValueError('Multiclass scoring with ' + self.objective_func.name + ' does not support unknown classes.')
                return trainer.score(X=X, y=y)
        else:
            return trainer.score(X=X, y=y)

    # Scores both learner and all individual models, along with computing the optimal ensemble score + weights (oracle)
    def score_debug(self, X: DataFrame, y=None):
        if y is None:
            X, y = self.extract_label(X)
        X = self.feature_generator.transform(X)
        y = self.label_cleaner.transform(y)
        y = y.fillna(-1)
        trainer = self.load_trainer()
        scores = {}
        model_names = trainer.model_names
        if (self.problem_type == MULTICLASS) and (not trainer.objective_func_expects_y_pred):
            # Handles case where we need to add empty columns to represent classes that were not used for training
            y_pred_proba = trainer.predict_proba(X)
            y_pred_proba = self.label_cleaner.inverse_transform_proba(y_pred_proba)
            scores['weighted_ensemble'] = self.objective_func(y, y_pred_proba)

            pred_probas = trainer.pred_proba_predictions(models=model_names, X_test=X)
            pred_probas = [self.label_cleaner.inverse_transform_proba(pred_proba) for pred_proba in pred_probas]
            for i, model_name in enumerate(model_names):
                scores[model_name] = self.objective_func(y, pred_probas[i])

        else:
            scores['weighted_ensemble'] = trainer.score(X=X, y=y)
            for model_name in model_names:
                model = trainer.load_model(model_name)
                scores[model_name] = model.score(X=X, y=y)
            pred_probas = trainer.pred_proba_predictions(models=model_names, X_test=X)

        ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=self.problem_type, metric=self.objective_func)
        ensemble_selection.fit(predictions=pred_probas, labels=y, identifiers=None)
        oracle_weights = ensemble_selection.weights_
        oracle_pred_proba_norm = [pred * weight for pred, weight in zip(pred_probas, oracle_weights)]
        oracle_pred_proba_ensemble = np.sum(oracle_pred_proba_norm, axis=0)
        if trainer.objective_func_expects_y_pred:
            oracle_pred_ensemble = get_pred_from_proba(y_pred_proba=oracle_pred_proba_ensemble, problem_type=self.problem_type)
            scores['oracle_ensemble'] = self.objective_func(y, oracle_pred_ensemble)
        else:
            scores['oracle_ensemble'] = self.objective_func(y, oracle_pred_proba_ensemble)

        print('MODEL SCORES:')
        print(scores)
        return scores

    def evaluate(self, y_true, y_pred, silent=False, auxiliary_metrics=False, detailed_report=True):
        """ Evaluate predictions. 
            Args:
                silent (bool): Should we print which metric is being used as well as performance.
                auxiliary_metrics (bool): Should we compute other (problem_type specific) metrics in addition to the default metric?
                detailed_report (bool): Should we computed more-detailed versions of the auxiliary_metrics? (requires auxiliary_metrics=True) 
            Returns single performance-value if auxiliary_metrics=False.
            Otherwise returns dict where keys = metrics, values = performance along each metric.
        """
        
        # Remove missing labels and produce warning if any are found:
        if self.problem_type == REGRESSION:
            missing_indicators = [(y is None or np.isnan(y)) for y in y_true]
        else:
            missing_indicators = [(y is None or y=='') for y in y_true]
        missing_inds = [i for i,j in enumerate(missing_indicators) if j]
        if len(missing_inds) > 0:
            nonmissing_inds = [i for i,j in enumerate(missing_indicators) if j]
            y_true = y_true[nonmissing_inds]
            y_pred = y_pred[nonmissing_inds]
            warnings.warn("There are %s (out of %s) evaluation datapoints for which the label is missing. " 
                          "AutoGluon removed these points from the evaluation, which thus may not be entirely representative. " 
                          "You should carefully study why there are missing labels in your evaluation data." % (len(missing_inds),len(y_true)))
        
        perf = self.objective_func(y_true, y_pred)
        metric = self.objective_func.__name__
        if not silent:
            print("Evaluation: %s on test data: %f" % (metric, perf))
        if not auxiliary_metrics:
            return perf
        # Otherwise compute auxiliary metrics:
        perf_dict = OrderedDict({metric: perf})
        if self.problem_type == REGRESSION: # Additional metrics: R^2, Mean-Absolute-Error, Pearson correlation
            pearson_corr = lambda x,y: corrcoef(x,y)[0][1]
            pearson_corr.__name__ = 'pearson_correlation'
            regression_metrics = [mean_absolute_error, explained_variance_score, r2_score, pearson_corr, mean_squared_error, median_absolute_error,
                                  # max_error
                                  ]
            for reg_metric in regression_metrics:
                metric_name = reg_metric.__name__
                if metric_name not in perf_dict:
                    perf_dict[metric_name] = reg_metric(y_true, y_pred)
        else: # Compute classification metrics
            classif_metrics = [accuracy_score, balanced_accuracy_score, matthews_corrcoef]
            if self.problem_type == BINARY: # binary-specific metrics
                # def auc_score(y_true, y_pred): # TODO: this requires y_pred to be probability-scores
                #     fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label)
                #   return auc(fpr, tpr)
                f1micro_score = lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
                f1micro_score.__name__ = f1_score.__name__
                classif_metrics += [f1micro_score] # TODO: add auc?
            elif self.problem_type == MULTICLASS: # multiclass metrics
                classif_metrics += [] # TODO: No multi-class specific metrics for now. Include, top-1, top-5, top-10 accuracy here.
            for cl_metric in classif_metrics:
                metric_name = cl_metric.__name__
                if metric_name not in perf_dict:
                    perf_dict[metric_name] = cl_metric(y_true, y_pred)
        if not silent:
            print("Evaluations on test data:")
            print(json.dumps(perf_dict, indent=4))
        if detailed_report and (self.problem_type != REGRESSION):
            # One final set of metrics to report
            cl_metric = lambda y_true,y_pred: classification_report(y_true,y_pred, output_dict=True)
            metric_name = cl_metric.__name__
            if metric_name not in perf_dict:
                perf_dict[metric_name] = cl_metric(y_true, y_pred)
                if not silent:
                    print("Detailed (per-class) classification report:")
                    print(json.dumps(perf_dict[metric_name], indent=4))
        return perf_dict

    def extract_label(self, X):
        y = X[self.label].copy()
        X = X.drop(self.label, axis=1)
        return X, y

    def submit_from_preds(self, X_test: DataFrame, y_pred_proba, save=True, save_proba=False):
        submission = X_test[self.submission_columns].copy()
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

        submission[self.label] = y_pred
        submission[self.label] = self.label_cleaner.inverse_transform(submission[self.label])

        submission_proba = pd.DataFrame(y_pred_proba)

        if save:
            utcnow = datetime.datetime.utcnow()
            timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
            path_submission = self.model_context + 'submissions/submission_' + timestamp_str_now + '.csv'
            path_submission_proba = self.model_context + 'submissions/submission_proba_' + timestamp_str_now + '.csv'
            save_pd.save(path=path_submission, df=submission)
            if save_proba:
                save_pd.save(path=path_submission_proba, df=submission_proba)

        return submission

    def predict_and_submit(self, X_test: DataFrame, save=True, save_proba=False):
        y_pred_proba = self.predict_proba(X_test=X_test, inverse_transform=False)
        return self.submit_from_preds(X_test=X_test, y_pred_proba=y_pred_proba, save=save, save_proba=save_proba)

    @staticmethod
    def get_problem_type(y: Series):
        """ Identifies which type of prediction problem we are interested in (if user has not specified).
            Ie. binary classification, multi-class classification, or regression. 
        """
        if len(y) == 0:
            raise ValueError("provided labels cannot have length = 0")
        y = y.dropna() # Remove missing values from y (there should not be any though as they were removed in Learner.general_data_processing())
        unique_vals = y.unique()
        # print(unique_vals)
        REGRESS_THRESHOLD = 0.1 # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers 
        if len(unique_vals) == 2:
            problem_type = BINARY
            reason = "only two unique label-values observed"
        elif unique_vals.dtype == 'float':
            unique_ratio = len(unique_vals) / float(len(y))
            if unique_ratio <= REGRESS_THRESHOLD:
                try:
                    pd.to_numeric(y, downcast='integer')
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                except:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            else:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and many unique label-values observed"
        elif unique_vals.dtype == 'object':
            problem_type = MULTICLASS
            reason = "dtype of label-column == object"
        elif unique_vals.dtype == 'int':
            unique_ratio = len(unique_vals)/float(len(y))
            if unique_ratio > REGRESS_THRESHOLD:
                problem_type = REGRESSION
                reason = "dtype of label-column == int and many unique label-values observed"
            else:
                problem_type = MULTICLASS
                reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            raise NotImplementedError('label dtype', unique_vals.dtype, 'not supported!')
        print("\n AutoGluon infers your prediction problem is: %s  (because %s)" % (problem_type, reason))
        print("If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['%s', '%s', '%s']) \n\n" % (BINARY, MULTICLASS, REGRESSION))
        return problem_type

    def save(self):
        save_pkl.save(path=self.save_path, object=self)

    # reset_paths=True if the learner files have changed location since fitting.
    @classmethod
    def load(cls, path_context, reset_paths=False):
        load_path = path_context + cls.save_file_name
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
