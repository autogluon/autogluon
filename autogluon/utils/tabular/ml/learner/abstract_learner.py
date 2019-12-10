import datetime, json, warnings, logging
from collections import OrderedDict
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, classification_report # , roc_curve, auc
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error, median_absolute_error # , max_error
import numpy as np
from numpy import corrcoef

from ..constants import BINARY, MULTICLASS, REGRESSION
from ...data.label_cleaner import LabelCleaner
from ..utils import get_pred_from_proba
from ...utils.loaders import load_pkl, load_pd
from ...utils.savers import save_pkl, save_pd
from ..trainer.abstract_trainer import AbstractTrainer
from ..tuning.ensemble_selection import EnsembleSelection

logger = logging.getLogger(__name__)

# TODO: - Semi-supervised learning
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class AbstractLearner:
    save_file_name = 'learner.pkl'

    def __init__(self, path_context: str, label: str, id_columns: list, feature_generator, label_count_threshold=10, 
                 problem_type=None, objective_func=None, is_trainer_present=False):
        self.path_context, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)
        self.label = label
        self.submission_columns = id_columns
        self.threshold = label_count_threshold
        self.problem_type = problem_type
        self.trainer_problem_type = None
        self.objective_func = objective_func
        self.is_trainer_present = is_trainer_present
        self.cleaner = None
        self.label_cleaner: LabelCleaner = None
        self.feature_generator = feature_generator
        self.feature_generators = [self.feature_generator]

        self.trainer: AbstractTrainer = None
        self.trainer_type = None
        self.trainer_path = None
        self.reset_paths = False

    @property
    def class_labels(self):
        if self.problem_type == MULTICLASS:
            return self.label_cleaner.ordered_class_labels
        else:
            return None

    def set_contexts(self, path_context):
        self.path_context, self.model_context, self.latest_model_checkpoint, self.eval_result_path, self.pred_cache_path, self.save_path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        model_context = path_context + 'models/'
        latest_model_checkpoint = model_context + 'model_checkpoint_latest.pointer'
        eval_result_path = model_context + 'eval_result.pkl'
        predictions_path = path_context + 'predictions.csv'
        save_path = path_context + self.save_file_name
        return path_context, model_context, latest_model_checkpoint, eval_result_path, predictions_path, save_path

    def fit(self, X: DataFrame, X_test: DataFrame = None, scheduler_options=None, hyperparameter_tune=True, 
            feature_prune=False, holdout_frac=0.1, hyperparameters={}, verbosity=2):
        raise NotImplementedError

    # TODO: Add pred_proba_cache functionality as in predict()
    def predict_proba(self, X_test: DataFrame, as_pandas=False, inverse_transform=True, sample=None):
        ##########
        # Enable below for local testing # TODO: do we want to keep sample option?
        if sample is not None:
            X_test = X_test.head(sample)
        ##########
        trainer = self.load_trainer()

        X_test = self.transform_features(X_test)
        y_pred_proba = trainer.predict_proba(X_test)
        if inverse_transform:
            y_pred_proba = self.label_cleaner.inverse_transform_proba(y_pred_proba)
        if as_pandas:
            if self.problem_type == MULTICLASS:
                y_pred_proba = pd.DataFrame(data=y_pred_proba, columns=self.class_labels)
            else:
                y_pred_proba = pd.Series(data=y_pred_proba, name=self.label)
        return y_pred_proba

    # TODO: Add decorators for cache functionality, return core code to previous state
    # use_pred_cache to check for a cached prediction of rows, can dramatically speedup repeated runs
    # add_to_pred_cache will update pred_cache with new predictions
    def predict(self, X_test: DataFrame, as_pandas=False, sample=None, use_pred_cache=False, add_to_pred_cache=False):
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
            logger.log(20, 'Using cached predictions for '+str(len(X_in_cache_with_pred))+' out of '+str(len(X_test))+' rows, which have already been predicted previously. To make new predictions, set use_pred_cache=False')
        else:
            X_in_cache_with_pred = pd.DataFrame(data=None, columns=self.submission_columns + [self.label])
            X_test_cache_miss = X_test

        if len(X_test_cache_miss) > 0:
            y_pred_proba = self.predict_proba(X_test=X_test_cache_miss, inverse_transform=False, sample=sample)
            if self.trainer_problem_type is not None:
                problem_type = self.trainer_problem_type
            else:
                problem_type = self.problem_type
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
            y_pred = self.label_cleaner.inverse_transform(pd.Series(y_pred))
            y_pred.index = X_test_cache_miss.index
        else:
            logger.debug('All X_test rows found in cache, no need to load model')
            y_pred = X_in_cache_with_pred[self.label].values
            if as_pandas:
                y_pred = pd.Series(data=y_pred, name=self.label)
            return y_pred

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

        y_pred = y_pred.values
        if as_pandas:
            y_pred = pd.Series(data=y_pred, name=self.label)
        return y_pred

    def fit_transform_features(self, X, y=None):
        for feature_generator in self.feature_generators:
            X = feature_generator.fit_transform(X, y)
        return X

    def transform_features(self, X):
        for feature_generator in self.feature_generators:
            X = feature_generator.transform(X)
        return X

    def score(self, X: DataFrame, y=None):
        if y is None:
            X, y = self.extract_label(X)
        X = self.transform_features(X)
        y = self.label_cleaner.transform(y)
        trainer = self.load_trainer()
        if self.problem_type == MULTICLASS:
            y = y.fillna(-1)
            if trainer.objective_func_expects_y_pred:
                return trainer.score(X=X, y=y)
            else:
                # Log loss
                if -1 in y.unique():
                    raise ValueError('Multiclass scoring with eval_metric=' + self.objective_func.name + ' does not support unknown classes.')
                return trainer.score(X=X, y=y)
        else:
            return trainer.score(X=X, y=y)

    # Scores both learner and all individual models, along with computing the optimal ensemble score + weights (oracle)
    def score_debug(self, X: DataFrame, y=None):
        if y is None:
            X, y = self.extract_label(X)
        X = self.transform_features(X)
        y = self.label_cleaner.transform(y)
        trainer = self.load_trainer()
        if self.problem_type == MULTICLASS:
            y = y.fillna(-1)
            if (not trainer.objective_func_expects_y_pred) and (-1 in y.unique()):
                # Log loss
                raise ValueError('Multiclass scoring with eval_metric=' + self.objective_func.name + ' does not support unknown classes.')
        # TODO: Move below into trainer, should not live in learner
        max_level = trainer.max_level
        max_level_auxiliary = trainer.max_level_auxiliary

        max_level_to_check = max(max_level, max_level_auxiliary)
        scores = {}
        pred_probas = None
        for level in range(max_level_to_check+1):
            model_names_core = trainer.models_level[level]
            if level >= 1:
                X_stack = trainer.get_inputs_to_stacker(X, level_start=0, level_end=level, y_pred_probas=pred_probas)
            else:
                X_stack = X

            if len(model_names_core) > 0:
                pred_probas = self.get_pred_probas_models(X=X_stack, trainer=trainer, model_names=model_names_core)
                for i, model_name in enumerate(model_names_core):
                    pred_proba = pred_probas[i]
                    if (trainer.problem_type == BINARY) and (self.problem_type == MULTICLASS):
                        pred_proba = self.label_cleaner.inverse_transform_proba(pred_proba)
                    if trainer.objective_func_expects_y_pred:
                        pred = get_pred_from_proba(y_pred_proba=pred_proba, problem_type=self.problem_type)
                        scores[model_name] = self.objective_func(y, pred)
                    else:
                        scores[model_name] = self.objective_func(y, pred_proba)

                ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=trainer.problem_type, metric=self.objective_func)
                ensemble_selection.fit(predictions=pred_probas, labels=y, identifiers=None)
                oracle_weights = ensemble_selection.weights_
                oracle_pred_proba_norm = [pred * weight for pred, weight in zip(pred_probas, oracle_weights)]
                oracle_pred_proba_ensemble = np.sum(oracle_pred_proba_norm, axis=0)
                if (trainer.problem_type == BINARY) and (self.problem_type == MULTICLASS):
                    oracle_pred_proba_ensemble = self.label_cleaner.inverse_transform_proba(oracle_pred_proba_ensemble)
                if trainer.objective_func_expects_y_pred:
                    oracle_pred_ensemble = get_pred_from_proba(y_pred_proba=oracle_pred_proba_ensemble, problem_type=self.problem_type)
                    scores['oracle_ensemble_l' + str(level+1)] = self.objective_func(y, oracle_pred_ensemble)
                else:
                    scores['oracle_ensemble_l' + str(level+1)] = self.objective_func(y, oracle_pred_proba_ensemble)

            model_names_aux = trainer.models_level_auxiliary[level]
            if len(model_names_aux) > 0:
                pred_probas_auxiliary = self.get_pred_probas_models(X=X_stack, trainer=trainer, model_names=model_names_aux)
                for i, model_name in enumerate(model_names_aux):
                    pred_proba = pred_probas_auxiliary[i]
                    if (trainer.problem_type == BINARY) and (self.problem_type == MULTICLASS):
                        pred_proba = self.label_cleaner.inverse_transform_proba(pred_proba)
                    if trainer.objective_func_expects_y_pred:
                        pred = get_pred_from_proba(y_pred_proba=pred_proba, problem_type=self.problem_type)
                        scores[model_name] = self.objective_func(y, pred)
                    else:
                        scores[model_name] = self.objective_func(y, pred_proba)

        logger.debug('Model scores:')
        logger.debug(str(scores))
        model_names = []
        scores_test = []
        for model in scores.keys():
            model_names.append(model)
            scores_test.append(scores[model])
        df = pd.DataFrame(data={
            'model': model_names,
            'score_test': scores_test,
        })

        df = df.sort_values(by='score_test', ascending=False).reset_index(drop=True)

        leaderboard_df = self.leaderboard()

        df_merged = pd.merge(df, leaderboard_df, on='model')

        return df_merged

    def get_pred_probas_models(self, X, trainer, model_names):
        if (self.problem_type == MULTICLASS) and (not trainer.objective_func_expects_y_pred):
            # Handles case where we need to add empty columns to represent classes that were not used for training
            pred_probas = trainer.pred_proba_predictions(models=model_names, X_test=X)
            pred_probas = [self.label_cleaner.inverse_transform_proba(pred_proba) for pred_proba in pred_probas]
        else:
            pred_probas = trainer.pred_proba_predictions(models=model_names, X_test=X)
        return pred_probas

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
        metric = self.objective_func.name
        if not high_always_good:
            sign = self.objective_func._sign
            perf = perf * sign # flip negative once again back to positive (so higher is no longer necessarily better)
        if not silent:
            logger.log(20, "Evaluation: %s on test data: %f" % (metric, perf))
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
            logger.log(20, "Evaluations on test data:")
            logger.log(20, json.dumps(perf_dict, indent=4))
        if detailed_report and (self.problem_type != REGRESSION):
            # One final set of metrics to report
            cl_metric = lambda y_true,y_pred: classification_report(y_true,y_pred, output_dict=True)
            metric_name = cl_metric.__name__
            if metric_name not in perf_dict:
                perf_dict[metric_name] = cl_metric(y_true, y_pred)
                if not silent:
                    logger.log(20, "Detailed (per-class) classification report:")
                    logger.log(20, json.dumps(perf_dict[metric_name], indent=4))
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


        if save:
            utcnow = datetime.datetime.utcnow()
            timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
            path_submission = self.model_context + 'submissions/submission_' + timestamp_str_now + '.csv'
            path_submission_proba = self.model_context + 'submissions/submission_proba_' + timestamp_str_now + '.csv'
            save_pd.save(path=path_submission, df=submission)
            if save_proba:
                submission_proba = pd.DataFrame(y_pred_proba)  # TODO: Fix for multiclass
                save_pd.save(path=path_submission_proba, df=submission_proba)

        return submission

    def predict_and_submit(self, X_test: DataFrame, save=True, save_proba=False):
        y_pred_proba = self.predict_proba(X_test=X_test, inverse_transform=False)
        return self.submit_from_preds(X_test=X_test, y_pred_proba=y_pred_proba, save=save, save_proba=save_proba)

    def leaderboard(self, X=None, y=None, silent=False):
        if X is not None:
            leaderboard = self.score_debug(X=X, y=y)
        else:
            trainer = self.load_trainer()
            leaderboard = trainer.leaderboard()
        if not silent:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(leaderboard)
        return leaderboard

    def info(self):
        trainer = self.load_trainer()
        return trainer.info()

    @staticmethod
    def get_problem_type(y: Series):
        """ Identifies which type of prediction problem we are interested in (if user has not specified).
            Ie. binary classification, multi-class classification, or regression. 
        """
        if len(y) == 0:
            raise ValueError("provided labels cannot have length = 0")
        y = y.dropna()  # Remove missing values from y (there should not be any though as they were removed in Learner.general_data_processing())
        unique_vals = y.unique()
        num_rows = len(y)
        # print(unique_vals)
        logger.log(20, 'Here are the first 10 unique label values in your data:  '+str(unique_vals[:10]))
        unique_count = len(unique_vals)
        MULTICLASS_LIMIT = 1000  # if numeric and class count would be above this amount, assume it is regression
        if num_rows > 1000:
            REGRESS_THRESHOLD = 0.05  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
        else:
            REGRESS_THRESHOLD = 0.1
        if len(unique_vals) == 2:
            problem_type = BINARY
            reason = "only two unique label-values observed"
        elif unique_vals.dtype == 'float':
            unique_ratio = len(unique_vals) / float(len(y))
            if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
                try:
                    can_convert_to_int = np.array_equal(y, y.astype(int))
                    if can_convert_to_int:
                        problem_type = MULTICLASS
                        reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                    else:
                        problem_type = REGRESSION
                        reason = "dtype of label-column == float and label-values can't be converted to int"
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
            if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
                problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
                reason = "dtype of label-column == int, but few unique label-values observed"
            else:
                problem_type = REGRESSION
                reason = "dtype of label-column == int and many unique label-values observed"
        else:
            raise NotImplementedError('label dtype', unique_vals.dtype, 'not supported!')
        logger.log(25, "AutoGluon infers your prediction problem is: %s  (because %s)" % (problem_type, reason))
        logger.log(25, "If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['%s', '%s', '%s'])\n" % (BINARY, MULTICLASS, REGRESSION))
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
