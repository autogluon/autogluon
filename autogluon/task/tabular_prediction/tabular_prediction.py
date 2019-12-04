import logging
import numpy as np

from .dataset import TabularDataset
from ..base import BaseTask
from ..base.base_task import schedulers
from ...utils.tabular.ml.learner.default_learner import DefaultLearner as Learner
from ...utils.tabular.ml.trainer.auto_trainer import AutoTrainer
from ...utils.tabular.features.auto_ml_feature_generator import AutoMLFeatureGenerator
from ...utils.tabular.ml.utils import setup_outputdir, setup_compute, setup_trial_limits
from ...utils.tabular.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from ...utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

__all__ = ['TabularPrediction']

logger = logging.getLogger(__name__)


class TabularPrediction(BaseTask):
    """ 
    AutoGluon task for predicting a column of tabular dataset (both classification and regression).
    """
    
    Dataset = TabularDataset
    
    @staticmethod
    def load(output_directory):
        """ 
        Loads a predictor object previously produced by fit() from file and returns this object.
        
        Parameters
        ----------
        output_directory : (str)
            Path to directory where trained models are stored, ie. the output_directory specified in previous fit() call.
        """
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")
        output_directory = setup_outputdir(output_directory) # replace ~ with absolute path if it exists
        return Learner.load(output_directory)
    
    @staticmethod
    def fit(train_data, label, tuning_data=None, output_directory=None, problem_type=None, eval_metric=None,
            hyperparameter_tune=True, feature_prune=False, holdout_frac=None, num_bagging_folds=0, stack_ensemble_levels=0,
            hyperparameters = {'NN': {'num_epochs': 500}, 
                               'GBM': {'num_boost_round': 10000},
                               'CAT': {'iterations': 10000},
                               'RF': {'n_estimators': 300},
                               'XT': {'n_estimators': 150},
                               'KNN': {},
                              },
            time_limits=None, num_trials=None, dist_ip_addrs=[], visualizer='none',
            nthreads_per_trial=None, ngpus_per_trial=None,
            search_strategy='random', search_options={}, **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.
        
        Parameters
        ----------
        train_data : (TabularDataset object)
            Table of the training data, which is similar to pandas DataFrame.
        label : (str)
            Name of column that contains the target variable to predict.
        tuning_data : (TabularDataset object)
            Another dataset containing validation data reserved for hyperparameter tuning (in same format as training data).
            Note: final model returned may be fit on this tuning_data as well as train_data. Do not provide your test data here!
            If tuning_data = None, fit() will automatically hold out some random validation examples from train_data.
        output_directory : (str) 
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called 'autogluon-fit-TIMESTAMP" will be created in the working directory to store all models.
            Note: To call fit() twice and save all results of each fit, you must specify different locations for output_directory.
                  Otherwise files from first fit() will be overwritten by second fit().
        problem_type : (str) 
            Type of prediction problem, ie. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
            If problem_type = None, the prediction problem type will be automatically inferred based on target LABEL column in dataset.
        objective_func : (func or str)
            Metric by which performance will be ultimately evaluated on test data.
            AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.
            If = None, objective_func is automatically chosen based on problem_type.
            Otherwise options for classification include: 'accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'log_loss', 'pac_score'
            and options for regression include: 'r2', 'mean_squared_error', 'root_mean_squared_error' 'mean_absolute_error', 'median_absolute_error'
            You can also pass your own function here as long as it follows the format of the functions defined in utils/tabular/metrics/.
        hyperparameter_tune : (bool)
            Whether to tune hyperparameters or just use fixed hyperparameter values for each model
        feature_prune : (bool)
            Whether or not to perform feature selection
        hyperparameters : (dict) 
            Keys are strings that indicate which model types to train.
                Options include: 'NN' (neural network), 'GBM' (lightGBM boosted trees), 'CAT' (CatBoost boosted trees), 'RF' (random forest).
                If certain key is missing from hyperparameters, then fit() will not train any models of that type.
            Values = dict of hyperparameter settings for each model type.
                Each hyperparameter can be fixed value or search space. For full list of options, please see documentation.
                Hyperparameters not specified will be set to default values (or default search spaces if hyperparameter_tune = True).
                Caution: Any provided search spaces will be overriden by fixed defauls if hyperparameter_tune = False.
                # TODO: create documentation file describing all models and  all hyperparameters.
        holdout_frac : (float) 
            Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless tuning_data=None, ignored if kfolds != 0).
            Default value is 0.2 if hyperparameter_tune = True, otherwise 0.1 is used.
        num_bagging_folds : (int)
            Kfolds used for bagging of models. Roughly increases model training time by a factor of k (0: disabled)
            Default is 0 (disabled). Use values between 5-10 to improve model quality.
        stack_ensemble_levels : (int)
            Number of stacking levels to use in ensemble stacking. Roughly increases model training time by factor of stack_levels+1 (0: disabled)
            Default is 0 (disabled). Use values between 1-3 to improve model quality.
            Ignored unless kfolds is also set >= 2
        search_strategy : (str) 
            Which hyperparameter search algorithm to use. 
            Options include: 'random' (random search), 'skopt' (SKopt Bayesian optimization), 'grid' (grid search), 'hyperband' (Hyperband), 'rl' (reinforcement learner)
        search_options : (dict)
            Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization
        time_limits : (int)
            Approximately how long this call to fit() should run for (wallclock time in seconds).
            fit() will stop scheduling new trials after this amount of time has elapsed.
        num_trials : (int) 
            Maximal number of different hyperparameter settings of each model type to evaluate.
            If both time_limits and num_trials are specified, time_limits takes precedent.
        dist_ip_addrs : (list)
            List of IP addresses corresponding to remote workers.
        visualizer : (str)
            Describes method to visualize training progress during fit(). Options include: 'mxboard','tensorboard','none'
        nthreads_per_trial : (int)
            How many CPUs to use in each trial (ie. single training run of a model).
        ngpus_per_trial : (int)
            How many GPUs to use in each trial (ie. single training run of a model).
        
        Kwargs can include addtional arguments for advanced users:
            feature_generator_type : (FeatureGenerator class, default=AutoMLFeatureGenerator)
                A FeatureGenerator class (see AbstractFeatureGenerator) that specifies which feature engineering protocol to follow.
                Note: class file must be imported into Python session in order to use a custom class.
            feature_generator_kwargs : (dict, default={}) 
                Keyword arguments dictionary to pass into FeatureGenerator constructor.
            trainer_type : (Trainer class, default=AutoTrainer)
                A Trainer class (see AbstractTrainer) that controls training/ensembling of many models, default = AutoTrainer.
                Note: class file must be imported into Python session in order to use custom class.
                TODO(Nick): does trainer constructor ever require kwargs? If so should have trainer_type_kwargs dict used similarly as feature_generator_kwargs
            label_count_threshold : (int, default = 10)
                For multi-class classification problems, this is the minimum number of times a label must appear in dataset in order to be considered an output class.
                AutoGluon will ignore any classes whose labels do not appear at least this many times in the dataset (ie. will never predict them), default = 10.
            id_columns : (list)
                Banned subset of column names that model may not use as predictive features (eg. contains label, user-ID, etc), default = [].
                These columns are ignored during fit(), but DataFrame of just these columns with appended predictions may be submitted for a ML competition.
        
        Returns
        -------
            DefaultLearner object with methods: predict(), predict_proba(), score(), evaluate(), load(), save()
            # TODO: document Learner object.
        
        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset(file_path=''https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')
        >>> label_column = 'class'
        >>> predictor = task.fit(train_data=train_data, label=label_column, hyperparameter_tune=False)
        >>> test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv') # Another Pandas object
        >>> y_test = test_data[label_column]
        >>> test_data = test_data.drop(labels=[label_column], axis=1)
        >>> y_pred = predictor.predict(test_data)
        >>> perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)
        """
        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None and np.any(train_data.columns != tuning_data.columns):
            raise ValueError("Column names must match between training and tuning data")

        if feature_prune:
            feature_prune = False  # TODO: Fix feature pruning to add back as an option
            # Currently disabled, needs to be updated to align with new model class functionality
            print('Warning: feature_prune was set to True, but feature_prune does not currently work. Setting to False.')

        # Process kwargs to create feature generator, trainer, schedulers, searchers for each model:
        output_directory = setup_outputdir(output_directory) # Format directory name
        feature_generator_type = kwargs.get('feature_generator_type', AutoMLFeatureGenerator)
        feature_generator_kwargs = kwargs.get('feature_generator_kwargs', {})
        feature_generator = feature_generator_type(**feature_generator_kwargs) # instantiate FeatureGenerator object
        label_count_threshold = kwargs.get('label_count_threshold', 10)
        id_columns = kwargs.get('id_columns', [])
        trainer_type = kwargs.get('trainer_type', AutoTrainer)
        nthreads_per_trial, ngpus_per_trial = setup_compute(nthreads_per_trial, ngpus_per_trial)
        time_limits, num_trials = setup_trial_limits(time_limits, num_trials, hyperparameters)
        if holdout_frac is None:
            holdout_frac = 0.2 if hyperparameter_tune else 0.1
        # Add visualizer to NN hyperparameters:
        if ((visualizer is not None) and (visualizer != 'none') and 
            ('NN' in hyperparameters)):
            hyperparameters['NN']['visualizer'] = visualizer
        if objective_func is not None and isinstance(objective_func, str): # convert to function
                if objective_func in CLASSIFICATION_METRICS:
                    if problem_type is not None and problem_type not in [BINARY, MULTICLASS]:
                        raise ValueError("objective_func=%s cannot be used for problem_type=%s" % 
                            (objective_func, problem_type))
                    objective_func = CLASSIFICATION_METRICS[objective_func]
                elif objective_func in REGRESSION_METRICS:
                    if problem_type is not None and problem_type != REGRESSION:
                        raise ValueError("objective_func=%s cannot be used for problem_type=%s" % 
                            (objective_func, problem_type))
                    objective_func = REGRESSION_METRICS[objective_func]
                else:
                    raise ValueError("%s is unknown metric, see utils/tabular/metrics/ for available options or how to define your own objective_func function")
        
        # All models use the same scheduler:
        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'num_trials': num_trials,
            'time_out': time_limits,
            'visualizer': visualizer,
            'time_attr': 'epoch',  # For tree ensembles, each new tree (ie. boosting round) is considered one epoch
            'reward_attr': 'validation_performance',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler_options = (scheduler, scheduler_options)  # wrap into tuple
        predictor = Learner(path_context=output_directory, label=label, problem_type=problem_type, objective_func=objective_func, 
            id_columns=id_columns, feature_generator=feature_generator, trainer_type=trainer_type, label_count_threshold=label_count_threshold)
        predictor.fit(X=train_data, X_test=tuning_data, scheduler_options=scheduler_options,
                      hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                      holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, stack_ensemble_levels=stack_ensemble_levels, hyperparameters=hyperparameters)
        return predictor
