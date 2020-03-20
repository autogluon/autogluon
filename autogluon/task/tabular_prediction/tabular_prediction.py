import copy
import logging
import math

import numpy as np

from .dataset import TabularDataset
from .predictor import TabularPredictor
from ..base import BaseTask
from ..base.base_task import schedulers
from ...utils import verbosity2loglevel
from ...utils.tabular.features.auto_ml_feature_generator import AutoMLFeatureGenerator
from ...utils.tabular.metrics import get_metric
from ...utils.tabular.ml.learner.default_learner import DefaultLearner as Learner
from ...utils.tabular.ml.trainer.auto_trainer import AutoTrainer
from ...utils.tabular.ml.utils import setup_outputdir, setup_compute, setup_trial_limits

__all__ = ['TabularPrediction']

logger = logging.getLogger()  # return root logger


class TabularPrediction(BaseTask):
    """
    AutoGluon Task for predicting values in column of tabular dataset (classification or regression)
    """

    Dataset = TabularDataset
    Predictor = TabularPredictor

    @staticmethod
    def load(output_directory, verbosity=2):
        """
        Load a predictor object previously produced by `fit()` from file and returns this object.

        Parameters
        ----------
        output_directory : str
            Path to directory where trained models are stored (i.e. the output_directory specified in previous call to `fit`).
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information will be printed by the loaded `Predictor`.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where L ranges from 0 to 50 (Note: higher values L correspond to fewer print statements, opposite of verbosity levels)

        Returns
        -------
        :class:`autogluon.task.tabular_prediction.TabularPredictor` object that can be used to make predictions.
        """
        logger.setLevel(verbosity2loglevel(verbosity)) # Reset logging after load (since we may be in new Python session)
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")

        output_directory = setup_outputdir(output_directory) # replace ~ with absolute path if it exists
        learner = Learner.load(output_directory)
        return TabularPredictor(learner=learner)

    @staticmethod
    def fit(train_data, label, tuning_data=None, output_directory=None, problem_type=None, eval_metric=None, stopping_metric=None,
            auto_stack=False, hyperparameter_tune=False, feature_prune=False, holdout_frac=None,
            num_bagging_folds=0, num_bagging_sets=None, stack_ensemble_levels=0,
            hyperparameters=None, cache_data=True,
            time_limits=None, num_trials=None, search_strategy='random', search_options=None,
            nthreads_per_trial=None, ngpus_per_trial=None, dist_ip_addrs=None, visualizer='none',
            verbosity=2, **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.

        Parameters
        ----------
        train_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`
            Table of the training data, which is similar to pandas DataFrame.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        label : str
            Name of the column that contains the target variable to predict.
        tuning_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            Another dataset containing validation data reserved for hyperparameter tuning (in same format as training data).
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: final model returned may be fit on this tuning_data as well as train_data. Do not provide your evaluation test data here!
            In particular, when `num_bagging_folds` > 0 or `stack_ensemble_levels` > 0, models will be trained on both `tuning_data` and `train_data`.
            If `tuning_data = None`, `fit()` will automatically hold out some random validation examples from `train_data`.
        output_directory : str
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "autogluon-fit-[TIMESTAMP]" will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `output_directory` locations.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        problem_type : str, default = None
            Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
            If `problem_type = None`, the prediction problem type is inferred based on the label-values in provided dataset.
        eval_metric : function or str, default = None
            Metric by which predictions will be ultimately evaluated on test data.
            AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.

            If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for binary and multiclass classification and 'root_mean_squared_error' for regression.
            Otherwise, options for classification: [
                'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                'roc_auc', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
                'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score'].
            Options for regression: ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'].
            For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

            You can also pass your own evaluation function here as long as it follows formatting of the functions defined in `autogluon/utils/tabular/metrics/`.
        stopping_metric : function or str, default = None
            Metric which iteratively-trained models use to early stop to avoid overfitting.
            `stopping_metric` is not used by weighted ensembles, instead weighted ensembles maximize `eval_metric`.
            Defaults to `eval_metric` value except when `eval_metric='roc_auc'`, where it defaults to `log_loss`.
            Options are identical to options for `eval_metric`.
        auto_stack : bool, default = False
            Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy.
            Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy!
            Note: This overrides `num_bagging_folds` and `stack_ensemble_levels` arguments (selects optimal values for these parameters based on dataset properties).
            Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.
        hyperparameter_tune : bool, default = False
            Whether to tune hyperparameters or just use fixed hyperparameter values for each model. Setting as True will increase `fit()` runtimes.
            It is currently not recommended to use `hyperparameter_tune` with `auto_stack` due to potential overfitting.
            Use `auto_stack` to maximize predictive accuracy; use `hyperparameter_tune` if you prefer to deploy just a single model rather than an ensemble.
        feature_prune : bool, default = False
            Whether or not to perform feature selection.
        hyperparameters : dict
            Keys are strings that indicate which model types to train.
                Options include: 'NN' (neural network), 'GBM' (lightGBM boosted trees), 'CAT' (CatBoost boosted trees), 'RF' (random forest), 'XT' (extremely randomized trees), 'KNN' (k-nearest neighbors)
                If certain key is missing from hyperparameters, then `fit()` will not train any models of that type.
                For example, set `hyperparameters = { 'NN':{...} }` if say you only want to train neural networks and no other types of models.
            Values = dict of hyperparameter settings for each model type.
                Each hyperparameter can either be single fixed value or a search space containing many possible values.
                Unspecified hyperparameters will be set to default values (or default search spaces if `hyperparameter_tune = True`).
                Caution: Any provided search spaces will be overriden by fixed defauls if `hyperparameter_tune = False`.

            Note: `hyperparameters` can also take a special key 'custom', which maps to a list of model names (currently supported options = 'GBM').
            If `hyperparameter_tune = False`, then these additional models will also be trained using custom pre-specified hyperparameter settings that are known to work well.

            Details regarding the hyperparameters you can specify for each model are provided in the following files:
                NN: `autogluon/utils/tabular/ml/models/tabular_nn/hyperparameters/parameters.py`
                    Note: certain hyperparameter settings may cause these neural networks to train much slower.
                GBM: `autogluon/utils/tabular/ml/models/lgb/hyperparameters/parameters.py`
                     See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
                CAT: `autogluon/utils/tabular/ml/models/catboost/hyperparameters/parameters.py`
                     See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html
                RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                    Note: 'criterion' parameter will be overriden. Both 'gini' and 'entropy' are used automatically, training two models.
                XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                    Note: 'criterion' parameter will be overriden. Both 'gini' and 'entropy' are used automatically, training two models.
                KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                    Note: Hyperparameter tuning is disabled for this model.
                    Note: 'weights' parameter will be overriden. Both 'distance' and 'uniform' are used automatically, training two models.

        holdout_frac : float
            Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`, ignored if `num_bagging_folds != 0`).
            Default value is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
            Default value is doubled if `hyperparameter_tune = True`, up to a maximum of 0.2.
            Disabled if `num_bagging_folds >= 2`.
        num_bagging_folds : int, default = 0
            Number of folds used for bagging of models. When `num_bagging_folds = k`, training time is roughly increased by a factor of `k` (set = 0 to disable bagging).
            Disabled by default, but we recommend values between 5-10 to maximize predictive performance.
            Increasing num_bagging_folds will result in models with lower bias but that are more prone to overfitting.
            Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
            To further improve predictions, avoid increasing num_bagging_folds much beyond 10 and instead increase num_bagging_sets.
        num_bagging_sets : int
            Number of repeats of kfold bagging to perform (values must be >= 1). Total number of models trained during bagging = num_bagging_folds * num_bagging_sets.
            Defaults to 1 if time_limits is not specified, otherwise 20 (always disabled if num_bagging_folds is not specified).
            Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
        stack_ensemble_levels : int, default = 0
            Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `stack_ensemble_levels+1` (set = 0 to disable stack ensembling).
            Disabled by default, but we recommend values between 1-3 to maximize predictive performance.
            To prevent overfitting, this argument is ignored unless you have also set `num_bagging_folds >= 2`.
        cache_data : bool, default = True
            When enabled, the training and validation data are saved to disk for future reuse.
            Enables advanced functionality in the resulting Predictor object such as feature importance calculation on the original data.
        time_limits : int
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until all models have completed training, but will not repeatedly bag models unless `num_bagging_sets` is specified.
        num_trials : int
            Maximal number of different hyperparameter settings of each model type to evaluate during HPO (only matters if `hyperparameter_tune = True`).
            If both `time_limits` and `num_trials` are specified, `time_limits` takes precedent.
        search_strategy : str
            Which hyperparameter search algorithm to use (only matters if `hyperparameter_tune = True`).
            Options include: 'random' (random search), 'skopt' (SKopt Bayesian optimization), 'grid' (grid search), 'hyperband' (Hyperband)
        search_options : dict
            Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization.
        nthreads_per_trial : int
            How many CPUs to use in each training run of an individual model.
            This is automatically determined by AutoGluon when left as None (based on available compute).
        ngpus_per_trial : int
            How many GPUs to use in each trial (ie. single training run of a model).
            This is automatically determined by AutoGluon when left as None.
        dist_ip_addrs : list
            List of IP addresses corresponding to remote workers, in order to leverage distributed computation.
        visualizer : str
            How to visualize the neural network training progress during `fit()`. Options: ['mxboard', 'tensorboard', 'none'].
        verbosity: int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed during fit().
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)

        Kwargs can include addtional arguments for advanced users:
            feature_generator_type : `FeatureGenerator` class, default=`AutoMLFeatureGenerator`
                A `FeatureGenerator` class specifying which feature engineering protocol to follow
                (see autogluon.utils.tabular.features.abstract_feature_generator.AbstractFeatureGenerator).
                Note: The file containing your `FeatureGenerator` class must be imported into current Python session in order to use a custom class.
            feature_generator_kwargs : dict, default={}
                Keyword arguments to pass into the `FeatureGenerator` constructor.
            trainer_type : `Trainer` class, default=`AutoTrainer`
                A class inheritng from `autogluon.utils.tabular.ml.trainer.abstract_trainer.AbstractTrainer` that controls training/ensembling of many models.
                Note: In order to use a custom `Trainer` class, you must import the class file that defines it into the current Python session.
            label_count_threshold : int, default = 10
                For multi-class classification problems, this is the minimum number of times a label must appear in dataset in order to be considered an output class.
                AutoGluon will ignore any classes whose labels do not appear at least this many times in the dataset (i.e. will never predict them).
            id_columns : list, default = []
                Banned subset of column names that model may not use as predictive features (e.g. contains label, user-ID, etc).
                These columns are ignored during `fit()`, but DataFrame of just these columns with appended predictions may be produced, for example to submit in a ML competition.

        Returns
        -------
        :class:`autogluon.task.tabular_prediction.TabularPredictor` object which can make predictions on new data and summarize what happened during `fit()`.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv')
        >>> label_column = 'class'
        >>> predictor = task.fit(train_data=train_data, label=label_column)
        >>> test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv')
        >>> y_test = test_data[label_column]
        >>> test_data = test_data.drop(labels=[label_column], axis=1)
        >>> y_pred = predictor.predict(test_data)
        >>> perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)
        >>> results = predictor.fit_summary()

        To maximize predictive performance, use the following:

        >>> eval_metric = 'roc_auc'  # set this to the metric you ultimately care about
        >>> time_limits = 360  # set as long as you are willing to wait (in sec)
        >>> predictor = task.fit(train_data=train_data, label=label_column, eval_metric=eval_metric, auto_stack=True, time_limits=time_limits)
        """
        if verbosity < 0:
            verbosity = 0
        elif verbosity > 4:
            verbosity = 4

        logger.setLevel(verbosity2loglevel(verbosity))
        allowed_kwarg_names = {
            'feature_generator_type',
            'feature_generator_kwargs',
            'trainer_type',
            'label_count_threshold',
            'id_columns',
            'enable_fit_continuation'  # TODO: Remove on 0.1.0 release
        }
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                raise ValueError("Unknown keyword argument specified: %s" % kwarg_name)

        if isinstance(train_data,  str):
            train_data = TabularDataset(file_path=train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(file_path=tuning_data)

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None and np.any(train_data.columns != tuning_data.columns):
            raise ValueError("Column names must match between training and tuning data")

        if feature_prune:
            feature_prune = False  # TODO: Fix feature pruning to add back as an option
            # Currently disabled, needs to be updated to align with new model class functionality
            logger.log(30, 'Warning: feature_prune does not currently work, setting to False.')

        # TODO: Remove on 0.1.0 release
        if 'enable_fit_continuation' in kwargs.keys():
            logger.log(30, 'Warning: `enable_fit_continuation` is a deprecated parameter. It has been renamed to `cache_data`. Starting from AutoGluon 0.1.0, specifying `enable_fit_continuation` as a parameter will cause an exception.')
            logger.log(30, 'Setting `cache_data` value equal to `enable_fit_continuation` value.')
            cache_data = kwargs['enable_fit_continuation']
        if not cache_data:
            logger.log(30, 'Warning: `cache_data=False` will disable or limit advanced functionality after training such as feature importance calculations. It is recommended to set `cache_data=True` unless you explicitly wish to not have the data saved to disk.')

        if hyperparameter_tune:
            logger.log(30, 'Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')

        if dist_ip_addrs is None:
            dist_ip_addrs = []

        if search_options is None:
            search_options = dict()

        if hyperparameters is None:
            hyperparameters = {
               'NN': {'num_epochs': 500},
               'GBM': {'num_boost_round': 10000},
               'CAT': {'iterations': 10000},
               'RF': {'n_estimators': 300},
               'XT': {'n_estimators': 300},
               'KNN': {},
               'custom': ['GBM'],
             }

        # Process kwargs to create feature generator, trainer, schedulers, searchers for each model:
        output_directory = setup_outputdir(output_directory)  # Format directory name
        feature_generator_type = kwargs.get('feature_generator_type', AutoMLFeatureGenerator)
        feature_generator_kwargs = kwargs.get('feature_generator_kwargs', {})
        feature_generator = feature_generator_type(**feature_generator_kwargs) # instantiate FeatureGenerator object
        id_columns = kwargs.get('id_columns', [])
        trainer_type = kwargs.get('trainer_type', AutoTrainer)
        nthreads_per_trial, ngpus_per_trial = setup_compute(nthreads_per_trial, ngpus_per_trial)
        num_train_rows = len(train_data)
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limits? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            num_bagging_folds = min(10, max(5, math.floor(num_train_rows / 100)))
            stack_ensemble_levels = min(1, max(0, math.floor(num_train_rows / 750)))

        if num_bagging_sets is None:
            if num_bagging_folds >= 2:
                if time_limits is not None:
                    num_bagging_sets = 20
                else:
                    num_bagging_sets = 1
            else:
                num_bagging_sets = 1

        label_count_threshold = kwargs.get('label_count_threshold', 10)
        if num_bagging_folds is not None:  # Ensure there exist sufficient labels for stratified splits across all bags
            label_count_threshold = max(label_count_threshold, num_bagging_folds)

        time_limits_orig = copy.deepcopy(time_limits)
        time_limits_hpo = copy.deepcopy(time_limits)

        if num_bagging_folds >= 2 and (time_limits_hpo is not None):
            time_limits_hpo = time_limits_hpo / (1 + num_bagging_folds * (1 + stack_ensemble_levels))
        time_limits_hpo, num_trials = setup_trial_limits(time_limits_hpo, num_trials, hyperparameters)  # TODO: Move HPO time allocation to Trainer

        if (num_trials is not None) and hyperparameter_tune and (num_trials == 1):
            hyperparameter_tune = False
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limits is too small for hyperparameter_tune, setting to False.')

        if holdout_frac is None:
            # Between row count 5,000 and 25,000 keep 0.1 holdout_frac, as we want to grow validation set to a stable 2500 examples
            if num_train_rows < 5000:
                holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
            else:
                holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

            if hyperparameter_tune:
                holdout_frac = min(0.2, holdout_frac * 2)  # We want to allocate more validation data for HPO to avoid overfitting

        # Add visualizer to NN hyperparameters:
        if (visualizer is not None) and (visualizer != 'none') and ('NN' in hyperparameters):
            hyperparameters['NN']['visualizer'] = visualizer

        eval_metric = get_metric(eval_metric, problem_type, 'eval_metric')
        stopping_metric = get_metric(stopping_metric, problem_type, 'stopping_metric')

        # All models use the same scheduler:
        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'num_trials': num_trials,
            'time_out': time_limits_hpo,
            'visualizer': visualizer,
            'time_attr': 'epoch',  # For tree ensembles, each new tree (ie. boosting round) is considered one epoch
            'reward_attr': 'validation_performance',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
            # This is a fix for now. But we need to separate between scheduler
            # (mainly FIFO and Hyperband) and searcher. Currently, most searchers
            # only work with FIFO, and Hyperband works only with random searcher,
            # but this will be different in the future.
            if search_strategy == 'hyperband':
                # Currently, HyperbandScheduler only supports random searcher
                scheduler_options['searcher'] = 'random'
        else:
            # TODO: Check that search_strategy is a subclass of TaskScheduler
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler_options = (scheduler, scheduler_options)  # wrap into tuple
        learner = Learner(path_context=output_directory, label=label, problem_type=problem_type, objective_func=eval_metric, stopping_metric=stopping_metric,
                          id_columns=id_columns, feature_generator=feature_generator, trainer_type=trainer_type,
                          label_count_threshold=label_count_threshold)
        learner.fit(X=train_data, X_test=tuning_data, scheduler_options=scheduler_options,
                    hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                    holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
                    hyperparameters=hyperparameters, time_limit=time_limits_orig, save_data=cache_data, verbosity=verbosity)
        return TabularPredictor(learner=learner)
