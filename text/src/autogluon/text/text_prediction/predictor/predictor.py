import copy
import logging
import math
import pprint
import time
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from autogluon_contrib_nlp.utils.registry import Registry
from autogluon_contrib_nlp.utils.misc import logging_config

from autogluon.core.dataset import TabularDataset
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pkl, load_pd
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import setup_outputdir, setup_compute, setup_trial_limits,\
    default_holdout_frac
from ..presets import ag_text_presets, merge_params
from ..infer_types import infer_problem_type, infer_column_problem_types


logger = logging.getLogger()  # return root logger


class TextPredictor:
    """AutoGluon Predictor predicts values in a column of a tabular dataset that contains text
    (classification or regression).

    Parameters
    ----------
    label : str
        Name of the column that contains the target variable to predict.
    problem_type : str, default = None
        Type of prediction problem, i.e. is this a binary/multiclass classification or regression
        problem (options: 'binary', 'multiclass', 'regression').
        If `problem_type = None`, the prediction problem type is inferred based on the
        label-values in provided dataset.
    eval_metric : function or str, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        AutoGluon tunes factors such as hyper-parameters, early-stopping, ensemble-weights, etc.
        in order to improve this metric on validation data.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification and
        'root_mean_squared_error' for regression.

        Otherwise, options for classification:
            ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision',
             'precision_macro', 'precision_micro',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
             'recall_weighted', 'log_loss', 'pac_score']
        Options for regression:
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error',
             'median_absolute_error', 'r2', 'spearmanr', 'pearsonr']
        For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

        You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
    path : str, default = None
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonTextModel/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
    """

    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=2
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self._label = label
        self._problem_type = problem_type
        self._eval_metric = eval_metric
        self._path = setup_outputdir(path)
        self._model = None

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    @property
    def problem_type(self):
        return self._problem_type

    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            feature_columns=None,
            column_types=None,
            num_cpus=None,
            num_gpus=None,
            seed=None):
        """Fit the predictor

        Parameters
        ----------
        train_data
            The training data
        tuning_data
            The tuning data
        time_limit
            The time limits
        presets
            The user can specify the presets of the hyper-parameters.
        hyperparameters
            The hyper-parameters
        feature_columns
            Specify which columns in the data
        column_types
            The provided type of the columns
        num_cpus
            The number of CPUs to use for each trial
        num_gpus
            The number of GPUs to use for each trial
        seed
            The seed of the experiment

        Returns
        -------
        self
        """
        if presets is not None:
            preset_hparams = ag_text_presets.create(presets)
        else:
            preset_hparams = ag_text_presets.create('default')
        hyperparameters = merge_params(preset_hparams, hyperparameters)
        if seed is not None:
            hyperparameters['seed'] = seed
        seed = hyperparameters['seed']
        if isinstance(self._label, str):
            label_columns = [self._label]
        else:
            label_columns = list(self._label)
        # Get the training and tuning data as pandas dataframe
        if not isinstance(train_data, pd.DataFrame):
            train_data = load_pd.load(train_data)
        if feature_columns is None:
            all_columns = list(train_data.columns)
            feature_columns = [ele for ele in all_columns if ele not in label_columns]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            for col in feature_columns:
                assert col not in label_columns, 'Feature columns and label columns cannot overlap.'
                assert col in train_data.columns,\
                    'Feature columns must be in the pandas dataframe! Received col = "{}", ' \
                    'all columns = "{}"'.format(col, train_data.columns)
            all_columns = feature_columns + label_columns
        train_data = train_data[all_columns]
        # Get tuning data
        if tuning_data is not None:
            if not isinstance(tuning_data, pd.DataFrame):
                tuning_data = load_pd.load(tuning_data)
            tuning_data = tuning_data[all_columns]
        else:
            if hyperparameters['misc']['holdout_frac'] is not None:
                holdout_frac = hyperparameters['misc']['holdout_frac']
            else:
                num_trials = hyperparameters['hpo_params']['num_trials']
                if num_trials == 1:
                    holdout_frac = default_holdout_frac(len(train_data), False)
                else:
                    # For HPO, we will need to use a larger held-out ratio
                    holdout_frac = default_holdout_frac(len(train_data), True)
            train_data, tuning_data = train_test_split(train_data,
                                                       test_size=holdout_frac,
                                                       random_state=np.random.RandomState(seed))
        column_types, problem_type = infer_column_problem_types(train_data, tuning_data,
                                                                label_columns=label_columns,
                                                                problem_type=self._problem_type,
                                                                provided_column_types=column_types)
        self._problem_type = problem_type
        model_hparams = hyperparameters['models']['MultimodalTextModel']
        if model_hparams['backend'] == 'gluonnlp_v0':
            from ..mx.models import MultiModalTextModel
            self._model = MultiModalTextModel(column_types=column_types,
                                              feature_columns=feature_columns,
                                              label_columns=label_columns,
                                              problem_type=self._problem_type,
                                              eval_metric=self._eval_metric,
                                              output_directory=self._path,
                                              logger=logger,
                                              search_space=model_hparams['search_spaces'])
            self._model.train(train_data=train_data,
                              tuning_data=tuning_data,
                              num_cpus=num_cpus,
                              num_gpus=num_gpus,
                              time_limit=time_limit)
        else:
            raise NotImplementedError("Currently, we only support using "
                                      "the autogluon-contrib-nlp and MXNet "
                                      "as the backend of AutoGluon-Text. In the future, "
                                      "we will support other models.")
        return self

    def predict(self, dataset, as_pandas=False):
        """Predict the

        Returns
        -------
        output
            Array of predictions. One element corresponds to the prediction value of one
        """
        pass

    def predict_proba(self, dataset, as_pandas=False):
        """Predict the probability from the input

        Returns
        -------

        """
        pass

    def predict_feature(self):
        """Extract the feature from the neural network

        Returns
        -------

        """
        pass

    def save(self):
        pass

    @classmethod
    def load(cls, path, verbosity=2):
        set_logger_verbosity(verbosity,
                             logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")
