import copy
import logging
import math
import pprint
import time

import numpy as np
import pandas as pd

from autogluon_contrib_nlp.utils.registry import Registry
from autogluon_contrib_nlp.utils.misc import logging_config

from autogluon.core.dataset import TabularDataset
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import setup_outputdir, setup_compute, setup_trial_limits,\
    default_holdout_frac
from ..config import ag_text_config, merge_params


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

    @property
    def path(self):
        return self._path

    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            column_types=None,
            num_cpus=None,
            num_gpus=None):
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
            The user can specify the presets for providing the
        hyperparameters
            The hyper-parameters
        column_types
            Type of the columns
        num_cpus
            The number of CPUs to use for each trial
        num_gpus
            The number of GPUs to use for each trial

        Returns
        -------
        self
        """
        if presets is not None:
            preset_hparams = ag_text_config.create(presets)
            hyperparameters = merge_params(preset_hparams, hyperparameters)


        return self



    def predict(self, dataset, as_pandas=False):
        """Predict the

        Returns
        -------
        output
            Array of predictions. One element corresponds to the prediction value of one
        """

    def predict_proba(self, dataset, as_pandas=False):
        """Predict the probability from the input

        Returns
        -------

        """

    def predict_feature(self):
        """Extract the feature from the neural network

        Returns
        -------

        """

    def save(self):
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)

    @classmethod
    def load(cls, path, verbosity=2):
        set_logger_verbosity(verbosity,
                             logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")
