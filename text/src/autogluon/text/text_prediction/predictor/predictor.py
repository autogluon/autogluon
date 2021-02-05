import logging
import os
from sklearn.model_selection import train_test_split
import numpy as np
import json
import pandas as pd

from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.utils import setup_outputdir, default_holdout_frac

from ..presets import ag_text_presets, merge_params
from ..infer_types import infer_column_problem_types, printable_column_type_string
from ..metrics import infer_eval_log_metrics
from .. import constants as _C

logger = logging.getLogger()  # return root logger


class TextPredictor:
    """AutoGluon TextPredictor predicts values in a column of a tabular dataset that contains texts
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
    warn_if_exist
        Whether to raise warning if the path exists
    """

    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=2,
            warn_if_exist=True
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self._label = label
        self._problem_type = problem_type
        self._eval_metric = eval_metric
        self._path = setup_outputdir(path, warn_if_exist=warn_if_exist)
        self._model = None
        self._fit_called = False
        self._backend = None

    def set_verbosity(self, target_verbosity):
        self.verbosity = target_verbosity
        set_logger_verbosity(self.verbosity, logger=logger)

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
            num_trials=None,
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
        num_trials
            The number of trials. By default, we will use the provided number of trials in the
            hyperparameters or presets. This will overwrite the provided value.
        seed
            The seed of the experiment

        Returns
        -------
        self
        """
        assert self._fit_called is False
        if presets is not None:
            preset_hparams = ag_text_presets.create(presets)
        else:
            preset_hparams = ag_text_presets.create('default')
        hyperparameters = merge_params(preset_hparams, hyperparameters)
        if seed is not None:
            hyperparameters['seed'] = seed
        seed = hyperparameters['seed']
        if num_trials is not None:
            hyperparameters['hpo_params']['num_trials'] = num_trials
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
        self._eval_metric, log_metrics = infer_eval_log_metrics(problem_type=problem_type,
                                                                eval_metric=self._eval_metric)
        has_text_column = False
        for k, v in column_types.items():
            if v == _C.TEXT:
                has_text_column = True
                break
        if not has_text_column:
            raise AssertionError('No Text Column is found! This is currently not supported by '
                                 'the TextPrediction task. You may try to use '
                                 'autogluon.tabular.TabularPredictor.\n'
                                 'The inferred column properties of the training data is {}'
                                 .format(train_data))
        logger.log(25, 'Problem Type="{}"'.format(problem_type))
        logger.log(25, printable_column_type_string(column_types))
        self._problem_type = problem_type
        model_hparams = hyperparameters['models']['MultimodalTextModel']
        self._backend = model_hparams['backend']
        if model_hparams['backend'] == 'gluonnlp_v0':
            from ..mx.models import MultiModalTextModel
            self._model = MultiModalTextModel(column_types=column_types,
                                              feature_columns=feature_columns,
                                              label_columns=label_columns,
                                              problem_type=self._problem_type,
                                              eval_metric=self._eval_metric,
                                              log_metrics=log_metrics,
                                              output_directory=self._path)
            self._model.train(train_data=train_data,
                              tuning_data=tuning_data,
                              num_cpus=num_cpus,
                              num_gpus=num_gpus,
                              search_space=model_hparams['search_space'],
                              hpo_params=hyperparameters['hpo_params'],
                              time_limit=time_limit,
                              seed=seed,
                              verbosity=self.verbosity)
        else:
            raise NotImplementedError("Currently, we only support using "
                                      "the autogluon-contrib-nlp and MXNet "
                                      "as the backend of AutoGluon-Text. In the future, "
                                      "we will support other models.")
        return self

    def evaluate(self, data, metrics=None, stochastic_chunk=None, num_repeat=None):
        """ Report the predictive performance evaluated for a given dataset.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or `pandas.DataFrame`
            This Dataset must also contain the label-column with the same column-name as specified during `fit()`.
            If str is passed, `valid_data` will be loaded using the str value as the file path.
        metrics : str or List[str] or None
            Name of metric or a list of names of metrics to report.
            If it is not given, we will return the score of the stored eval_metric.
        stochastic_chunk
            Whether to use stochastic chunk
        num_repeat
            The number of repeats
        return_type :
            Can be list or dict

        Returns
        -------
        ret : a single number or a dict of metric --> metric scores
            Output
        """
        return self._model.evaluate(data, metrics=metrics,
                                    stochastic_chunk=stochastic_chunk,
                                    num_repeat=num_repeat)

    def predict(self, dataset, stochastic_chunk=None, num_repeat=None,
                as_pandas=False):
        """Get the prediction from

        Returns
        -------
        output
            Array of predictions. One element corresponds to the prediction value of one
        """
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        output = self._model.predict(dataset,
                                     stochastic_chunk=stochastic_chunk,
                                     num_repeat=num_repeat)
        if as_pandas:
            output = pd.DataFrame({self.label: output})
        return output

    def predict_proba(self, dataset, stochastic_chunk=None, num_repeat=None, as_pandas=False):
        """Predict the probability from the input

        Returns
        -------
        output
        """
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        output = self._model.predict_proba(dataset,
                                           stochastic_chunk=stochastic_chunk,
                                           num_repeat=num_repeat)
        if as_pandas:
            output = pd.DataFrame(output)
        return output

    def extract_embedding(self, dataset, stochastic_chunk=None, num_repeat=None, as_pandas=False):
        """Extract the feature from the neural network

        Returns
        -------
        output
        """
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        output = self._model.extract_embedding(dataset,
                                               stochastic_chunk=stochastic_chunk,
                                               num_repeat=num_repeat)
        if as_pandas:
            output = pd.DataFrame({self.label: output})
        return output

    def save(self, dir_path):
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, 'assets.json'), 'w') as of:
            if not isinstance(self._eval_metric, str):
                eval_metric = self._eval_metric.name
            else:
                eval_metric = self._eval_metric
            json.dump({'eval_metric': eval_metric,
                       'label': self._label,
                       'problem_type': self._problem_type,
                       'backend': self._backend}, of)
        self._model.save(os.path.join(dir_path, 'model'))

    @classmethod
    def load(cls, dir_path):
        assert os.path.exists(dir_path), f'"{dir_path}" does not exist. You may check the path again.'
        with open(os.path.join(dir_path, 'assets.json'), 'r') as in_f:
            assets = json.load(in_f)
        ret = cls(eval_metric=assets['eval_metric'],
                  label=assets['label'],
                  problem_type=assets['problem_type'],
                  path=dir_path,
                  warn_if_exist=False)
        ret._backend = assets['backend']
        if ret._backend == 'gluonnlp_v0':
            from ..mx.models import MultiModalTextModel
            model = MultiModalTextModel.load(os.path.join(dir_path, 'model'))
        else:
            raise NotImplementedError(f'Backend = "{ret._backend}" is not supported.')
        ret._model = model
        return ret
