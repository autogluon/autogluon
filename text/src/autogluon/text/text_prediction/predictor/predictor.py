import logging
import os
from sklearn.model_selection import train_test_split
from typing import Optional
import numpy as np
import json
import pandas as pd

from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.utils import setup_outputdir, default_holdout_frac
from autogluon.core.utils.miscs import in_ipynb

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
        Defaults to 'accuracy' for binary, 'accuracy' for multiclass classification, and
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
    def results(self):
        if self._model is not None:
            return self._model.results
        else:
            return None

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def backend(self):
        return self._backend

    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            column_types=None,
            num_cpus=None,
            num_gpus=None,
            num_trials=None,
            plot_results=None,
            holdout_frac=None,
            seed=0):
        """Fit the predictor

        Parameters
        ----------
        train_data
            Table of the training data. It can be a pandas dataframe.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        tuning_data
            Another dataset containing validation data reserved for tuning processes such as early
            stopping and hyperparameter tuning.
            This dataset should be in the same format as `train_data`.
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: final model returned may be fit on `tuning_data` as well as `train_data`.
            Do not provide your evaluation test data here!
            In particular, when `num_bag_folds` > 0 or `num_stack_levels` > 0, models will be
            trained on both `tuning_data` and `train_data`.
            If `tuning_data = None`, `fit()` will automatically hold out some random validation
            examples from `train_data`.
        time_limit
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until the model has completed training.
        presets : str or None, optional, default is None
            Presets defines the pre-registered configurations. By default,
            we will use the "medium_quality_faster_train".
            There are some other options like
                - best_quality: Model with the best quality but will be slower
                                for training and inference
                - medium_quality_faster_train: Model with the medium quality
                - lower_quality_fast_train: Model that can be trained and inferenced very fast compared to the other two options.
            You may try to list all presets via `autogluon.text.ag_text_presets.list_keys()`.
        hyperparameters
            The hyper-parameters of the fit function. This can be used to specify the
            search space and the configuration of the network.
        column_types
            The provided type of the columns. It will be a dictionary that maps the column name
            to the type of the column. For example, it can be
            {"item_name": "text", "brand": "text", "product_description": "text", "height": "numerical"}
            If there are "item_name", "brand", "product_description", and "height" as column names.
            The current supported types are:

            - "text": Text Column
            - "numerical" Numerical Column
            - "categorical" Categorical Column

        num_cpus
            The number of CPUs to use for each trial
        num_gpus
            The number of GPUs to use for each trial
        num_trials
            The number of trials in HPO if HPO is to be used.
            By default, we will use the provided number of trials in the
            hyperparameters or presets. This will overwrite the provided value.
        plot_results
            Whether to plot results. During the fit function.
        holdout_frac
            The holdout frac of the validation set. We will set it automatically
            if it is not specified based on whether we are using hyperparameter search or not.
        seed
            The seed of the experiment. If it is None, no seed will be specified and
            each run will be random. By default, the seed will be 0.

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
        if num_trials is not None:
            hyperparameters['tune_kwargs']['num_trials'] = num_trials
        if isinstance(self._label, str):
            label_columns = [self._label]
        else:
            label_columns = list(self._label)
        # Get the training and tuning data as pandas dataframe
        if not isinstance(train_data, pd.DataFrame):
            train_data = load_pd.load(train_data)
        all_columns = list(train_data.columns)
        feature_columns = [ele for ele in all_columns if ele not in label_columns]
        train_data = train_data[all_columns]
        # Get tuning data
        if tuning_data is not None:
            if not isinstance(tuning_data, pd.DataFrame):
                tuning_data = load_pd.load(tuning_data)
            tuning_data = tuning_data[all_columns]
        else:
            if holdout_frac is None:
                num_trials = hyperparameters['tune_kwargs']['num_trials']
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
                                 'the TextPredictor. You may try to use '
                                 'autogluon.tabular.TabularPredictor.\n'
                                 'The inferred column properties of the training data is {}'
                                 .format(column_types))
        logger.log(25, 'Problem Type="{}"'.format(problem_type))
        logger.log(25, printable_column_type_string(column_types))
        self._problem_type = problem_type
        if 'models' not in hyperparameters or 'MultimodalTextModel' not in hyperparameters['models']:
            raise ValueError('The current TextPredictor only supports "MultimodalTextModel" '
                             'and you must ensure that '
                             'hyperparameters["models"]["MultimodalTextModel"] can be accessed.')
        model_hparams = hyperparameters['models']['MultimodalTextModel']
        self._backend = model_hparams['backend']
        if plot_results is None:
            plot_results = in_ipynb()
        if self._backend == 'gluonnlp_v0':
            import warnings
            warnings.filterwarnings('ignore', module='mxnet')
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
                              tune_kwargs=hyperparameters['tune_kwargs'],
                              time_limit=time_limit,
                              seed=seed,
                              plot_results=plot_results,
                              verbosity=self.verbosity)
        else:
            raise NotImplementedError("Currently, we only support using "
                                      "the autogluon-contrib-nlp and MXNet "
                                      "as the backend of AutoGluon-Text. In the future, "
                                      "we will support other models.")
        logger.log(25, f'Training completed. Auto-saving to "{self.path}". '
                       f'For loading the model, you can use'
                       f' `predictor = TextPredictor.load("{self.path}")`')
        self.save(self.path)
        return self

    def evaluate(self, data, metrics=None):
        """ Report the predictive performance evaluated for a given dataset.

        Parameters
        ----------
        data : str or `pandas.DataFrame`
            This Dataset must also contain the label-column with the same column-name as specified during `fit()`.
            If str is passed, `valid_data` will be loaded using the str value as the file path.
        metrics : str or List[str] or None
            Name of metric or a list of names of metrics to report.
            If it is not given, we will return the score of the stored eval_metric.

        Returns
        -------
        ret : a single number or a dict of metric --> metric scores
            The output metrics. It will be a single value if there is only one metric and
            will be a dictionary of {metric_name --> value} if there are multiple metrics to report.
        """
        return self._model.evaluate(data, metrics=metrics)

    def predict(self, data, as_pandas=True):
        """

        Parameters
        ----------
        data
            The input dataset.
        as_pandas
            Whether the output will be converted to a pandas dataframe

        Returns
        -------
        output
            Array of predictions. One element corresponds to the prediction value of one
        as_pandas
            Whether to convert the output to a pandas dataframe

        """
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        output = self._model.predict(data)
        if as_pandas:
            output = pd.DataFrame({self.label: output})[self.label]
        return output

    def predict_proba(self, data, as_pandas=True):
        """Predict the probability from the input

        Parameters
        ----------
        dataset
            The dataset to use. It can either be a pandas dataframe or the path to the pandas dataframe.
        as_pandas
            Whether to convert the output to pandas dataframe.

        Returns
        -------
        output
            The output matrix contains the probability. It will have shape (#samples, #choices).
        """
        assert self._model is not None,\
            'Model does not seem to have been constructed. ' \
            'Have you called fit(), or load()?'
        output = self._model.predict_proba(data)
        if as_pandas:
            output = pd.DataFrame(output)
        return output

    def extract_embedding(self, data, as_pandas=False):
        """Extract the feature from the neural network

        Parameters
        ----------
        data
            The dataset to extract the embedding. It can either be a pandas dataframe or
            the path to the pandas dataframe.
        as_pandas
            Whether to return a pandas dataframe

        Returns
        -------
        embeddings
            The output will have shape (#sample, #embedding dimension)
        """
        assert self._model is not None, 'Model does not seem to have been constructed. ' \
                                        'Have you called fit(), or load()?'
        output = self._model.extract_embedding(data)
        if as_pandas:
            output = pd.DataFrame(output)
        return output

    def save(self, path):
        """Save the model to directory path

        It will contains two parts:

        - DIR_PATH/text_predictor_assets.json
            Contains the general assets about the configuration of the predictor.
        - DIR_PATH/saved_model
            Contains the model weights and other features


        Parameters
        ----------
        path
            The directory path to save the model artifacts

        """
        assert self._model is not None, 'Model does not seem to have been constructed.' \
                                        ' Have you called fit(), or load()?'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'text_predictor_assets.json'), 'w') as of:
            json.dump({'backend': self._backend,
                       'label': self._label}, of)
        self._model.save(os.path.join(path, 'saved_model'))

    @classmethod
    def load(cls, path):
        """Load from the corresponding path

        Parameters
        ----------
        path
            The path to load the data

        """
        assert os.path.exists(path),\
            f'"{path}" does not exist. You may check the path again.'
        with open(os.path.join(path,
                               'text_predictor_assets.json'), 'r') as in_f:
            assets = json.load(in_f)
        backend = assets['backend']
        label = assets['label']
        if backend == 'gluonnlp_v0':
            from ..mx.models import MultiModalTextModel
            model = MultiModalTextModel.load(os.path.join(path, 'saved_model'))
        else:
            raise NotImplementedError(f'Backend = "{backend}" is not supported.')
        ret = cls(eval_metric=model._eval_metric,
                  label=label,
                  problem_type=model._problem_type,
                  path=path,
                  warn_if_exist=False)
        ret._backend = assets['backend']
        ret._model = model
        return ret
