import logging
import os
from sklearn.model_selection import train_test_split
from typing import Optional
import numpy as np
import json
import pandas as pd

from autogluon.core.constants import BINARY
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
    """AutoGluon TextPredictor predicts values in a column of a tabular dataset that contains text fields
    (classification or regression). TabularPredictor can also do this but it uses an ensemble of many types of models and may featurize text.
    TextPredictor instead directly fits individual Transformer neural network models directly to the raw text (which are also capable of handling additional numeric/categorical columns).
    We generally recommend TabularPredictor if your table contains numeric/categorical columns and TextPredictor if your table contains only text columns, but you may easily try both.
    In fact, `TabularPredictor.fit(..., hyperparameters='multimodal')` will train a TextPredictor along with many tabular models and ensemble them together.

    Parameters
    ----------
    label : str
        Name of the column that contains the target variable to predict.
    problem_type : str, default = None
        Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
        If `problem_type = None`, the prediction problem type is inferred based on the label-values in provided dataset.
    eval_metric : function or str, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        AutoGluon tunes factors such as hyperparameters, early-stopping, etc. in order to improve this metric on validation data.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression.

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
    verbosity : int, default = 3
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
    warn_if_exist : bool, default = True
        Whether to raise warning if the specified path already exists.
    """

    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=3,
            warn_if_exist=True
    ):
        self.verbosity = verbosity
        if self.verbosity is not None:
            set_logger_verbosity(self.verbosity, logger=logger)
        self._label = label
        self._problem_type = problem_type
        self._eval_metric = eval_metric
        self._path = setup_outputdir(path, warn_if_exist=warn_if_exist)
        self._model = None
        self._fit_called = False
        self._backend = None

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity
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

    @property
    def positive_class(self):
        """Name of the class label that will be mapped to 1. This is only meaningful for binary classification problems.

        It is useful for computing metrics such as F1 which require a positive and negative class.
        You may refer to https://en.wikipedia.org/wiki/F-score for more details.
        In binary classification, :class:`TextPredictor.predict_proba(as_multiclass=False)`
        returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        if self.problem_type != BINARY:
            logger.warning(f"Warning: Attempted to retrieve positive class label in a non-binary problem. Positive class labels only exist in binary classification. Returning None instead. self.problem_type is '{self.problem_type}' but positive_class only exists for '{BINARY}'.")
            return None
        else:
            return self.class_labels[1]

    @property
    def class_labels(self):
        """The original name of the class labels.

        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...

        This function returns the original names of these raw labels.

        Returns
        -------
        ret
            List that contain the class names. It will be None if the predictor is not solving a classification problem.
        """
        return self._model.class_labels

    @property
    def class_labels_internal(self):
        """The internal integer labels.

        For example, if the possible labels are ["entailment", "contradiction", "neutral"],
        the internal labels can be [0, 1, 2]

        Returns
        -------
        ret
            List that contains the internal integer labels. It will be None if the predictor is not solving a classification problem.
        """
        if self.class_labels is None:
            return None
        return list(range(len(self.class_labels)))

    @property
    def class_labels_internal_map(self):
        """The map that projects label names to the internal ids. For example,
        if the internal labels are ["entailment", "contradiction", "neutral"] and the
        internal ids are [0, 1, 2], the label mapping will be
        {"entailment": 0, "contradiction": 1, "neutral": 2}

        Returns
        -------
        ret
            The label mapping dictionary. It will be None if the predictor is not solving a classification problem.
        """
        if self.class_labels is None:
            return None
        return {k: v for k, v in zip(self.class_labels, self.class_labels_internal)}

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
        """
        Fit Transformer models to predict label column of a data table based on the other columns (which may contain text or numeric/categorical features).

        Parameters
        ----------
        train_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            Table of the training data, which is similar to a pandas DataFrame.
            If str is passed, `train_data` will be loaded using the str value as the file path.
        tuning_data : str or :class:`TabularDataset` or :class:`pd.DataFrame`, default = None
            Another dataset containing validation data reserved for tuning processes such as early stopping and hyperparameter tuning.
            This dataset should be in the same format as `train_data`.
            If str is passed, `tuning_data` will be loaded using the str value as the file path.
            Note: final model returned may be fit on `tuning_data` as well as `train_data`. Do not provide your evaluation test data here!
            If `tuning_data = None`, `fit()` will automatically hold out some random validation examples from `train_data`.
        time_limit : int, default = None
            Approximately how long `fit()` should run for (wallclock time in seconds).
            If not specified, `fit()` will run until the model has completed training.
        presets : str, default = None
            Presets are pre-registered configurations that control training (hyperparameters and other aspects).
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            Print all available presets via `autogluon.text.list_presets()`.
            Some notable presets include:
                - "best_quality": produce the most accurate overall predictor (regardless of its efficiency).
                - "medium_quality_faster_train": produce an accurate predictor but take efficiency into account (this is the default preset).
                - "lower_quality_fast_train": produce a predict that is quick to train and make predictions with, even if its accuracy is worse.
        hyperparameters : dict, default = None
            The hyperparameters of the `fit()` function, which affect the resulting accuracy of the trained predictor.
            Experienced AutoGluon users can use this argument to specify neural network hyperparameter values/search-spaces as well as which hyperparameter-tuning strategy should be employed. See the "Text Prediction" tutorials for examples.
        column_types : dict, default = None
            The type of data in each table column can be specified via a dictionary that maps the column name to its data type.
            For example: `column_types = {"item_name": "text", "brand": "text", "product_description": "text", "height": "numerical"}` may be used for a table with columns: "item_name", "brand", "product_description", and "height".
            If None, column_types will be automatically inferred from the data.
            The current supported types are:
            - "text": each row in this column contains text (sentence, paragraph, etc.).
            - "numerical": each row in this column contains a number.
            - "categorical": each row in this column belongs to one of K categories.
        num_cpus : int, default = None
            The number of CPUs to use for each training run (i.e. one hyperparameter-tuning trial).
        num_gpus : int, default = None
            The number of GPUs to use to use for each training run (i.e. one hyperparameter-tuning trial). We recommend at least 1 GPU for TextPredictor as its neural network models are computationally intensive.
        num_trials : int, default = None
            If hyperparameter-tuning is used, specifies how many HPO trials should be run (assuming `time_limit` has not been exceeded).
            By default, this is the provided number of trials in the `hyperparameters` or `presets`.
            If specified here, this value will overwrite the value in `hyperparameters['tune_kwargs']['num_trials']`.
        plot_results : bool, default = None
            Whether to plot intermediate results from training. If None, will be decided based on the environment in which `fit()` is run.
        holdout_frac : float, default = None
            Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless `tuning_data = None`).
            Default value (if None) is selected based on the number of rows in the training data and whether hyperparameter-tuning is utilized.
        seed : int, default = 0
            The random seed to use for this training run. If None, no seed will be specified and repeated runs will produce different results.

        Returns
        -------
        :class:`TextPredictor` object. Returns self.
        """
        assert self._fit_called is False
        verbosity = self.verbosity
        if verbosity is None:
            verbosity = 3
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
        if isinstance(train_data, str):
            train_data = load_pd.load(train_data)
        if not isinstance(train_data, pd.DataFrame):
            raise AssertionError(f'train_data is required to be a pandas DataFrame, but was instead: {type(train_data)}')
        all_columns = list(train_data.columns)
        feature_columns = [ele for ele in all_columns if ele not in label_columns]
        train_data = train_data[all_columns]
        # Get tuning data
        if tuning_data is not None:
            if isinstance(tuning_data, str):
                tuning_data = load_pd.load(tuning_data)
            if not isinstance(tuning_data, pd.DataFrame):
                raise AssertionError(f'tuning_data is required to be a pandas DataFrame, but was instead: {type(tuning_data)}')
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
        logger.info('Problem Type="{}"'.format(problem_type))
        logger.info(printable_column_type_string(column_types))
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
                              verbosity=verbosity)
        else:
            raise NotImplementedError("Currently, we only support using "
                                      "the autogluon-contrib-nlp and MXNet "
                                      "as the backend of AutoGluon-Text. In the future, "
                                      "we will support other models.")
        logger.info(f'Training completed. Auto-saving to "{self.path}". '
                       f'For loading the model, you can use'
                       f' `predictor = TextPredictor.load("{self.path}")`')
        self.save(self.path)
        return self

    def evaluate(self, data, metrics=None):
        """ Report the predictive performance evaluated over a given dataset.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or `pandas.DataFrame`
            This dataset must also contain the `label` with the same column-name as previously specified.
            If str is passed, `data` will be loaded using the str value as the file path.
        metrics : str or List[str], default = None
            Name of metric or a list of multiple metric names to report (options are the same as for `eval_metric`).
            If None, we only return the score for the stored `eval_metric`.

        Returns
        -------
        metrics_values : float or dict
            The metrics computed on the data. This is a single value if there is only one metric and is a dictionary of {metric_name --> value} if there are multiple metrics.
        """
        return self._model.evaluate(data, metrics=metrics)

    def predict(self, data, as_pandas=True):
        """
        Use trained model to produce predictions of `label` column values for new data.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training Dataset and follow same format (except for the `label` column).
            If str is passed, `data` will be loaded using the str value as the file path.
        as_pandas : bool, default = True
            Whether to return the output as a :class:`pd.Series` (True) or :class:`np.ndarray` (False).

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        assert self._model is not None, 'Model does not seem to have been constructed. Have you called fit(), or load()?'
        output = self._model.predict(data)
        if as_pandas:
            if isinstance(data, pd.DataFrame):
                index = data.index
            else:
                index = None
            output = pd.Series(data=output, index=index, name=self.label)
        return output

    def predict_proba(self, data, as_pandas=True, as_multiclass=True):
        """
        Use trained model to produce predicted class probabilities rather than class-labels (if task is classification).
        If `predictor.problem_type` is regression, this functions identically to `predict`, returning the same output.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training dataset and follow same format (except for the `label` column).
            If str is passed, `data` will be loaded using the str value as the file path.
        as_pandas : bool, default = True
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass : bool, default = True
            Whether to return the probability of all labels or just return the probability of the positive class for binary classification problems.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        assert self._model is not None,\
            'Model does not seem to have been constructed. ' \
            'Have you called fit(), or load()?'
        output = self._model.predict_proba(data)
        if not as_multiclass:
            if self.problem_type == BINARY:
                output = output[:, 1]
        if as_pandas:
            if isinstance(data, pd.DataFrame):
                index = data.index
            else:
                index = None
            if output.ndim == 1:
                output = pd.Series(output, index=index, name=self.label)
            else:
                output = pd.DataFrame(output, index=index, columns=self.class_labels)
        return output

    def extract_embedding(self, data, as_pandas=False):
        """
        Extract intermediate feature representations of a row from the trained neural network.

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to extract embeddings for. Should contain same column names as training dataset and follow same format (except for the `label` column).
            If str is passed, `data` will be loaded using the str value as the file path.
        as_pandas : bool, default = False
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined by the neural network's architecture.
        """
        assert self._model is not None, 'Model does not seem to have been constructed. ' \
                                        'Have you called fit(), or load()?'
        output = self._model.extract_embedding(data)
        if as_pandas:
            if isinstance(data, pd.DataFrame):
                index = data.index
            else:
                index = None
            output = pd.DataFrame(output, index=index)
        return output

    def save(self, path):
        """
        Save this Predictor to file in directory specified by `path`.
        The relevant files will be saved in two parts:

        - PATH/text_predictor_assets.json
            Contains the configuration information of this Predictor.
        - PATH/saved_model
            Contains the model weights and other features

        Note that :meth:`TextPredictor.fit` already saves the predictor object automatically (we do not recommend modifying the Predictor object yourself as it tracks many trained models).

        Parameters
        ----------
        path, str
            The path to directory in which to save this Predictor.
        """
        assert self._model is not None, 'Model does not seem to have been constructed.' \
                                        ' Have you called fit(), or load()?'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'text_predictor_assets.json'), 'w') as of:
            json.dump({'backend': self._backend,
                       'label': self._label}, of)
        self._model.save(os.path.join(path, 'saved_model'))

    @classmethod
    def load(cls, path: str, verbosity: int = None):
        """
        Load a TextPredictor object previously produced by `fit()` from file and returns this object. It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        Parameters
        ----------
        path : str
            The path to directory in which this Predictor was previously saved.
        verbosity : int, default = None
            Sets the verbosity level of this Predictor after it is loaded.
            Valid values range from 0 (least verbose) to 4 (most verbose).
            If None, logging verbosity is not changed from existing values.
            Specify larger values to see more information printed when using Predictor during inference, smaller values to see less information.
            Refer to TextPredictor init for more information.
        """
        assert os.path.exists(path), f'"{path}" does not exist. You may check the path again.'
        with open(os.path.join(path, 'text_predictor_assets.json'), 'r') as in_f:
            assets = json.load(in_f)
        backend = assets['backend']
        label = assets['label']
        if backend == 'gluonnlp_v0':
            from ..mx.models import MultiModalTextModel
            model = MultiModalTextModel.load(os.path.join(path, 'saved_model'))
        else:
            raise NotImplementedError(f'Backend = "{backend}" is not supported.')
        predictor: TextPredictor = cls(label=label,
                                       problem_type=model._problem_type,
                                       eval_metric=model._eval_metric,
                                       path=path,
                                       verbosity=verbosity,
                                       warn_if_exist=False)
        predictor._backend = assets['backend']
        predictor._model = model
        return predictor
