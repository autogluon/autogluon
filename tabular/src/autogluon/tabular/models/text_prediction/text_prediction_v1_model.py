"""Text Prediction Model based on BERT"""
from typing import Union, Optional
import collections
import logging
import pandas as pd
import os
from autogluon.core.constants import BINARY, REGRESSION
from autogluon.core import metrics

from ..abstract.abstract_model import AbstractModel
from ...features.feature_metadata import FeatureMetadata, R_OBJECT, R_INT, R_FLOAT, R_CATEGORY,\
    S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_TEXT


logger = logging.getLogger(__name__)


AG_TEXT_IMPORT_ERROR = 'autogluon.text has not been installed. ' \
                       'You may try to install "autogluon.text" first by running. ' \
                       '`python3 -m pip install autogluon.text`'


class TextPredictionV1Model(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str,
                 eval_metric: Optional[Union[str, metrics.Scorer]] = None,
                 num_classes=None,
                 stopping_metric: Optional[Union[str, metrics.Scorer]] = None,
                 model=None,
                 hyperparameters=None, features=None,
                 feature_metadata: FeatureMetadata = None,
                 debug=0, **kwargs):
        """The TextPredictionV1Model.

        The features can be a mix of
        - text column
        - categorical column
        - numerical column

        The labels can be categorical or numerical.

        Parameters
        ----------
        path
            The directory to store the modeling outputs.
        name
            Name of subdirectory inside path where model will be saved.
        problem_type
            Type of problem that this model will handle.
            Valid options: ['binary', 'multiclass', 'regression'].
        eval_metric
            The evaluation metric.
        num_classes
            The number of classes.
        stopping_metric
            The stopping metric.
        model
            The internal model object.
        hyperparameters
            The hyperparameters of the model
        features
            Names of the features.
        feature_metadata
            The feature metadata.
        debug
            Whether to turn on debug mode
        kwargs
            Other arguments
        """
        super().__init__(path=path, name=name, problem_type=problem_type, eval_metric=eval_metric,
                         num_classes=num_classes, stopping_metric=stopping_metric, model=model,
                         hyperparameters=hyperparameters, features=features,
                         feature_metadata=feature_metadata, debug=debug, **kwargs)
        self._label_column_name = None

    def _build_model(self, X_train, y_train, X_val, y_val, hyperparameters):
        try:
            from autogluon.text.text_prediction.text_prediction \
                import ag_text_prediction_params, merge_params, get_column_properties, \
                infer_problem_type, infer_eval_stop_log_metrics
            from autogluon.text.text_prediction.models.basic_v1 import BertForTextPredictionBasic
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)

        # Decide the name of the label column
        if 'label' in X_train.columns:
            label_col_id = 0
            while True:
                self._label_column_name = 'label{}'.format(label_col_id)
                if self._label_column_name not in X_train.columns:
                    break
                label_col_id += 1
        else:
            self._label_column_name = 'label'
        feature_column_properties = get_column_properties(
            pd.concat([X_train, X_val]),
            metadata=None,
            label_columns=None,
            provided_column_properties=None
        )
        label_column_property = get_column_properties(
            pd.DataFrame({self._label_column_name: pd.concat([y_train, y_val])}),
            metadata=None,
            label_columns=None,
            provided_column_properties=None
        )
        column_properties = collections.OrderedDict(list(feature_column_properties.items()) +
                                                    list(label_column_property.items()))
        problem_type, label_shape = infer_problem_type(column_properties=column_properties,
                                                       label_col_name=self._label_column_name)
        eval_metric, stopping_metric, log_metrics =\
            infer_eval_stop_log_metrics(problem_type,
                                        label_shape=label_shape,
                                        eval_metric=self.eval_metric,
                                        stopping_metric=self.stopping_metric)
        search_space = hyperparameters['models']['BertForTextPredictionBasic']['search_space']
        self.model = BertForTextPredictionBasic(column_properties=column_properties,
                                                feature_columns=list(X_train.columns),
                                                label_columns=[self._label_column_name],
                                                problem_types=[problem_type],
                                                label_shapes=[label_shape],
                                                stopping_metric=stopping_metric,
                                                log_metrics=log_metrics,
                                                output_directory=os.path.join(self.path, self.name),
                                                logger=logger,
                                                base_config=None,
                                                search_space=search_space)
        return column_properties

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_feature_kwargs=dict(
                valid_raw_types=[R_INT, R_FLOAT],
                valid_special_types=[S_TEXT],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _set_default_params(self):
        try:
            from autogluon.text.text_prediction.dataset import TabularDataset
            from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)
        super()._set_default_params()
        self.params = ag_text_prediction_params.create('default')

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             time_limit: Optional[int] = None, **kwargs):
        """The internal fit function

        Parameters
        ----------
        X_train
            Features of the training dataset
        y_train
            Labels of the training dataset
        X_val
            Features of the validation dataset
        y_val
            Labels of the validation dataset
        time_limit
            The time limits for the fit function
        kwargs
            Other keyword arguments

        """
        try:
            from autogluon.text.text_prediction.dataset import TabularDataset
            from autogluon.text.text_prediction.text_prediction import get_recommended_resource
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)

        # Get arguments from kwargs
        verbosity = kwargs.get('verbosity', 2)
        num_cpus = kwargs.get('num_cpus', None)
        num_gpus = kwargs.get('num_gpus', None)

        # Infer resource
        resource = get_recommended_resource(nthreads_per_trial=num_cpus,
                                            ngpus_per_trial=num_gpus)
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        column_properties = self._build_model(X_train=X_train,
                                              y_train=y_train,
                                              X_val=X_val,
                                              y_val=y_val,
                                              hyperparameters=self.params)
        print('X_train=', X_train)
        # Insert the label column
        X_train[self._label_column_name] = y_train
        X_val[self._label_column_name] = y_val
        scheduler_options = self.params['hpo_params']['scheduler_options']
        search_strategy = self.params['hpo_params']['search_strategy']
        if scheduler_options is None:
            scheduler_options = dict()
        if search_strategy.endswith('hyperband'):
            # Specific defaults for hyperband scheduling
            scheduler_options['reduction_factor'] = scheduler_options.get(
                'reduction_factor', 4)
            scheduler_options['grace_period'] = scheduler_options.get(
                'grace_period', 10)
            scheduler_options['max_t'] = scheduler_options.get(
                'max_t', 50)
        train_data = TabularDataset(X_train,
                                    column_properties=column_properties,
                                    label_columns=self._label_column_name)
        tuning_data = TabularDataset(X_val,
                                     column_properties=column_properties,
                                     label_columns=self._label_column_name)
        self.model.train(train_data=train_data,
                         tuning_data=tuning_data,
                         resource=resource,
                         time_limits=time_limit,
                         search_strategy=search_strategy,
                         search_options=self.params['hpo_params']['search_options'],
                         scheduler_options=scheduler_options,
                         num_trials=self.params['hpo_params']['num_trials'],
                         console_log=verbosity >= 2,
                         ignore_warning=verbosity < 2)

    def save(self, path: str = None, verbose=True) -> str:
        if path is None:
            path = self.path
        logger.log(15, f'Save to {path}.')
        self.model.save(path)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        try:
            from autogluon.text.text_prediction.dataset import TabularDataset
            from autogluon.text.text_prediction.models.basic_v1 import BertForTextPredictionBasic
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)

        logger.log(15, f'Load from {path}.')
        cls.model = BertForTextPredictionBasic.load(path)
