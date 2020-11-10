"""Text Prediction Model based on BERT"""
from typing import Union, Optional
import collections
import logging
import pandas as pd
import os
from ...constants import BINARY, REGRESSION
from ..abstract.abstract_model import AbstractModel
from ... import metrics
from ...features.feature_metadata import FeatureMetadata, R_OBJECT, R_INT, R_FLOAT, R_CATEGORY,\
    S_TEXT_NGRAM, S_TEXT_AS_CATEGORY


# Import autogluon text specific dependencies
try:
    from autogluon.text.text_prediction.text_prediction\
        import ag_text_prediction_params, merge_params, get_column_properties,\
        infer_problem_type, infer_eval_stop_log_metrics
    from autogluon.text.text_prediction.dataset import TabularDataset
    from autogluon.text.text_prediction.models.basic_v1 import BertForTextPredictionBasic
except ImportError:
    raise ImportError('autogluon.text has not been installed. '
                      'You may try to install "autogluon.text" first by running. '
                      '`python3 -m pip install autogluon.text`')


logger = logging.getLogger(__name__)


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
        print('output_directory=', self.path)
        ch = input()
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

    def _predict_proba(self, X, **kwargs):
        """Predict the probability from the model.

        Parameters
        ----------
        X
            The data input. It can be either a pandas dataframe or a
        **kwargs
            Other keyword arguments.

        Returns
        -------
        y_pred_proba
            The predicted probability
        """
        if self.problem_type == REGRESSION:
            return self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] == 2:
                return y_pred_proba[:, 1]
            elif y_pred_proba.shape[1] > 2:
                raise ValueError('The shape of the predicted probability does not match '
                                 'with the inferred problem type. '
                                 'Inferred problem type={}, predicted proba shape={}'.format(
                    self.problem_type, y_pred_proba.shape))
            else:
                return y_pred_proba
        else:
            return y_pred_proba

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_feature_kwargs=dict(
                valid_raw_types=[R_OBJECT, R_INT, R_FLOAT],
                ignored_type_group_raw=[R_CATEGORY],
                ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
            )
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _fit(self,
             X_train: pd.DataFrame, y_train: pd.Series,
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
        # Get arguments from kwargs
        verbosity = kwargs.get('verbosity', 2)
        num_cpus = kwargs.get('num_cpus')
        num_gpus = kwargs.get('num_gpus')
        hyperparameters = None
        if hyperparameters is None:
            hyperparameters = ag_text_prediction_params.create('default')
        elif isinstance(hyperparameters, str):
            hyperparameters = ag_text_prediction_params.create(hyperparameters)
        else:
            base_params = ag_text_prediction_params.create('default')
            hyperparameters = merge_params(base_params, hyperparameters)
        column_properties = self._build_model(X_train=X_train,
                          y_train=y_train,
                          X_val=X_val,
                          y_val=y_val,
                          hyperparameters=hyperparameters)
        # Insert the label column
        X_train.insert(len(X_train.columns), self._label_column_name, y_train)
        X_val.insert(len(X_val.columns), self._label_column_name, y_val)
        scheduler_options = hyperparameters['hpo_params']['scheduler_options']
        search_strategy = hyperparameters['hpo_params']['search_strategy']
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
                         resource={'num_cpus': num_cpus,
                                   'num_gpus': num_gpus},
                         time_limits=time_limit,
                         search_strategy=search_strategy,
                         search_options=hyperparameters['hpo_params']['search_options'],
                         scheduler_options=scheduler_options,
                         num_trials=hyperparameters['hpo_params']['num_trials'],
                         console_log=verbosity > 2,
                         ignore_warning=verbosity <= 2)
