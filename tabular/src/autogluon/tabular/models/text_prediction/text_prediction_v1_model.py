"""Text Prediction Model based on Pretrained Language Model. Version 1"""
from typing import Optional
import collections
import logging
import pandas as pd
import os
import random
import numpy as np

from autogluon.core.utils import get_cpu_count, get_gpu_count
from autogluon.core.utils.exceptions import NoGPUError
from autogluon.core.utils.utils import default_holdout_frac

from ..abstract.abstract_model import AbstractModel
from ...features.feature_metadata import R_OBJECT, R_INT, R_FLOAT, R_CATEGORY, \
    S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL


logger = logging.getLogger(__name__)


AG_TEXT_IMPORT_ERROR = 'autogluon.text has not been installed. ' \
                       'You may try to install "autogluon.text" first by running. ' \
                       '`python3 -m pip install autogluon.text`'


class TextPredictionV1Model(AbstractModel):
    nn_model_name = 'text_nn'

    def __init__(self, **kwargs):
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
        """
        super().__init__(**kwargs)
        self._label_column_name = None
        self._feature_generator = None

    def _preprocess(self, X, fit=False, **kwargs):
        from ...features.feature_metadata import R_CATEGORY
        if fit:
            from ...features.generators import BulkFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
            # TODO: This feature generator improves scores for TextPrediction when rare categories are present. This should be fixed in TextPrediction.
            self._feature_generator = BulkFeatureGenerator(generators=[
                [
                    CategoryFeatureGenerator(features_in=self.feature_metadata.get_features(valid_raw_types=[R_CATEGORY]), minimum_cat_count=1),
                    IdentityFeatureGenerator(features_in=self.feature_metadata.get_features(invalid_raw_types=[R_CATEGORY])),
                ],
            ], verbosity=0)
            self._feature_generator.fit(X)
        return self._feature_generator.transform(X)

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
        if X_val is not None:
            concat_feature_df = pd.concat([X_train, X_val])
            concat_feature_df.reset_index(drop=True, inplace=True)
            concat_label_df = pd.DataFrame({self._label_column_name: pd.concat([y_train, y_val])})
            concat_label_df.reset_index(drop=True, inplace=True)
        else:
            concat_feature_df = X_train
            concat_label_df = pd.DataFrame({self._label_column_name: y_train})
        feature_column_properties = get_column_properties(
            df=concat_feature_df,
            metadata=None,
            label_columns=None,
            provided_column_properties=None
        )

        label_column_property = get_column_properties(
            df=concat_label_df,
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
            get_features_kwargs=dict(
                valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
                invalid_special_types=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {'valid_stacker': False}
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    def _set_default_params(self):
        try:
            from autogluon.text.text_prediction.dataset import TabularDataset
            from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)
        super()._set_default_params()
        self.params = ag_text_prediction_params.create('default_no_hpo')

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
            import mxnet as mx
            from autogluon.text.text_prediction.dataset import TabularDataset, random_split_train_val
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

        if resource['num_gpus'] == 0:
            raise NoGPUError(f'\tNo GPUs available to train {self.name}. Resources: {resource}')

        # Set seed
        seed = self.params.get('seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            mx.random.seed(seed)

        X_train = self.preprocess(X_train, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        column_properties = self._build_model(X_train=X_train,
                                              y_train=y_train,
                                              X_val=X_val,
                                              y_val=y_val,
                                              hyperparameters=self.params)
        # Insert the label column
        X_train.insert(len(X_train.columns), self._label_column_name, y_train)
        if X_val is not None:
            X_val.insert(len(X_val.columns), self._label_column_name, y_val)
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
        if X_val is None:
            # FIXME: v0.1 Update TextPrediction to use all training data in refit_full
            holdout_frac = default_holdout_frac(len(X_train), True)
            X_train, X_val = random_split_train_val(X_train, valid_ratio=holdout_frac)
        train_data = TabularDataset(X_train,
                                    column_properties=column_properties,
                                    label_columns=self._label_column_name)
        logger.info('Train Dataset:')
        logger.info(train_data)
        tuning_data = TabularDataset(X_val,
                                     column_properties=column_properties,
                                     label_columns=self._label_column_name)
        logger.info('Tuning Dataset:')
        logger.info(tuning_data)
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
        model = self.model
        self.model = None
        # save this AbstractModel object without NN weights
        path = super().save(path=path, verbose=verbose)
        self.model = model

        text_nn_path = os.path.join(path, self.nn_model_name)
        model.save(text_nn_path)
        logger.log(15, f"\tSaved Text NN weights and model hyperparameters to '{text_nn_path}'.")

        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        try:
            from autogluon.text.text_prediction.dataset import TabularDataset
            from autogluon.text.text_prediction.models.basic_v1 import BertForTextPredictionBasic
        except ImportError:
            raise ImportError(AG_TEXT_IMPORT_ERROR)

        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        model.model = BertForTextPredictionBasic.load(os.path.join(path, cls.nn_model_name))
        return model

    def get_memory_size(self) -> int:
        """Return the memory size by calculating the total number of parameters.

        Returns
        -------
        memory_size
            The total memory size in bytes.
        """
        total_size = 0
        for k, v in self.model.net.collect_params().items():
            total_size += np.dtype(v.dtype).itemsize * np.prod(v.shape)
        return total_size

    def _get_default_resources(self):
        num_cpus = get_cpu_count()
        num_gpus = get_gpu_count()
        return num_cpus, num_gpus
