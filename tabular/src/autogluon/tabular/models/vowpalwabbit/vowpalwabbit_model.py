import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.common.features.types import *
from autogluon.core.utils.try_import import try_import_vowpalwabbit
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS, SOFTCLASS, QUANTILE,\
    PROBLEM_TYPES_CLASSIFICATION, PROBLEM_TYPES_REGRESSION
from autogluon.core.constants import AG_ARGS_FIT
from .vowpalwabbit_utils import VWFeaturesConverter


class VowpalWabbitModel(AbstractModel):
    """
    VowpalWabbit Model: https://vowpalwabbit.org/

    VowpalWabbit Command Line args: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-line-arguments

    """
    model_internals_file_name = 'model-internals.pkl'
    namespace_separator_key = 'namespace_separator'
    use_different_namespace_key = 'use_different_namespace'
    UNSUPPORTED_PROBLEM_TYPES = [SOFTCLASS, QUANTILE]
    sparse_weights = 'sparse_weights'

    # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
    CLASSIFICATION_LOSS_FUNCTIONS = ['logistic', 'hinge']
    REGRESSION_LOSS_FUNCTIONS = ['squared', 'quantile', 'poisson', 'classic']

    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self.labels = None
        self.oaa = None
        self._load_model = None  # Used for saving and loading internal model file

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.Series:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)
        namespace_separator = kwargs.get(self.namespace_separator_key, '')
        # self._feature_metadata contains the information related to features metadata.
        X_series = VWFeaturesConverter(
            namespace_separator=namespace_separator).convert_features_to_vw_format(X,
                                                                                   self._feature_metadata.to_dict())
        return X_series

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             X_val=None,
             y_val=None,
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')

        try_import_vowpalwabbit()
        from vowpalwabbit import pyvw

        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type in self.UNSUPPORTED_PROBLEM_TYPES:
            raise TypeError(f"Vowpal Wabbit does not support {self.problem_type}")

        # Certain parameters like passes and use_different_namespace are passed as hyperparameters but are not used
        # while initialising the model.
        # use_different_namespace: Used for generating different namespace for features
        # passes: Used as epoch

        ag_params = self._get_ag_params()
        params = self._get_model_params()
        namespace_separator = ''
        if self.use_different_namespace_key in params:
            namespace_separator = '|'
            params.pop(self.use_different_namespace_key)
        params['loss_function'] = params.get('loss_function', self._get_default_loss_function())
        print(f'Hyperparameters: {params}')
        epochs = params['passes']
        params.pop('passes')

        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X_series = self.preprocess(X, is_train=True, namespace_separator=namespace_separator)

        self._validate_loss_function(hyperparams=params)

        # VW expects label from 1 to N for Binary and Multiclass classification problems
        # AutoGluon does label encoding from 0 to N-1, hence we increment the value of y by 1
        if self.problem_type != REGRESSION:
            y = y.apply(lambda row: row + 1)
        y = y.astype(str) + ' '

        # Concatenate y and X to get the training data in VW format
        final_training_data = y + X_series
        final_training_data = final_training_data.tolist()

        # Initialize the model
        if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Predicting-probabilities#multi-class---oaa
            self.labels = list(set(y.tolist()))
            self.oaa: int = len(self.labels)
            print(f'Setting oaa to {self.oaa}')
            self.model = pyvw.vw(
                **params,
                oaa=self.oaa,
                cache_file='train.cache',
                holdout_off=True,
                probabilities=True,  # Only to be used for classification
                quiet=True,
            )
        elif self.problem_type in PROBLEM_TYPES_REGRESSION:
            self.model = pyvw.vw(
                **params,
                cache_file='train.cache',
                holdout_off=True,
                quiet=True,
            )
        # Train the model
        for epoch in range(1, epochs + 1):
            # TODO: Add Early Stopping support
            self._train_single_epoch(training_data=final_training_data)

        print('Exiting the `_fit` method')

    def _train_single_epoch(self, training_data):
        np.random.seed(42)
        row_order = np.arange(0, len(training_data))
        row_order = np.random.permutation(row_order)
        for row_i in row_order:
            row = training_data[row_i]
            self.model.learn(row)

    def _validate_loss_function(self, hyperparams):
        # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
        loss_function = hyperparams.get('loss_function')
        if loss_function:
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                assert loss_function in self.CLASSIFICATION_LOSS_FUNCTIONS, \
                    f'For {self.problem_type} problem, VW supports: {self.CLASSIFICATION_LOSS_FUNCTIONS}. ' \
                    f'Got loss_function:{loss_function}'
            elif self.problem_type in PROBLEM_TYPES_REGRESSION:
                assert loss_function in self.REGRESSION_LOSS_FUNCTIONS, \
                    f'For {self.problem_type} problem, VW supports: {self.REGRESSION_LOSS_FUNCTIONS}. ' \
                    f'Got loss_function:{loss_function}'

    def _get_default_loss_function(self):
        # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
        if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            return 'logistic'
        elif self.problem_type in PROBLEM_TYPES_REGRESSION:
            return 'squared'

    def save(self, path: str = None, verbose=True) -> str:
        """
        AutoGluon by default saves the complete Abstract Model in a pickle file format.
        This includes the internal self.model which is the actual model.
        However, saving VW model in pickle is not possible.
        Hence, we dump the Abstract Model by setting setting self.model as None
        and save self.model as a separate internal file using that model's saving mechanism

        :param path: path where model is to be saved
        :param verbose: verbosity
        :return: path where model is saved
        """

        self._load_model = self.model is not None
        __model = self.model
        self.model = None
        path = super().save(path=path, verbose=verbose)
        self.model = __model
        # Export model
        if self._load_model:
            file_path = path + self.model_internals_file_name
            self.model.save(file_path)
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        There are two files which needs to be loaded.
        First is the Abstract Model pickle dump and second is the internal model file.
        For VW, based on different problem_type/hyperparams, loading arguments will be different
        """
        try_import_vowpalwabbit()
        from vowpalwabbit import pyvw
        # Load Abstract Model. This is without the internal model
        model = super().load(path, reset_paths=reset_paths, verbose=verbose)
        params = model._get_model_params()
        # Load the internal model file
        if model._load_model:
            file_path = path + cls.model_internals_file_name

            model_load_params = f" -i {file_path}"
            if model.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                model_load_params += " --probabilities"
            if params[cls.sparse_weights]:
                model_load_params += " --sparse_weights"

            model.model = pyvw.vw(model_load_params)
        model._load_model = None
        return model

    def _predict_proba(self, X, **kwargs):
        # Preprocess the set of X features
        X = self.preprocess(X, **kwargs)

        X_list = X.values.flatten().tolist()

        if self.problem_type in self.UNSUPPORTED_PROBLEM_TYPES:
            raise TypeError(f'Vowpal Wabbit does not support {self.problem_type}')


        y_pred_proba = np.array([self.model.predict(row) for row in X_list])
        return self._convert_proba_to_unified_form(y_pred_proba)


    def get_info(self) -> dict:
        """
        Returns a dictionary of numerous fields describing the model.
        """
        info = {
            'name': self.name,
            'model_type': type(self).__name__,
            'problem_type': self.problem_type,
            'eval_metric': self.eval_metric.name,
            'stopping_metric': self.stopping_metric.name,
            'fit_time': self.fit_time,
            'num_classes': self.num_classes,
            'quantile_levels': self.quantile_levels,
            'predict_time': self.predict_time,
            'val_score': self.val_score,
            'hyperparameters': self.params,
            'hyperparameters_fit': self.params_trained,
            # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            'hyperparameters_nondefault': self.nondefault_params,
            AG_ARGS_FIT: self.params_aux,
            'num_features': len(self.features) if self.features else None,
            'features': self.features,
            'feature_metadata': self.feature_metadata,
            # 'disk_size': self.get_disk_size(),
            # If internal model cannot be saved in pickle format, then disable get_memory_size
            'memory_size': self.get_memory_size(),  # Memory usage of model in bytes
        }
        return info

    def get_memory_size(self) -> int:
        # TODO: Can be improved further to make it more accurate
        return int(5e6)

    # The `_set_default_params` method defines the default hyperparameters of the model.
    # User-specified parameters will override these values on a key-by-key basis.
    def _set_default_params(self):
        default_params = {
            'passes': 10,
            'bit_precision': 32,
            'ngram': 2,
            'skips': 1,
            'learning_rate': 1,
            'sparse_weights': True,
            'use_different_namespace': False,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        # Ignore the below mentioned special types. Only those features that are not of the below mentioned
        # type are passed to the model for training list are passed features
        extra_auxiliary_params = dict(
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_BINNED]
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            'valid_stacker': False,
            'problem_types': [BINARY, MULTICLASS, REGRESSION],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

