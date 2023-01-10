import logging
import time

import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.common.features.types import R_INT, R_FLOAT, R_CATEGORY, R_OBJECT, S_IMAGE_PATH, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL
from autogluon.core.utils.try_import import try_import_vowpalwabbit
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS, \
    PROBLEM_TYPES_CLASSIFICATION, PROBLEM_TYPES_REGRESSION
from autogluon.core.utils.exceptions import TimeLimitExceeded
from .vowpalwabbit_utils import VWFeaturesConverter

logger = logging.getLogger(__name__)


class VowpalWabbitModel(AbstractModel):
    """
    VowpalWabbit Model: https://vowpalwabbit.org/

    VowpalWabbit Command Line args: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-line-arguments

    """
    model_internals_file_name = 'model-internals.pkl'

    # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
    CLASSIFICATION_LOSS_FUNCTIONS = ['logistic', 'hinge']
    REGRESSION_LOSS_FUNCTIONS = ['squared', 'quantile', 'poisson', 'classic']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model = None  # Used for saving and loading internal model file

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.Series:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._features_converter = VWFeaturesConverter()
            self._feature_metadata_dict = self._feature_metadata.to_dict()
        # self._feature_metadata contains the information related to features metadata.
        X_series = self._features_converter.convert_features_to_vw_format(X, self._feature_metadata_dict)
        return X_series

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             time_limit=None,
             verbosity=2,
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        time_start = time.time()
        try_import_vowpalwabbit()
        from vowpalwabbit import pyvw
        seed = 0  # Random seed

        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type not in PROBLEM_TYPES_REGRESSION + PROBLEM_TYPES_CLASSIFICATION:
            raise TypeError(f"Vowpal Wabbit does not support {self.problem_type}")

        # Certain parameters like passes are passed as hyperparameters but are not used
        # while initialising the model.
        # passes: Used as epochs

        params = self._get_model_params()
        params['loss_function'] = params.get('loss_function', self._get_default_loss_function())
        passes = params.pop('passes')

        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X_series = self.preprocess(X, is_train=True)

        self._validate_loss_function(loss_function=params['loss_function'])

        # VW expects label from 1 to N for Binary and Multiclass classification problems
        # AutoGluon does label encoding from 0 to N-1, hence we increment the value of y by 1
        if self.problem_type != REGRESSION:
            y = y.apply(lambda row: row + 1)
        y = y.astype(str) + ' '

        # Concatenate y and X to get the training data in VW format
        final_training_data = y + X_series
        final_training_data = final_training_data.tolist()

        extra_params = {
            'cache_file': 'train.cache',
            'holdout_off': True,
        }

        if verbosity <= 3:
            extra_params['quiet'] = True

        # Initialize the model
        if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Predicting-probabilities#multi-class---oaa
            extra_params['oaa'] = self.num_classes
            extra_params['probabilities'] = True
        self.model = pyvw.vw(**params, **extra_params)

        time_start_fit = time.time()
        if time_limit is not None:
            time_limit_fit = time_limit - (time_start_fit - time_start) - 0.3  # Account for 0.3s overhead
            if time_limit_fit <= 0:
                raise TimeLimitExceeded
        else:
            time_limit_fit = None

        # Train the model
        np.random.seed(seed)
        epoch = 0

        for epoch in range(1, passes + 1):
            # TODO: Add Early Stopping support via validation
            self._train_single_epoch(training_data=final_training_data)
            if time_limit_fit is not None and epoch < passes:
                time_fit_used = time.time() - time_start_fit
                time_fit_used_per_epoch = time_fit_used / epoch
                time_left = time_limit_fit - time_fit_used
                if time_left <= (time_fit_used_per_epoch*2):
                    logger.log(30, f'\tEarly stopping due to lack of time. Fit {epoch}/{passes} passes...')
                    break

        self.params_trained['passes'] = epoch

    def _train_single_epoch(self, training_data):
        row_order = np.arange(0, len(training_data))
        row_order = np.random.permutation(row_order)
        for row_i in row_order:
            row = training_data[row_i]
            self.model.learn(row)

    def _validate_loss_function(self, loss_function):
        # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
        if loss_function:
            if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                assert loss_function in self.CLASSIFICATION_LOSS_FUNCTIONS, \
                    f'For {self.problem_type} problem, VW supports: {self.CLASSIFICATION_LOSS_FUNCTIONS}. ' \
                    f'Got loss_function:{loss_function}'
            elif self.problem_type in PROBLEM_TYPES_REGRESSION:
                assert loss_function in self.REGRESSION_LOSS_FUNCTIONS, \
                    f'For {self.problem_type} problem, VW supports: {self.REGRESSION_LOSS_FUNCTIONS}. ' \
                    f'Got loss_function:{loss_function}'

    def _get_default_loss_function(self) -> str:
        # Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions
        if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            return 'logistic'
        else:
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

            model_load_params = f" -i {file_path} --quiet"
            if model.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                model_load_params += " --probabilities --loss_function=logistic"
            if params['sparse_weights']:
                model_load_params += " --sparse_weights"

            model.model = pyvw.vw(model_load_params)
        model._load_model = None
        return model

    def _predict_proba(self, X, **kwargs):
        # Preprocess the set of X features
        X = self.preprocess(X, **kwargs)

        y_pred_proba = np.array([self.model.predict(row) for row in X])
        return self._convert_proba_to_unified_form(y_pred_proba)

    def get_memory_size(self) -> int:
        # TODO: Can be improved further to make it more accurate
        # Returning 5MB as the value
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
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
            ignored_type_group_special=[S_IMAGE_PATH, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL]
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

    def _more_tags(self):
        # `can_refit_full=True` because best epoch is communicated at end of `_fit`: `self.params_trained['passes'] = epoch`
        return {'can_refit_full': True}

    @classmethod
    def _class_tags(cls):
        return {'handles_text': True}
