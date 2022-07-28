"""Wrapper of the Text Prediction in AutoGluon Text."""
from typing import Dict, Optional
import logging
import os
import pandas as pd


from autogluon.common.features.types import R_OBJECT, R_INT, R_FLOAT, R_CATEGORY, \
    S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_IMAGE_PATH
from autogluon.core.constants import REGRESSION
from autogluon.core.utils import get_cpu_count, get_gpu_count_torch, try_import_autogluon_text
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


class TextPredictorModel(AbstractModel):
    nn_model_name = 'text_nn'

    def __init__(self, **kwargs):
        """Wrapper of autogluon.text.TextPredictor.

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
        """
        super().__init__(**kwargs)
        self._label_column_name = None
        self._load_model = None  # Whether to load inner model when loading.

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_IMAGE_PATH],
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
        super()._set_default_params()
        try_import_autogluon_text()

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             time_limit: Optional[int] = None,
             sample_weight=None,
             **kwargs):
        """The internal fit function

        Parameters
        ----------
        X
            Features of the training dataset
        y
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
        try_import_autogluon_text()
        from autogluon.text import TextPredictor

        # Decide name of the label column
        if 'label' in X.columns:
            label_col_id = 0
            while True:
                self._label_column_name = 'label{}'.format(label_col_id)
                if self._label_column_name not in X.columns:
                    break
                label_col_id += 1
        else:
            self._label_column_name = 'label'
        X_train = self.preprocess(X, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # Get arguments from kwargs
        verbosity = kwargs.get('verbosity', 2)
        num_cpus = kwargs.get('num_cpus', None)
        num_gpus = kwargs.get('num_gpus', None)
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for TextPredictorModel, this model will ignore them in training.")

        X_train.insert(len(X_train.columns), self._label_column_name, y)
        if X_val is not None:
            X_val.insert(len(X_val.columns), self._label_column_name, y_val)

        verbosity_text = max(0, verbosity - 1)
        root_logger = logging.getLogger('autogluon')
        root_log_level = root_logger.level
        self.model = TextPredictor(label=self._label_column_name,
                                   problem_type=self.problem_type,
                                   path=self.path,
                                   eval_metric=self.eval_metric,
                                   verbosity=verbosity_text)
        params = self._get_model_params()
        presets = params.pop('presets', None)
        self.model.fit(train_data=X_train,
                       tuning_data=X_val,
                       time_limit=time_limit,
                       num_gpus=num_gpus,
                       num_cpus=num_cpus,
                       presets=presets,
                       hyperparameters=params,
                       seed=params.pop('seed', 0))
        self.model.set_verbosity(verbosity)
        root_logger.setLevel(root_log_level)  # Reset log level

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            return self.model.predict(X, as_pandas=False)

        y_pred_proba = self.model.predict_proba(X, as_pandas=False)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def save(self, path: str = None, verbose=True) -> str:
        self._load_model = self.model is not None
        __model = self.model
        self.model = None
        # save this AbstractModel object without NN weights
        path = super().save(path=path, verbose=verbose)
        self.model = __model

        if self._load_model:
            text_nn_path = os.path.join(path, self.nn_model_name)
            self.model.save(text_nn_path)
            logger.log(15, f"\tSaved Text NN weights and model hyperparameters to '{text_nn_path}'.")
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._load_model:
            try_import_autogluon_text()
            from autogluon.text import TextPredictor
            model.model = TextPredictor.load(os.path.join(path, cls.nn_model_name))
        model._load_model = None
        return model

    def get_memory_size(self) -> int:
        """Return the memory size by calculating the total number of parameters.

        Returns
        -------
        memory_size
            The total memory size in bytes.
        """
        total_size = sum(param.numel() for param in self.model._predictor._model.parameters())

        return total_size

    def _get_default_resources(self):
        num_cpus = get_cpu_count()
        num_gpus = get_gpu_count_torch()
        return num_cpus, num_gpus

    def get_minimum_resources(self) -> Dict[str, int]:
        return {
            'num_cpus': 1,
            'num_gpus': 1,
        }

    def _more_tags(self):
        # `can_refit_full=False` because TextPredictor does not communicate how to train until the best epoch in refit_full.
        return {'can_refit_full': False}
