"""Wrapper of the MultiModalPredictor."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd

from autogluon.common.features.types import (
    R_CATEGORY,
    R_FLOAT,
    R_INT,
    R_OBJECT,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_AS_CATEGORY,
    S_TEXT_NGRAM,
    S_TEXT_SPECIAL,
)
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_autogluon_multimodal
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


class MultiModalPredictorModel(AbstractModel):
    ag_key = "AG_AUTOMM"
    ag_name = "MultiModalPredictor"
    _NN_MODEL_NAME = "automm_model"

    def __init__(self, **kwargs):
        """Wrapper of autogluon.multimodal.MultiModalPredictor.

        The features can be a mix of
        - image column
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

        .. versionadded:: 0.3.0
        """
        super().__init__(**kwargs)
        self._label_column_name = None
        self._load_model = None  # Whether to load inner model when loading.

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "valid_stacker": False,
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    # FIXME: Enable parallel bagging once AutoMM supports being run within Ray without hanging
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {"fold_fitting_strategy": "sequential_local"}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _set_default_params(self):
        super()._set_default_params()
        try_import_autogluon_multimodal()

    def preprocess_fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """
        Preprocessing training and validation data.
        This method is a placeholder for inheriting models to override with more complex functionality if needed.
        """
        X = self.preprocess(X=X, **kwargs)
        if X_val is not None:
            X_val = self.preprocess(X=X_val, **kwargs)
        return X, y, X_val, y_val

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        time_limit: Optional[int] = None,
        sample_weight=None,
        **kwargs,
    ):
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
        sample_weight
            The weights of the samples
        kwargs
            Other keyword arguments

        """
        try_import_autogluon_multimodal()
        from autogluon.multimodal import MultiModalPredictor

        # Decide name of the label column
        if "label" in X.columns:
            label_col_id = 0
            while True:
                self._label_column_name = "label{}".format(label_col_id)
                if self._label_column_name not in X.columns:
                    break
                label_col_id += 1
        else:
            self._label_column_name = "label"

        X, y, X_val, y_val = self.preprocess_fit(X=X, y=y, X_val=X_val, y_val=y_val)
        params = self._get_model_params()
        max_features = params.pop(
            "_max_features", None
        )  # FIXME: `_max_features` is a hack. Instead use ag_args_fit and make generic
        num_features = len(X.columns)
        if max_features is not None and num_features > max_features:
            raise AssertionError(
                f"Feature count ({num_features}) is greater than max allowed features ({max_features}) for {self.name}. Skipping model... "
                f"To increase the max allowed features, specify the value via the `_max_features` parameter "
                f"(Fully ignore by specifying `None`. "
                f"`_max_features` is experimental and will likely change API without warning in future releases."
            )

        # Get arguments from kwargs
        verbosity = kwargs.get("verbosity", 2)
        if verbosity <= 2:
            enable_progress_bar = False
        else:
            enable_progress_bar = True
        num_gpus = kwargs.get("num_gpus", None)
        if sample_weight is not None:  # TODO: support
            logger.log(
                15,
                "sample_weight not yet supported for MultiModalPredictorModel, "
                "this model will ignore them in training.",
            )

        # Need to deep copy to avoid altering outer context
        X = X.copy()
        X.insert(len(X.columns), self._label_column_name, y)
        if X_val is not None:
            X_val = X_val.copy()
            X_val.insert(len(X_val.columns), self._label_column_name, y_val)

        column_types = self._construct_column_types()

        verbosity_text = max(0, verbosity - 1)
        root_logger = logging.getLogger("autogluon")
        root_log_level = root_logger.level
        # in self.save(), the model is saved to automm_nn_path
        automm_nn_path = os.path.join(self.path, self._NN_MODEL_NAME)
        self.model = MultiModalPredictor(
            label=self._label_column_name,
            problem_type=self.problem_type,
            path=automm_nn_path,
            eval_metric=self.eval_metric,
            verbosity=verbosity_text,
            enable_progress_bar=enable_progress_bar,
        )

        if num_gpus is not None:
            params["env.num_gpus"] = num_gpus
        presets = params.pop("presets", None)
        seed = params.pop("seed", 0)

        self.model.fit(
            train_data=X,
            tuning_data=X_val,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=params,
            column_types=column_types,
            seed=seed,
        )

        self.model.set_verbosity(verbosity)
        root_logger.setLevel(root_log_level)  # Reset log level

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        self.model._enable_progress_bar = False
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
            automm_nn_path = os.path.join(path, self._NN_MODEL_NAME)
            self.model.save(automm_nn_path)
            logger.log(15, f"\tSaved AutoMM model weights and model hyperparameters to '{automm_nn_path}'.")
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._load_model:
            try_import_autogluon_multimodal()
            from autogluon.multimodal import MultiModalPredictor

            model.model = MultiModalPredictor.load(os.path.join(path, cls._NN_MODEL_NAME))
        model._load_model = None
        return model

    def _get_memory_size(self) -> int:
        """Return the memory size by calculating the total number of parameters.

        Returns
        -------
        memory_size
            The total memory size in bytes.
        """
        total_size = self.model.model_size * 1e6  # convert from megabytes to bytes

        return total_size

    def _get_default_resources(self):
        num_cpus = ResourceManager.get_cpu_count()
        num_gpus = min(
            ResourceManager.get_gpu_count_torch(), 1
        )  # Use single gpu training by default. Consider to revise it later.
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available=False) -> Dict[str, int]:
        return {
            "num_cpus": 1,
            "num_gpus": 1,
        }

    def _construct_column_types(self) -> dict:
        # Construct feature types input to MultimodalPredictor
        features_image_path = set(self._feature_metadata.get_features(required_special_types=[S_IMAGE_PATH]))
        features_text = set(self._feature_metadata.get_features(required_special_types=[S_TEXT]))
        features_categorical = set(self._feature_metadata.get_features(valid_raw_types=[R_CATEGORY]))
        features_numerical = set(self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT]))

        key_map = {
            "image_path": features_image_path,
            "text": features_text,
            "categorical": features_categorical,
            "numerical": features_numerical,
        }

        features = self._feature_metadata.get_features()

        column_types = {}
        for feature in features:
            for key in ["image_path", "text", "categorical", "numerical"]:
                if feature in key_map[key]:
                    column_types[feature] = key
                    break
        return column_types

    def _more_tags(self):
        # `can_refit_full=False` because MultiModalPredictor does not communicate how to train until the best epoch in refit_full.
        return {"can_refit_full": False}

    @classmethod
    def _class_tags(cls):
        return {"handles_text": True}
