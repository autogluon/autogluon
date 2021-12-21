
from typing import Optional
import logging
import os
import pickle

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_OBJECT, S_IMAGE_PATH
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.utils import get_cpu_count, try_import_autogluon_vision

logger = logging.getLogger(__name__)


# FIXME: Avoid hard-coding 'image' column name
# TODO: Handle multiple image columns?
# TODO: Handle multiple images in a single image column?
# TODO: Add regression support
# TODO: refit_full does not work as expected: It won't use all data, will just split train data internally.
class ImagePredictorModel(AbstractModel):
    """Wrapper of autogluon.vision.ImagePredictor"""
    nn_model_name = 'image_predictor'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_column_name = 'label'

        # Whether to load inner model when loading. Set to None on init as it is only used during save/load
        self._load_model = None

        self._internal_feature_map = None
        self._dummy_pred_proba = None  # Dummy value to predict if image is NaN

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=[R_OBJECT],
                required_special_types=[S_IMAGE_PATH],
            ),
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

    def _preprocess(self, X, fit=False, **kwargs):
        X = super()._preprocess(X, **kwargs)
        if fit:
            X_features = list(X.columns)
            if len(X_features) != 1:
                raise AssertionError(f'ImagePredictorModel only supports one image feature, but {len(X_features)} were given: {X_features}')
            if X_features[0] != 'image':
                self._internal_feature_map = {X_features[0]: 'image'}
        if self._internal_feature_map:
            X = X.rename(columns=self._internal_feature_map)
        from autogluon.vision import ImageDataset
        if isinstance(X, ImageDataset):
            # Use normal DataFrame, otherwise can crash due to `class` attribute conflicts
            X = pd.DataFrame(X)
        return X

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             time_limit: Optional[int] = None,
             sample_weight=None,
             verbosity=2,
             **kwargs):
        # try_import_mxnet()
        try_import_autogluon_vision()
        from autogluon.vision import ImagePredictor
        params = self._get_model_params()

        X = self.preprocess(X, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        if sample_weight is not None:  # TODO: support
            logger.log(15, "\tsample_weight not yet supported for ImagePredictorModel, this model will ignore them in training.")

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if X_val is not None:
            X_val = X_val.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)
        X[self._label_column_name] = y
        if X_val is not None:
            X_val[self._label_column_name] = y_val

        null_indices = X['image'] == ''

        # TODO: Consider some kind of weighting of the two options so there isn't a harsh cutoff at 50
        # FIXME: What if all rows in a class are null? Will probably crash.
        if null_indices.sum() > 50:
            self._dummy_pred_proba = self._compute_dummy_pred_proba(y[null_indices])  # FIXME: Do this one for better results
        else:
            # Not enough null to get a confident estimate of null label average, instead use all data average
            self._dummy_pred_proba = self._compute_dummy_pred_proba(y)

        if null_indices.sum() > 0:
            X = X[~null_indices]
        if X_val is not None:
            null_indices_val = X_val['image'] == ''
            if null_indices_val.sum() > 0:
                X_val = X_val[~null_indices_val]

        verbosity_image = max(0, verbosity - 1)
        # TODO: ImagePredictor doesn't use problem_type in any way at present.
        #  It also doesn't error or warn if problem_type is not one it expects.
        self.model = ImagePredictor(
            problem_type=self.problem_type,
            path=self.path,
            # eval_metric=self.eval_metric,  # TODO: multiclass/binary vision problem works only with accuracy, regression with rmse
            verbosity=verbosity_image
        )

        logger.log(15, f'\tHyperparameters: {params}')

        # FIXME: ImagePredictor crashes if given float time_limit
        if time_limit is not None:
            time_limit = int(time_limit)

        self.model.fit(train_data=X,
                       tuning_data=X_val,
                       time_limit=time_limit,
                       hyperparameters=params,
                       random_state=0)
        # self.model.set_verbosity(verbosity)  # TODO: How to set verbosity of fit predictor?

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        # TODO: Add option to crash if null is present for faster predict_proba
        null_indices = X['image'] == ''
        if null_indices.sum() > 0:
            if self.num_classes is None:
                y_pred_proba = np.zeros(len(X))
            else:
                y_pred_proba = np.zeros((len(X), self.num_classes))
            X = X.reset_index(drop=True)
            null_indices = X['image'] == ''
            X = X[~null_indices]
            null_indices_rows = list(null_indices[null_indices].index)
            non_null_indices_rows = list(null_indices[~null_indices].index)
            y_pred_proba[null_indices_rows] = self._dummy_pred_proba
            y_pred_proba[non_null_indices_rows] = self.model.predict_proba(X, as_pandas=False)
        else:
            y_pred_proba = self.model.predict_proba(X, as_pandas=False)
        return self._convert_proba_to_unified_form(y_pred_proba)

    # TODO: Consider moving to AbstractModel or as a separate function
    # TODO: Test softclass
    def _compute_dummy_pred_proba(self, y):
        num_classes = self.num_classes
        if self.problem_type in [BINARY, MULTICLASS]:
            dummies = pd.get_dummies(y)
            dummy_columns = set(list(dummies.columns))
            if len(dummies.columns) < num_classes:
                for c in range(num_classes):
                    if c not in dummy_columns:
                        dummies[c] = 0
            dummies = dummies[list(range(num_classes))]
            pred_proba_mean = dummies.mean().values

        elif self.problem_type in [REGRESSION, QUANTILE, SOFTCLASS]:
            pred_proba_mean = y.mean()
        else:
            raise NotImplementedError(f'Computing dummy pred_proba is not implemented for {self.problem_type}.')
        return pred_proba_mean

    def _get_default_searchspace(self):
        try_import_autogluon_vision()
        from autogluon.vision.configs import presets_configs
        searchspace = presets_configs.preset_image_predictor['good_quality_fast_inference']['hyperparameters']
        return searchspace

    def save(self, path: str = None, verbose=True) -> str:
        self._load_model = self.model is not None
        __model = self.model
        self.model = None
        # save this AbstractModel object without NN weights
        path = super().save(path=path, verbose=verbose)
        self.model = __model

        if self._load_model:
            image_nn_path = os.path.join(path, self.nn_model_name)
            self.model.save(image_nn_path)
            logger.log(15, f"\tSaved Image NN weights and model hyperparameters to '{image_nn_path}'.")
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._load_model:
            try_import_autogluon_vision()
            from autogluon.vision import ImagePredictor
            model.model = ImagePredictor.load(os.path.join(path, cls.nn_model_name))
        model._load_model = None
        return model

    def get_memory_size(self) -> int:
        """Return the memory size by calculating the total number of parameters.

        Returns
        -------
        memory_size
            The total memory size in bytes.
        """
        return len(pickle.dumps(self.model._classifier, pickle.HIGHEST_PROTOCOL))

    def _get_default_resources(self):
        num_cpus = get_cpu_count()
        try_import_autogluon_vision()
        from autogluon.vision import ImagePredictor
        num_gpus = ImagePredictor._get_num_gpus_available()
        return num_cpus, num_gpus
