
from typing import Optional
import logging
import os
import pickle

import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.features.types import R_OBJECT, S_IMAGE_PATH
from autogluon.core.models import AbstractModel
from autogluon.core.utils import get_cpu_count, get_gpu_count, try_import_mxnet, try_import_autogluon_vision

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
            'problem_types': [BINARY, MULTICLASS],  # Does not support regression
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
        try_import_mxnet()
        try_import_autogluon_vision()
        from autogluon.vision import ImagePredictor
        params = self._get_model_params()

        if self.problem_type == REGRESSION:
            raise AssertionError(f'ImagePredictorModel does not support `problem_type="{REGRESSION}"`')

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

        verbosity_image = max(0, verbosity - 1)
        root_logger = logging.getLogger()
        root_log_level = root_logger.level
        # TODO: ImagePredictor doesn't use problem_type in any way at present.
        #  It also doesn't error or warn if problem_type is not one it expects.
        self.model = ImagePredictor(
            problem_type=self.problem_type,
            path=self.path,
            # eval_metric=self.eval_metric,  # TODO: Vision only works with accuracy
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
        root_logger.setLevel(root_log_level)  # Reset log level

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            # FIXME: This probably won't work
            # return self.model.predict(X, as_pandas=False)
            raise AssertionError(f'ImagePredictorModel does not support `problem_type="{REGRESSION}"`')

        y_pred_proba = self.model.predict_proba(X, as_pandas=False)
        return self._convert_proba_to_unified_form(y_pred_proba)

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
        num_gpus = get_gpu_count()
        return num_cpus, num_gpus
