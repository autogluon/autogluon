
from typing import Optional
import logging
import os

import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.features.types import R_OBJECT, S_IMAGE_URL
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
    nn_model_name = 'image_nn'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_column_name = 'label'
        self._load_model = None  # Whether to load inner model when loading.
        self._classes = None
        self._internal_feature_map = None

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=[R_OBJECT],
                required_special_types=[S_IMAGE_URL],
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
            The time limit for the fit function
        kwargs
            Other keyword arguments

        """
        try_import_mxnet()
        try_import_autogluon_vision()
        from autogluon.vision import ImagePredictor, ImageDataset
        params = self._get_model_params()

        X = self.preprocess(X, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for ImagePredictorModel, this model will ignore them in training.")

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
        classes = list(y.unique())
        classes.sort()
        self._classes = classes

        X = ImageDataset(X, classes=classes)
        if X_val is not None:
            X_val = ImageDataset(X_val, classes=classes)

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

        y_pred_proba = self.model.predict_proba(X, as_pandas=True)

        ##################
        # FIXME: Vision should just have the standard predict_proba format by default, this is a huge amount of computation to convert
        y_pred_proba = y_pred_proba[['score', 'id', 'image']]
        idx_to_image_map = X[['image']]
        idx_to_image_map = idx_to_image_map.reset_index(drop=True).reset_index(drop=False)
        y_pred_proba = y_pred_proba.merge(idx_to_image_map, on='image')
        y_pred_proba = y_pred_proba.drop(columns=['image'])
        class_preds = []
        for clss in self._classes:
            cur_clss = y_pred_proba[y_pred_proba['id'] == clss]
            cur_clss = cur_clss.set_index('index')['score']
            class_preds.append(cur_clss)
        y_pred_proba = pd.concat(class_preds, axis=1).to_numpy()
        ##################

        return self._convert_proba_to_unified_form(y_pred_proba)

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
        # FIXME: How to get correct size of model?
        total_size = 0
        return total_size

    def _get_default_resources(self):
        num_cpus = get_cpu_count()
        num_gpus = get_gpu_count()
        return num_cpus, num_gpus
