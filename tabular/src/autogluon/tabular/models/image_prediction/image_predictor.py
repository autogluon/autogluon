import logging

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_OBJECT, S_IMAGE_PATH
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, SOFTCLASS

from ..automm.automm_model import MultiModalPredictorModel

logger = logging.getLogger(__name__)


# TODO: Handle multiple image columns?
# TODO: Handle multiple images in a single image column?
# TODO: Consider fully replacing with MultiModalPredictorModel
#  first check if the null handling in this class provides value
class ImagePredictorModel(MultiModalPredictorModel):
    """
    MultimodalPredictor that only uses image features.
    Currently only supports 1 image column, with 1 image per sample.
    Additionally has special null image handling to improve performance in the presence of null images (aka image path of '')
        Note: null handling has not been compared to the built-in null handling of MultimodalPredictor yet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dummy_pred_proba = None  # Dummy value to predict if image is NaN
        self._image_col_name = None

    @property
    def _has_predict_proba(self):
        return self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]

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

    def preprocess_fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X, y, X_val, y_val = super().preprocess_fit(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)
        X_features = list(X.columns)
        if len(X_features) != 1:
            raise AssertionError(f'ImagePredictorModel only supports one image feature, but {len(X_features)} were given: {X_features}')
        self._image_col_name = X_features[0]
        null_indices = X[self._image_col_name] == ''

        # TODO: Consider some kind of weighting of the two options so there isn't a harsh cutoff at 50
        # FIXME: What if all rows in a class are null? Will probably crash.
        if null_indices.sum() > 50:
            self._dummy_pred_proba = self._compute_dummy_pred_proba(y[null_indices])  # FIXME: Do this one for better results
        else:
            # Not enough null to get a confident estimate of null label average, instead use all data average
            self._dummy_pred_proba = self._compute_dummy_pred_proba(y)

        if null_indices.sum() > 0:
            X = X[~null_indices]
            y = y[~null_indices]

        if X_val is not None:
            null_indices_val = X_val[self._image_col_name] == ''
            if null_indices_val.sum() > 0:
                X_val = X_val[~null_indices_val]
                y_val = y_val[~null_indices_val]

        return X, y, X_val, y_val

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        pred_method = self.model.predict_proba if self._has_predict_proba else self.model.predict
        # TODO: Add option to crash if null is present for faster predict_proba
        null_indices = X[self._image_col_name] == ''
        if null_indices.sum() > 0:
            if self.num_classes is None:
                y_pred_proba = np.zeros(len(X))
            else:
                y_pred_proba = np.zeros((len(X), self.num_classes))
            X = X.reset_index(drop=True)
            null_indices = X[self._image_col_name] == ''
            X = X[~null_indices]
            null_indices_rows = list(null_indices[null_indices].index)
            non_null_indices_rows = list(null_indices[~null_indices].index)
            y_pred_proba[null_indices_rows] = self._dummy_pred_proba
            y_pred_proba[non_null_indices_rows] = pred_method(X, as_pandas=False)
        else:
            y_pred_proba = pred_method(X, as_pandas=False)
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
