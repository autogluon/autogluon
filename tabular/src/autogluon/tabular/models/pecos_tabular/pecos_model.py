import numpy as np
import pandas as pd
import tempfile
import pathlib
import logging
import uuid
from typing import Optional

from autogluon.common.features.types import R_BOOL, R_INT, R_FLOAT, R_CATEGORY
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.models import AbstractModel
from .pecos_interface import PecosInterface
from .pecos_utils import clean_str

logger = logging.getLogger(__name__)


class PecosModel(AbstractModel):
    """
    PECOS Custom Model for AutoGluon.
    Wrapper for the PecosInterface to preprocess data and integrate with AutoGluon.
    
    PECOS github: https://github.com/amzn/pecos
    PECOS paper: https://arxiv.org/pdf/2010.05878.pdf
    More info on hyperparameters: https://github.com/amzn/pecos/blob/mainline/pecos/apps/text2text/train.py#L24
    """

    SUPPORTED_MODEL_TYPES = ["XRLinear"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """
        Convert X to the required format for PECOS. Currently only supports tabular data.

        Required text format for each line for PECOS:
        <comma-separated label indices><TAB><space-separated text string>
        Example: l_1<TAB>w_1 w_2 ... w_t
            l_1 can be one of two format:
                (1) the zero-based index for the first relevant label
                (2) double colon separated label index and label relevance
            w_t is the t-th token in the string
        """
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)
        
        feature_metadata = FeatureMetadata.from_df(X)
        num_features = feature_metadata.get_features(valid_raw_types=[R_BOOL, R_INT, R_FLOAT])
        cat_features = feature_metadata.get_features(valid_raw_types=[R_CATEGORY])
        
        # Preprocess X into the format that PECOS will accept
        preprocessed_X = []
        for i, r in X.reset_index().iterrows():

            if cat_features:
                cat_feature_str = ' '.join(f'{clean_str(c)}_{clean_str(r[c])}' for c in cat_features)
            else:
                cat_feature_str = ''

            if num_features:
                # naive discretization of numbers
                num_feature_str = ' '.join(f'{clean_str(c)}_{r[c]:.2e}' for c in num_features)
            else:
                num_feature_str = ''

            out = f'{cat_feature_str} {num_feature_str}'.lower()
            preprocessed_X.append(out)
        return np.asarray(preprocessed_X)

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             X_val: Optional[pd.DataFrame]=None,
             y_val: Optional[pd.Series]=None,
             time_limit=None,
             model_type="XRLinear",
             workdir = None,
             model_dir = None,
             **kwargs):
        """
        Fit model on X and y
        X_val and y_val are not currently used
        """
        print('Entering the `_fit` method')
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise f"model_type {model_type} not supported. model_type should be one of the following: {self.SUPPORTED_MODEL_TYPES}"
        if X_val is not None or y_val is not None:
            logger.warn("We do not utilize X_val or y_val in the current PECOS implementation")
        
        # Create directory to house model artifacts
        run_id = str(uuid.uuid4())[:10]
        if model_dir is None:
            model_dir = pathlib.Path(f'./pecos-workdir/{run_id}/model')
        else:
            model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create working directory to house input data and model output
        if workdir is None:
            workdir = pathlib.Path(f'./pecos-workdir/{run_id}/')
        else:
            workdir = pathlib.Path(workdir + f'{run_id}/')
        workdir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess training data
        X = self.preprocess(X, is_train=True)
        X_input = self._format_input_with_labels(X, y, is_train=True)

        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        # Fit model
        self.model = PecosInterface(**params)
        self.model.fit(X_input, y, time_limit, workdir, model_dir)
        print('Exiting the `_fit` method')

    def _format_input_with_labels(self, X, y, is_train=False):
        """
        Convert X and self.train_labels to the required format for PECOS.
        
        Currently only supports one label per training example

        Required text format for each line for PECOS:
        <comma-separated label indices><TAB><space-separated text string>
        Example: l_1<TAB>w_1 w_2 ... w_t
            l_1 can be one of two format:
                (1) the zero-based index for the first relevant label
                (2) double colon separated label index and label relevance
            w_t is the t-th token in the string
        """
        label_dict = {label:i for i, label in enumerate(y.unique())}
        preprocessed_X = []
        for i, r in enumerate(X):
            label = y.iloc[i] if is_train else -1
            label_id = label_dict.get(label, -1)
            out = (f'{label_id:d}\t' + r).lower()
            preprocessed_X.append(out)
        return np.asarray(preprocessed_X)

    def _set_default_params(self):
        default_params = {
                "problem_type": self.problem_type,
                "model_type": "XRLinear",
                "max_leaf_size": 100,  # The max size of the leaf nodes of hierarchical 2-means clustering
                "nr_splits": 16,  # number of splits used to construct hierarchy (a power of 2 is recommended)
                "spherical": True,  # If true, do l2-normalize cluster centers while clustering
                "kmeans_max_iter": 20,  # The max number of k-means iteration for indexing 
                'solver_type': "L2R_L2LOSS_SVC_DUAL",
                'coefficient_positive': 1.0,  # Coefficient for positive class in the loss function
                'coefficient_negative': 1.0,  # Coefficient for negative class in the loss function
                'bias': 1.0,  # Bias term
                'negative_sampling': "tfn",  # Negative sampling schemes
                'sparsity_threshold': 0.1  # Threshold to sparsify the model weights
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
