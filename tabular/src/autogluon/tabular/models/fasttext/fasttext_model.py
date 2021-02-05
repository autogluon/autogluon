__all__ = ["FastTextModel"]

import contextlib
import gc
import logging
import os
import psutil
import tempfile

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.features.types import S_TEXT

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract.model_trial import skip_hpo
from .hyperparameters.parameters import get_param_baseline

logger = logging.getLogger(__name__)


def try_import_fasttext():
    try:
        import fasttext

        _ = fasttext.__file__
    except Exception:
        raise ImportError('Import fasttext failed. Please run "pip install fasttext"')


class FastTextModel(AbstractModel):
    model_bin_file_name = "fasttext.ftz"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_bin_available = False

    def _set_default_params(self):
        default_params = get_param_baseline()
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Investigate allowing categorical features as well
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                required_special_types=[S_TEXT],
            )
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {'valid_stacker': False, 'problem_types': [BINARY, MULTICLASS]}
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    def _fit(self, X_train, y_train, **kwargs):
        if self.problem_type not in (BINARY, MULTICLASS):
            raise ValueError(
                "FastText model only supports binary or multiclass classification"
            )

        try_import_fasttext()
        import fasttext

        params = self.params.copy()
        quantize_model = params.pop('quantize_model', True)

        verbosity = kwargs.get('verbosity', 2)
        if 'verbose' not in params:
            if verbosity <= 2:
                params['verbose'] = 0
            elif verbosity == 3:
                params['verbose'] = 1
            else:
                params['verbose'] = 2

        X_train = self.preprocess(X_train)
        logger.debug("NLP features %s", self.features)

        self._label_dtype = y_train.dtype
        self._label_map = {label: f"__label__{i}" for i, label in enumerate(y_train.unique())}
        self._label_inv_map = {v: k for k, v in self._label_map.items()}
        np.random.seed(0)
        idxs = np.random.permutation(list(range(len(X_train))))
        with tempfile.NamedTemporaryFile(mode="w+t") as f:
            logger.debug("generate training data")
            for label, text in zip(y_train.iloc[idxs], (X_train[i] for i in idxs)):
                f.write(f"{self._label_map[label]} {text}\n")
            f.flush()
            mem_start = psutil.Process().memory_info().rss
            logger.debug("train FastText model")
            self.model = fasttext.train_supervised(f.name, **params)
            if quantize_model:
                self.model.quantize(input=f.name, retrain=True)
            gc.collect()
            mem_curr = psutil.Process().memory_info().rss 
            self._model_size_estimate = max(mem_curr - mem_start, 100000000 if quantize_model else 800000000)
            logger.debug("finish training FastText model")

    # TODO: move logic to self._preprocess_nonadaptive()
    # TODO: text features: alternate text preprocessing steps
    # TODO: categorical features: special encoding:  <feature name>_<feature value>
    def _preprocess(self, X: pd.DataFrame, **kwargs) -> list:
        text_col = (
            X[self.features]
            .astype(str)
            .fillna(" ")
            .apply(lambda r: " ".join(v for v in r.values), axis=1)
            .str.lower()
            .str.replace("<.*?>", " ")  # remove html tags
            # .str.replace('''(\\d[\\d,]*)(\\.\\d+)?''', ' __NUMBER__ ') # process numbers preserve dot
            .str.replace("""([\\W])""", " \\1 ")  # separate special characters
            .str.replace("\\s", " ")
            .str.replace("[ ]+", " ")
        )
        return text_col.to_list()

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        pred_labels, pred_probs = self.model.predict(X)
        y_pred = np.array(
            [self._label_inv_map[labels[0]] for labels in pred_labels],
            dtype=self._label_dtype,
        )
        return y_pred

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        pred_labels, pred_probs = self.model.predict(X, k=len(self.model.labels))

        recs = []
        for labels, probs in zip(pred_labels, pred_probs):
            recs.append(
                dict(zip((self._label_inv_map[label] for label in labels), probs))
            )

        y_pred_proba: np.ndarray = pd.DataFrame(recs).sort_index(axis=1).values

        if self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]

    def save(self, path: str = None, verbose=True) -> str:
        # pickle model parts
        model = self.model
        self.model = None
        self._model_bin_available = model is not None
        path_final = super().save(path=path, verbose=verbose)

        # save fasttext model: fasttext model cannot be pickled; saved it seperately
        # TODO: s3 support
        if self._model_bin_available:
            fasttext_model_file_name = path_final + self.model_bin_file_name
            model.save_model(fasttext_model_file_name)
        self.model = model
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        try_import_fasttext()
        import fasttext

        obj: FastTextModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        # load binary fasttext model
        if obj._model_bin_available:
            fasttext_model_file_name = obj.path + cls.model_bin_file_name
            # TODO: hack to subpress a deprecation warning from fasttext 
            # remove it once offcial fasttext is updated beyond 0.9.2 
            # https://github.com/facebookresearch/fastText/issues/1067
            with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
                obj.model = fasttext.load_model(fasttext_model_file_name)

        return obj

    def get_memory_size(self) -> int:
        return self._model_size_estimate

    # TODO: Add HPO
    def _hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)
