__all__ = ["FastTextModel"]


import logging
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from autogluon.utils.tabular.features.feature_types_metadata import FeatureTypesMetadata

from ....utils.loaders import load_pkl
from ...constants import BINARY, MULTICLASS
from ..abstract.abstract_model import AbstractModel
from .hyperparameters.parameters import get_param_baseline

logging.basicConfig(
    format="%(asctime)s: [%(funcName)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def try_import_fasttext():
    try:
        import fasttext

        _ = fasttext.__file__
    except Exception:
        raise ImportError('Import fasttext failed. Please run "pip install fasttext"')


class FastTextModel(AbstractModel):
    model_bin_file_name = "fasttext.bin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.problem_type not in (BINARY, MULTICLASS):
            raise ValueError(
                "FastText model only supports binary or multiclass classification"
            )

    def _set_default_params(self):
        default_params = get_param_baseline()
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _fit(self, X_train, y_train, **kwargs):
        try_import_fasttext()
        import fasttext

        if self.feature_types_metadata is None:
            feature_types_raw = defaultdict(list)
            feature_types_raw["all"] = X_train.columns.to_list()
            feature_types_special = defaultdict(list)
            feature_types_special["text_raw"] += [
                c for c in X_train.columns if not is_numeric_dtype(X_train[c])
            ]
            self.feature_types_metadata = FeatureTypesMetadata(
                feature_types_raw=feature_types_raw,
                feature_types_special=feature_types_special,
            )
        if (
            len(self.feature_types_metadata.get_features_by_type_special("text_raw"))
            == 0
        ):
            logger.error("cannot find any NLP columns ... exit")
            raise RuntimeError("FastText model cannot find any NLP features")
        logger.info(
            "NLP features %s",
            self.feature_types_metadata.get_features_by_type_special("text_raw"),
        )

        self.label_dtype = y_train.dtype
        self.label_map = {
            label: f"__label__{i}" for i, label in enumerate(y_train.unique())
        }
        self.label_inv_map = {v: k for k, v in self.label_map.items()}
        X_train = self.preprocess(X_train)
        idxs = np.random.permutation(list(range(len(X_train))))
        with tempfile.NamedTemporaryFile(mode="w+t") as f:
            logger.debug("generate training data")
            for label, text in zip(y_train.iloc[idxs], (X_train[i] for i in idxs)):
                f.write(f"{self.label_map[label]} {text}\n")
            f.flush()
            logger.debug("train FastText model")
            self.model = fasttext.train_supervised(f.name, **self.params)
            logger.debug("finish training FastText model")

    # TODO: use super().preprocess(X) once self.text_cols is altered to be self.features
    def preprocess(self, X: pd.DataFrame) -> list:
        text_col = (
            X[self.feature_types_metadata.get_features_by_type_special("text_raw")]
            .astype(str)
            .fillna(" ")
            .apply(lambda r: " __NEWCOL__ ".join(v for v in r.values), axis=1)
            .str.lower()
            .str.replace("<.*?>", " ")  # remove html tags
            # .str.replace('''(\\d[\\d,]*)(\\.\\d+)?''', ' __NUMBER__ ') # process numbers preserve dot
            .str.replace("""([\\W])""", " \\1 ")  # separate special characters
            .str.replace("\\s", " ")
            .str.replace("[ ]+", " ")
        )
        return text_col.to_list()

    def predict(self, X: pd.DataFrame, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self.preprocess(X)
        pred_labels, pred_probs = self.model.predict(X)
        y_pred = np.array(
            [self.label_inv_map[labels[0]] for labels in pred_labels],
            dtype=self.label_dtype,
        )
        return y_pred

    def predict_proba(self, X: pd.DataFrame, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self.preprocess(X)

        pred_labels, pred_probs = self.model.predict(X, k=len(self.model.labels))

        recs = []
        for labels, probs in zip(pred_labels, pred_probs):
            recs.append(
                dict(zip((self.label_inv_map[label] for label in labels), probs))
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

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        if directory is None:
            directory = self.path

        # pickle model parts
        model = self.model
        self.model = None
        self.model_bin_available = model is not None
        modelobj_filepath = super().save(
            file_prefix=file_prefix,
            directory=directory,
            return_filename=True,
            verbose=verbose,
        )

        # save fasttext model: fasttext model cannot be pickled; saved it seperately
        # TODO: s3 support
        if self.model_bin_available:
            fasttext_model_file_name = (
                directory + file_prefix + self.model_bin_file_name
            )
            model.save_model(fasttext_model_file_name)
        self.model = model
        if return_filename:
            return modelobj_filepath

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        try_import_fasttext()
        import fasttext

        path = path + file_prefix

        obj: FastTextModel = load_pkl.load(
            path=path + cls.model_file_name, verbose=verbose
        )
        if reset_paths:
            obj.set_contexts(path)

        # load binary fasttext model
        if obj.model_bin_available:
            fasttext_model_file_name = path + cls.model_bin_file_name
            obj.model = fasttext.load_model(fasttext_model_file_name)

        return obj
