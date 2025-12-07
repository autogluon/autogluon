import numpy as np

from typing import List
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd

from .abstract import AbstractFeatureGenerator


class OOFTargetEncodingFeatureGenerator(AbstractFeatureGenerator):
    """
    KFold out-of-fold target encoding (regression / binary / multiclass)
    Parameters
    ----------
    target_type : str
        The type of the target variable ('regression', 'binary', or 'multiclass').
    n_splits : int, default=5
        Number of folds for KFold or StratifiedKFold.
    alpha : float, default=10.0
        Smoothing parameter for target encoding.
    random_state : int, default=42
        Random state for reproducibility.
    keep_original : bool, default=False
        Whether to keep the original features.
    **kwargs
        Additional keyword arguments.
    Returns
    -------
    self : OOFTargetEncoderFeatureGenerator
        Fitted OOFTargetEncoderFeatureGenerator instance.
    Notes
    -----
    The target encoding is performed using K-Fold cross-validation to prevent data leakage.
    For regression and binary classification, the target mean is used for encoding.
    For multiclass classification, separate encodings are created for each class.
    During transformation on unseen data, the full training statistics are used to encode new data.
      - fit(...) computes + stores OOF TRAIN encodings and full-train stats
      - transform(..., is_train=True) returns stored OOF TRAIN encodings
      - transform(..., is_train=False) encodes new data using full stats
    """
    # TODO: Change the implementation to compute OOF encodings only during fit_transform and to never just return stored objects during transform
    def __init__(self, 
                 target_type:str,
                 keep_original:bool=False,
                 n_splits:int=5,
                 alpha:float=10.0,
                 random_state:int=42,
                 **kwargs):
        super().__init__(**kwargs)
        assert target_type in {"regression","binary","multiclass"}
        self.target_type = target_type
        self.keep_original = keep_original
        self.n_splits = n_splits
        self.alpha = alpha
        self.random_state = random_state

    def estimate_no_of_new_features(self, X: pd.DataFrame, num_classes: int, **kwargs) -> int:
        X_cat = X.select_dtypes(include=['object', 'category'])
        num_cat_cols = X_cat.shape[1]
        if self.target_type == "multiclass":
            return num_classes * num_cat_cols, X_cat.columns.tolist()
        else:
            return num_cat_cols, X_cat.columns.tolist()

    # -----------------------------------------------------------
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        original_index = X.index
        self.cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.passthrough_cols_ = [col for col in X.columns if col not in self.cols_]
        X = X[self.cols_].reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        X = X.astype('object')

        if len(self.cols_)==0:
            return self

        # convert y to matrix by target_type
        if self.target_type == "regression":
            kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
            Y = y.values[:,None]

        elif self.target_type == "binary":
            kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            # require y {0,1} or {labelA,labelB}
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            assert len(classes) == 2, "binary target_type but >2 classes"
            Y = (y.values == classes[-1]).astype(float)[:,None]
            self.classes_ = classes

        else: # multiclass
            kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            self.classes_ = classes
            K = len(classes)
            Y = np.zeros((len(y),K))
            for i,c in enumerate(classes):
                Y[:,i] = (y.values==c).astype(float)

        self.n_targets = Y.shape[1]

        store = []

        # full train stats (for test/inference)
        full_stats = {}

        for col in self.cols_:
            oof = np.zeros((len(X), Y.shape[1]))
            col_values = X[col]

            # Build once per column
            df_full = pd.DataFrame({"cat": col_values})
            for j in range(Y.shape[1]):
                df_full[f"y{j}"] = Y[:, j]

            for tr,val in kf.split(X, y):
                g = df_full.iloc[tr].groupby("cat", observed=True).agg(["count","mean"])
                for j in range(Y.shape[1]):
                    m = g[("y"+str(j),"mean")]
                    c = g[("y"+str(j),"count")]
                    enc = (m*c + self.alpha*m.mean())/(c + self.alpha)
                    oof[val,j] = col_values.iloc[val].map(enc).fillna(m.mean())

            # names
            if Y.shape[1]==1:
                names=[f"{col}__te"]
            else:
                names=[f"{col}__te_class{j}" for j in range(Y.shape[1])]
            store.append(pd.DataFrame(oof, columns=names, index=original_index))

            g = df_full.groupby("cat", observed=True).agg(["count","mean"])
            full_stats[col] = g
        self.full_stats_ = full_stats

        self.train_encoded_ = pd.concat(store, axis=1)

        return self

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self._fit(X, y)

        assert hasattr(self, "train_encoded_"), "fit() not called"
        if self.keep_original:
            X_out = pd.concat([X, self.train_encoded_], axis=1)
        else:
            X_out = pd.concat([X[self.passthrough_cols_], self.train_encoded_], axis=1)
        self.train_encoded_ = None
        return X_out, dict()

    # -----------------------------------------------------------
    def _transform(self, X_in, **kwargs):
        if len(self.cols_) == 0:
            return X_in.copy()

        # -------------------------------
        # Inference-time: use full_stats_
        # -------------------------------
        # Work directly on X_in; only cast the categorical columns we touch
        X = pd.DataFrame(X_in, copy=False)
        out_frames: List[pd.DataFrame] = []

        n_targets = self.n_targets
        alpha = self.alpha

        for col in self.cols_:
            col_series = X[col].astype("object")
            g = self.full_stats_[col]  # MultiIndex columns: (y0, count/mean), (y1, count/mean), ...

            if n_targets == 1:
                # -------- regression / binary: single column
                means = g.xs("mean", axis=1, level=1).iloc[:, 0]  # Series indexed by category
                counts = g.xs("count", axis=1, level=1).iloc[:, 0]  # Series indexed by category

                global_mean = means.mean()
                enc = (means * counts + alpha * global_mean) / (counts + alpha)  # Series

                encoded = col_series.map(enc).fillna(global_mean)
                df_col = encoded.to_frame(f"{col}__te")
                out_frames.append(df_col.astype(float))

            else:
                # -------- multiclass: vectorized over classes
                means = g.xs("mean", axis=1, level=1)  # DataFrame [cat x K], columns: y0, y1, ...
                counts = g.xs("count", axis=1, level=1)  # DataFrame [cat x K], columns: y0, y1, ...

                global_mean = means.mean(axis=0)  # Series with index matching means.columns (y0, y1, ...)

                enc = (means * counts + alpha * global_mean) / (counts + alpha)  # DataFrame [cat x K]

                # Join once instead of mapping per class
                encoded = col_series.to_frame("cat").join(enc, on="cat")
                encoded = encoded.drop(columns=["cat"])

                # Fill unseen categories with per-class global means
                encoded = encoded.fillna(global_mean)

                # Rename columns to expected names
                encoded.columns = [f"{col}__te_class{j}" for j in range(n_targets)]
                out_frames.append(encoded.astype(float))

        if self.keep_original:
            return pd.concat([X_in] + out_frames, axis=1)
        else:
            return pd.concat([X_in[self.passthrough_cols_]] + out_frames, axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()