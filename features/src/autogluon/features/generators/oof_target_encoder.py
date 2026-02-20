import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from autogluon.common.utils.cv_splitter import CVSplitter

from .abstract import AbstractFeatureGenerator


class OOFTargetEncodingFeatureGenerator(AbstractFeatureGenerator):
    """
    KFold out-of-fold target encoding (regression / binary / multiclass)
    Parameters
    ----------
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

    def __init__(
        self,
        keep_original: bool = False,
        n_splits: int = 5,
        alpha: float = 10.0,
        # TODO: Consider adding max_classes to select only the most frequent classes for multi-class to avoid feature explosion for many-class problems
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.target_type in {"regression", "binary", "multiclass", "quantile"}
        if self.target_type == "quantile":
            self.target_type = "regression"  # FIXME: this is a hack
        self.keep_original = keep_original
        self.n_splits = n_splits
        self.alpha = alpha
        self.random_state = random_state

    def estimate_new_dtypes(self, n_numeric, n_categorical, n_binary, num_classes=None, **kwargs) -> int:
        num_new_feats = n_categorical

        if self.target_type == "multiclass":
            num_new_feats *= num_classes
        if self.keep_original:
            return n_numeric + num_new_feats, n_categorical, n_binary
        else:
            return n_numeric + num_new_feats, 0, n_binary

    def estimate_no_of_new_features(self, X: pd.DataFrame, num_classes: int, **kwargs) -> int:
        X_cat = X.select_dtypes(include=["object", "category"])
        num_cat_cols = X_cat.shape[1]
        if self.target_type == "multiclass":
            return num_classes * num_cat_cols, X_cat.columns.tolist()
        else:
            return num_cat_cols, X_cat.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if y is None:
            raise AssertionError(f"y must be present during fit")
        original_index = X.index

        # Identify categorical vs passthrough cols
        self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.passthrough_cols_ = [col for col in X.columns if col not in self.cols_]

        # Early exit: nothing to encode
        if len(self.cols_) == 0:
            self.encodings_ = {}
            self.train_encoded_ = pd.DataFrame(index=original_index)
            return self

        # Work only on the categorical part (for encoding), keep object dtype
        X_cat = X[self.cols_].reset_index(drop=True).astype("object")
        y = pd.Series(y).reset_index(drop=True)

        n = len(X_cat)

        if self.target_type in ["binary", "multiclass"]:
            splitter_cls = StratifiedKFold
            stratify = True
        else:
            splitter_cls = KFold
            stratify = False

        kf = CVSplitter(
            splitter_cls=splitter_cls,
            n_splits=self.n_splits,
            random_state=self.random_state,
            stratify=stratify,
            shuffle=True,
            # bin=True if self.target_type == 'regression' else False,
            # n_bins=50 if self.target_type == 'regression' else None,
        )

        # ------------------------
        # Build target matrix Y
        # ------------------------
        if self.target_type == "regression":
            Y = y.to_numpy().reshape(-1, 1).astype(float)

        elif self.target_type == "binary":
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            assert len(classes) == 2, "binary target_type but >2 classes"
            self.classes_ = classes
            Y = (y.to_numpy() == classes[-1]).astype(float).reshape(-1, 1)

        else:  # multiclass
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            self.classes_ = classes
            arr = y.to_numpy()
            Y = (arr[:, None] == classes[None, :]).astype(float)  # [n_samples, n_classes]

        self.n_targets = Y.shape[1]

        # Precompute splits once
        kf_splits = list(kf.split(np.zeros(n), y))

        alpha = self.alpha

        store = []
        self.encodings_ = {}

        # =====================================================
        # Per-column encoding using factorize + np.bincount
        # =====================================================
        for col in self.cols_:
            col_values = X_cat[col]

            # Factorize categories once; sorted to mimic groupby index order
            codes, uniques = pd.factorize(col_values, sort=True)
            # codes == -1 corresponds to NaN / missing category
            mask_valid = codes >= 0
            codes_valid = codes[mask_valid]
            Y_valid = Y[mask_valid]
            n_cat = len(uniques)

            # ---------------------------------------
            # Global (all-data) sums & counts
            # ---------------------------------------
            count_all = np.bincount(codes_valid, minlength=n_cat).astype(float)  # (n_cat,)

            sum_all = np.vstack(
                [np.bincount(codes_valid, weights=Y_valid[:, j], minlength=n_cat) for j in range(self.n_targets)]
            ).T  # (n_cat, n_targets)

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_all = sum_all / count_all[:, None]  # (n_cat, n_targets)

            # global_mean as in original _transform:
            # mean of per-category means
            global_mean = np.nanmean(mean_all, axis=0)  # (n_targets,)

            # Smoothed per-category encodings used at inference
            denom_all = count_all[:, None] + alpha
            num_all = mean_all * count_all[:, None] + alpha * global_mean[None, :]

            with np.errstate(divide="ignore", invalid="ignore"):
                enc_all = num_all / denom_all  # (n_cat, n_targets)

            # Names for encoded features
            if self.n_targets == 1:
                names_out = [f"{col}__te"]
            else:
                names_out = [f"{col}__te_class{j}" for j in range(self.n_targets)]

            # enc_df = pd.DataFrame(enc_all, index=uniques, columns=names_out)
            # global_mean_series = pd.Series(global_mean, index=names_out)
            #
            # # Store lightweight stats for inference
            # self.encodings_[col] = dict(
            #     enc=enc_df,
            #     global_mean=global_mean_series,
            # )

            # Store lightweight, numeric-only info for fast transform
            self.encodings_[col] = dict(
                categories=uniques.to_numpy(copy=False),  # np.array of categories, length n_cat
                enc_matrix=enc_all.astype(float, copy=False),  # shape (n_cat, n_targets)
                global_mean=global_mean.astype(float, copy=False),  # shape (n_targets,)
                names=names_out,  # list of output column names for this feature
            )

            # ------------------------
            # OOF encodings (fast)
            # ------------------------
            oof = np.zeros((n, self.n_targets), dtype=float)

            for tr_idx, val_idx in kf_splits:
                val_mask = np.zeros(n, dtype=bool)
                val_mask[val_idx] = True
                tr_mask_valid = mask_valid & ~val_mask

                codes_tr = codes[tr_mask_valid]
                Y_tr = Y[tr_mask_valid]

                if codes_tr.size == 0:
                    # Degenerate case: no training rows for this fold
                    oof[val_idx, :] = global_mean[None, :]
                    continue

                # Per-category sums & counts on the *training* portion
                count_tr = np.bincount(codes_tr, minlength=n_cat).astype(float)  # (n_cat,)
                sum_tr = np.vstack(
                    [np.bincount(codes_tr, weights=Y_tr[:, j], minlength=n_cat) for j in range(self.n_targets)]
                ).T  # (n_cat, n_targets)

                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_tr = sum_tr / count_tr[:, None]  # (n_cat, n_targets)

                valid_cats = count_tr > 0

                # m_mean: mean of per-category means over categories that appear in training
                m_mean = np.where(valid_cats[:, None], mean_tr, np.nan)
                m_mean = np.nanmean(m_mean, axis=0)  # (n_targets,)

                denom = count_tr[:, None] + alpha
                num = mean_tr * count_tr[:, None] + alpha * m_mean[None, :]

                with np.errstate(divide="ignore", invalid="ignore"):
                    enc_tr = num / denom  # (n_cat, n_targets)

                enc_tr[~valid_cats, :] = m_mean

                # Assign encodings to OOF for this fold
                enc_val = np.zeros((len(val_idx), self.n_targets), dtype=float)
                enc_val[:] = m_mean[None, :]

                val_codes = codes[val_idx]
                non_nan_mask = val_codes >= 0
                if np.any(non_nan_mask):
                    enc_val[non_nan_mask, :] = enc_tr[val_codes[non_nan_mask]]

                oof[val_idx, :] = enc_val

            # Build DataFrame for this column, aligned to original index
            if self.n_targets == 1:
                df_oof = pd.DataFrame(oof, columns=names_out, index=original_index)
            else:
                df_oof = pd.DataFrame(oof, columns=names_out, index=original_index)
            store.append(df_oof)

        self.train_encoded_ = pd.concat(store, axis=1)

        return self

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self._fit(X, y)

        if len(self.cols_) == 0:
            return X.copy(), dict()
        if self.keep_original:
            X_out = pd.concat([X, self.train_encoded_], axis=1)
        else:
            X_out = pd.concat([X[self.passthrough_cols_], self.train_encoded_], axis=1)
        self.train_encoded_ = None
        return X_out, dict()

    def _transform(self, X, **kwargs):
        if len(self.cols_) == 0:
            return X.copy()

        n_rows = len(X)

        # Precompute how many encoded features we will create
        total_new_feats = 0
        for col in self.cols_:
            total_new_feats += len(self.encodings_[col]["names"])

        # Single big output array for all encoded columns
        encoded_all = np.empty((n_rows, total_new_feats), dtype=float)
        encoded_colnames: list[str] = []

        offset = 0
        for col in self.cols_:
            info = self.encodings_[col]
            categories = info["categories"]  # np.array, shape (n_cat,)
            enc_matrix = info["enc_matrix"]  # np.array, shape (n_cat, n_targets)
            global_mean = info["global_mean"]  # np.array, shape (n_targets,)
            names = info["names"]  # list[str], length n_targets

            n_targets = enc_matrix.shape[1]

            # Extract raw values once, no copy if possible
            col_vals = X[col].to_numpy(dtype=object, copy=False)

            # Use a Categorical with fixed categories from fit to get stable codes
            cat = pd.Categorical(col_vals, categories=categories, ordered=False)
            codes = cat.codes  # int array; -1 for NaN/unseen categories

            # Prepare block for this column
            block = np.empty((n_rows, n_targets), dtype=float)
            # Start with global_mean everywhere (handles NaN/unseen)
            block[:] = global_mean[None, :]

            # For valid codes, index into enc_matrix
            valid_mask = codes >= 0
            if np.any(valid_mask):
                block[valid_mask, :] = enc_matrix[codes[valid_mask]]

            # Place block into the big encoded_all array
            encoded_all[:, offset : offset + n_targets] = block
            encoded_colnames.extend(names)
            offset += n_targets

        # Wrap all encoded columns into a single DataFrame
        encoded_df = pd.DataFrame(encoded_all, index=X.index, columns=encoded_colnames)

        if self.keep_original:
            # Preserve original columns + new encodings
            return pd.concat([X, encoded_df], axis=1)
        else:
            # Only passthrough + new encodings
            return pd.concat([X[self.passthrough_cols_], encoded_df], axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
