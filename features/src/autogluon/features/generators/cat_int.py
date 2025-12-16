from collections import defaultdict
from itertools import combinations
from math import comb
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd

from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_FLOAT,
    R_INT,
    R_OBJECT,
    S_BOOL,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
)
from autogluon.features.generators.category import CategoryFeatureGenerator

from .abstract import AbstractFeatureGenerator
from .frequency import FrequencyFeatureGenerator


class CategoricalInteractionFeatureGenerator(AbstractFeatureGenerator):
    """
    Generate new categorical features by combining existing categorical features.
    Parameters
    ----------
    target_type : str
        The type of the target variable ('regression' or 'classification').
    max_order : int, default=2
        The maximum order of interactions to generate.
    max_new_feats : int, default=100
        Maximum number of new features to generate.
    candidate_cols : list of str or None, default=None
        List of candidate columns to consider for interaction generation. If None, all categorical columns are used.
    add_freq : bool, default=False
        Whether to add frequency encoding for the new features.
    only_freq : bool, default=False
        Whether to only keep frequency encoded features.
    min_cardinality : int, default=6
        Minimum cardinality of categorical features to be considered for interaction generation.
    min_count : int, default=2
        Minimum count of the most frequent category in a new feature to be retained.
    fillna : int, default=0
        Value to fill NaNs in frequency encoding.
    log : bool, default=False
        Whether to apply log transformation in frequency encoding.
    random_state : int, default=42
        Random state for reproducibility.
    **kwargs
        Additional keyword arguments.
    Returns
    -------
    self : CategoricalInteractionFeatureGenerator
        Fitted CategoricalInteractionFeatureGenerator instance.
    """

    def __init__(
        self,
        target_type: Literal["regression", "multiclass", "binary"],
        max_order: int = 3,
        max_new_feats: int = 100,
        candidate_cols: List[str] = None,
        add_freq: bool = False,
        only_freq: bool = False,
        min_cardinality: int = 2,
        min_count: int = 2,
        fillna: int = 0,
        log: bool = False,
        random_state: int = 42,
        inference_mode: Literal["string", "category"] = "category",
        make_categoricals: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.__name__ = "CatIntAdder"
        self.max_order = max_order
        self.candidate_cols = candidate_cols
        self.add_freq = add_freq
        self.only_freq = only_freq
        self.min_cardinality = min_cardinality
        self.min_count = min_count
        self.fillna = fillna
        self.log = log
        self.max_new_feats = max_new_feats
        self.inference_mode = inference_mode

        self.rng = np.random.default_rng(random_state)

        self.new_dtypes = {}

        self.make_categoricals = make_categoricals
        self.fitted_ = False
        self.cat_sizes_ = {}  # base categorical cardinalities
        self.reverse_mapping_ = {}  # interaction_col → sorted unique int combos

    # FIXME: Implement passthrough a bit better -> If feature in output that is also in input, drop the input feature?
    # TODO: This is the correct one, but due to the `passthrough` logic, need to keep all.
    #  Need to refactor passthrough logic to be more nuanced in order to enable this
    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
            # required_raw_special_pairs=[
            #     (R_BOOL, None),
            #     (R_OBJECT, None),
            #     (R_CATEGORY, None),
            #     # (R_INT, S_BOOL),
            #     # (R_FLOAT, S_BOOL),
            # ],
        )

    # @staticmethod
    # def get_default_infer_features_in_args() -> dict:
    #     return dict()

    def estimate_new_dtypes(self, n_numeric, n_categorical, n_binary, **kwargs) -> int:
        num_new_feats = 0
        for order in range(2, self.max_order + 1):
            if n_categorical < order:
                continue
            num_new_feats += comb(n_categorical, order)

            if num_new_feats >= self.max_new_feats:
                num_new_feats = self.max_new_feats
                break

        if self.only_freq:
            return n_numeric + num_new_feats, n_categorical, n_binary
        elif self.add_freq:
            return n_numeric + num_new_feats, n_categorical + num_new_feats, n_binary
        else:
            return n_numeric, n_categorical + num_new_feats, n_binary

    def estimate_no_of_new_features(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimate the number of new features that will be generated.
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        Returns
        -------
        int
            Estimated number of new features.
        Notes
        -----
        This is a rough estimate based on the number of categorical features
        that pass the cardinality filter and the specified maximum order.
        Currently, doesn't consider the min_count filter. Therefore, the actual
        number of generated features may be lower than this estimate.
        """
        X_cat = X.select_dtypes(include=["object", "category"])
        pass_cardinality_filter = X_cat.nunique().values >= self.min_cardinality
        num_base_feats = np.sum(pass_cardinality_filter)
        affected_features = X_cat.columns[pass_cardinality_filter].tolist()

        num_new_feats = 0
        for order in range(2, self.max_order + 1):
            if num_base_feats < order:
                continue
            num_new_feats += comb(num_base_feats, order)

        return np.min([self.max_new_feats, num_new_feats]), affected_features

    def group_feat_combs_by_order(self, new_col_set):
        feat_combs = [s.split("_&_") for s in new_col_set]
        by_order = defaultdict(list)
        for cols in feat_combs:
            by_order[len(cols)].append(cols)
        return dict(by_order)

    def get_interaction_names_dict(self, X: pd.DataFrame, max_order: int = 2, max_feats: int = 100):
        self.new_col_set = []
        for o in range(2, max_order + 1):
            feat_combs_use = list(combinations(np.unique(X.columns), o))
            add_cols = ["_&_".join(f) for f in feat_combs_use]
            max_feats_order = max_feats - len(self.new_col_set)
            if len(add_cols) > max_feats_order:
                add_cols = self.rng.choice(add_cols, max_feats_order, replace=False).tolist()
            self.new_col_set.extend(add_cols)
            if len(self.new_col_set) >= max_feats:
                self.new_col_set = self.new_col_set[:max_feats]
                break
        self.new_cols_by_order = self.group_feat_combs_by_order(self.new_col_set)

        self.used_cols = []
        for col in self.new_col_set:
            self.used_cols.extend(col.split("_&_"))
        self.used_cols = list(set(self.used_cols))

    def combine_predefined(self, X_in: pd.DataFrame, comb_lst: List[str], fit_mode=True, **kwargs) -> pd.DataFrame:
        """Generate interaction features based on predefined combinations.
        Parameters
        ----------
        X_in : pd.DataFrame
            Input categorical features DataFrame.
        comb_lst : List[str]
            List of predefined combinations as strings. Can be higher order combinations, but one call to this function should only handle one order at a time.
        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated interaction features.
        """
        X = X_in.copy()
        X = X_in.astype("U")
        feat_combs_use = [i.split("_&_") for i in comb_lst]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        X_out = pd.DataFrame(features, columns=comb_lst, index=X.index)

        return X_out

    # ----------------------------------------------------------------------
    # INTERNAL: build interactions (faster version)
    # ----------------------------------------------------------------------
    def _build_interactions(self, X, fit_mode):
        # Precompute base codes (faster than repeatedly calling .cat.codes)
        base_codes = {col: X[col].cat.codes.to_numpy(np.int32) for col in X.columns}

        outputs = {}  # column_name → numpy array

        # --------------------------
        # ORDER 2 interactions
        # --------------------------
        if 2 in self.new_cols_by_order:
            for cols in self.new_cols_by_order[2]:
                name = "_&_".join(cols)
                col1, col2 = cols
                k = self.cat_sizes_[col2]

                combined = base_codes[col1] * k + base_codes[col2]
                outputs[name] = self._fit_or_map(name, combined, fit_mode)

                # Expose newly created interaction codes for higher-order use
                base_codes[name] = outputs[name]

        # --------------------------
        # ORDER >= 3 interactions
        # --------------------------
        for order in sorted(k for k in self.new_cols_by_order if k >= 3):
            for cols in self.new_cols_by_order[order]:
                name = "_&_".join(cols)

                prefix = "_&_".join(cols[:-1])
                last = cols[-1]

                k = self.cat_sizes_[last]
                combined = base_codes[prefix] * k + base_codes[last]

                outputs[name] = self._fit_or_map(name, combined, fit_mode)
                base_codes[name] = outputs[name]

        # ------------------------------------------------------------
        # Build final DataFrame in one fast pass
        # ------------------------------------------------------------
        if not outputs:
            return pd.DataFrame(index=X.index)

        df_out = pd.DataFrame(outputs, index=X.index)

        # Convert to categoricals with consistent ordering
        if self.make_categoricals:
            for col in df_out.columns:
                uniques = self.reverse_mapping_[col]
                df_out[col] = pd.Categorical.from_codes(
                    df_out[col],
                    categories=range(len(uniques)),  # ensures 0..n-1
                    ordered=False,
                )

        return df_out

    # ----------------------------------------------------------------------
    # INTERNAL: fit or transform mapping (FAST version)
    # ----------------------------------------------------------------------
    def _fit_or_map(self, name, combined, fit_mode):
        """
        combined: int64 array of encoded interaction IDs
        """

        if fit_mode:
            valid = combined[combined >= 0]
            uniques = np.unique(valid)
            self.reverse_mapping_[name] = uniques
            return self._vectorized_map(combined, uniques)
        else:
            uniques = self.reverse_mapping_[name]
            return self._vectorized_map(combined, uniques)

    # ----------------------------------------------------------------------
    # INTERNAL: vectorized mapping using np.searchsorted (FAST)
    # ----------------------------------------------------------------------
    @staticmethod
    def _vectorized_map(combined, uniques):
        """
        Map combined IDs to 0..(n-1) using vectorized searchsorted.
        Unseen combos → -1.
        """
        mapped = np.full(combined.shape, -1, dtype=np.int64)

        valid_mask = combined >= 0
        vals = combined[valid_mask]

        # vectorized position lookup
        idx = np.searchsorted(uniques, vals)

        # Step 1: out-of-range positions (unseen)
        oob = idx == len(uniques)

        # Step 2: in-range but not equal → unseen mismatch
        in_range = ~oob
        mismatch = np.zeros_like(oob)
        mismatch[in_range] = uniques[idx[in_range]] != vals[in_range]

        miss = oob | mismatch

        # Assign -1 to misses, otherwise idx
        out = np.full(vals.shape, -1, dtype=np.int64)
        out[~miss] = idx[~miss]

        mapped[valid_mask] = out
        return mapped.astype(np.int32)

    def frequency_encode_new_features(self, X: pd.DataFrame, y: pd.Series = None):
        if self.add_freq or self.only_freq:
            if not self.fitted_:
                # TODO: Unclear whether there is a more efficient way to do this
                candidate_cols = FrequencyFeatureGenerator.filter_candidates_by_distinctiveness(X[self.new_col_set])
                if len(candidate_cols) > 0:
                    keep_original = not self.only_freq
                    self.cat_freq = FrequencyFeatureGenerator(
                        candidate_cols=candidate_cols, fillna=self.fillna, log=self.log, keep_original=keep_original
                    )
                    return self.cat_freq.fit_transform(X[candidate_cols], y)
                else:
                    return X
            else:
                return self.cat_freq.transform(X)
        else:
            return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if self.candidate_cols is None:
            self.candidate_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            self.candidate_cols = [
                i for i in self.candidate_cols if X[i].nunique() >= self.min_cardinality
            ]  # TODO: Make this a parameter

        # FIXME: why `self.max_order` and not 2?
        if len(self.candidate_cols) < self.max_order:
            self.new_col_set = []
            return self

        X = X[self.candidate_cols].copy()
        # Ensure categorical only when needed
        for col in self.candidate_cols:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")

        self.get_interaction_names_dict(X[self.candidate_cols], max_order=self.max_order, max_feats=self.max_new_feats)

        # Cache category sizes (+1 for missing index -1)
        self.cat_sizes_ = {col: len(X[col].cat.categories) + 1 for col in self.used_cols}

        self.fitted_ = True

        return self

    # FIXME: Filter if too unique?
    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X, fit_mode=True)

        # features_out = list(X_out.columns)

        # type_group_map_special = {R_CATEGORY: features_out} # TODO: Find out whether that is needed
        return X_out, dict()

    def _transform(self, X: pd.DataFrame, y: pd.Series = None, fit_mode: bool = False, **kwargs) -> pd.DataFrame:
        if len(self.new_col_set) == 0:
            return pd.DataFrame(index=X.index)

        if self.inference_mode == "string":
            X_out = pd.DataFrame(index=X.index)
            for degree in range(2, self.max_order + 1):
                col_set_use = [col for col in self.new_col_set if col.count("_&_") + 1 == degree]
                if len(col_set_use) > 0:
                    X_degree = self.combine_predefined(X[self.used_cols], col_set_use, fit_mode=fit_mode)
                    X_out = pd.concat([X_out, X_degree], axis=1)
            if fit_mode:
                self.cat_transformer = CategoryFeatureGenerator(minimum_cat_count=1)
                X_out = self.cat_transformer.fit_transform(X_out)
            else:
                X_out = self.cat_transformer.transform(X_out)

        else:  # new inference mode
            for col in self.used_cols:
                if not isinstance(X[col].dtype, pd.CategoricalDtype):
                    X[col] = X[col].astype("category")
            X_out = self._build_interactions(X[self.used_cols], fit_mode=fit_mode)

        X_out = self.frequency_encode_new_features(X_out)

        return X_out
