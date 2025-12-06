import numpy as np
import pandas as pd
from .frequency import FrequencyFeatureGenerator
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
from autogluon.features.generators.category import CategoryFeatureGenerator
from itertools import combinations

from typing import List, Dict, Any, Literal, Tuple

from math import comb

from .abstract import AbstractFeatureGenerator
from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_OBJECT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_AS_CATEGORY,
)

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
    def __init__(self, 
                 target_type: Literal['regression', 'multiclass', 'binary'],
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
                 **kwargs
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

        self.rng = np.random.default_rng(random_state)

        self.new_dtypes = {}

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
        X_cat = X.select_dtypes(include=['object', 'category'])
        pass_cardinality_filter = X_cat.nunique().values>=self.min_cardinality
        num_base_feats = np.sum(pass_cardinality_filter)
        affected_features = X_cat.columns[pass_cardinality_filter].tolist()


        num_new_feats = 0
        for order in range(2, self.max_order+1):
            if num_base_feats < order:
                continue
            num_new_feats += comb(num_base_feats, order)

        return np.min([self.max_new_feats, num_new_feats]), affected_features

    def combine(self, X_in: pd.DataFrame, order: int = 2, max_feats: int = 100, seed: int = 42, **kwargs) -> pd.DataFrame:
        """ Generate interaction features of a specified order by combining categorical features.
        Parameters
        ----------
        X_in : pd.DataFrame
            Input categorical features DataFrame.
        order : int, default=2
            The order of interactions to generate.
        max_feats : int, default=100
            Maximum number of interaction features to generate.
        seed : int, default=42
            Random seed for reproducibility.
        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated interaction features.
        """
        X = X_in.copy()
        X = X.astype('U')
        feat_combs_use = list(combinations(np.unique(X.columns), order))
        feat_combs_use_arr = np.array(feat_combs_use)

        if len(feat_combs_use_arr) > max_feats:
            feat_combs_use_arr = feat_combs_use_arr[self.rng.choice(len(feat_combs_use_arr), max_feats, replace=False)].T
        else:
            feat_combs_use_arr = feat_combs_use_arr.T

        new_names = ["_&_".join([str(i) for i in sorted(f_use)]) for f_use in feat_combs_use_arr.T]

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=new_names, index=X.index)

    def combine_predefined(self, X_in: pd.DataFrame, comb_lst: List[str], **kwargs) -> pd.DataFrame:
        """ Generate interaction features based on predefined combinations. 
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
        X = X.astype('U')
        feat_combs_use = [i.split("_&_") for i in comb_lst]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=comb_lst, index=X.index)

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series, **kwargs):
        X = X_in.copy()
        y = y_in.copy()

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        if self.candidate_cols is None:
            self.candidate_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.candidate_cols = [i for i in self.candidate_cols if X[i].nunique() >= self.min_cardinality]  # TODO: Make this a parameter

        if len(self.candidate_cols) < self.max_order:
            self.new_col_set = []
            return self

        # TODO: Add smarter cardinality filtering if it is already clear at this point that too many features will be generated 

        X_new = pd.DataFrame(index=X.index)
        for order in range(2, self.max_order+1):
            X_new = pd.concat([X_new,
                self.combine(X[self.candidate_cols], order=order, max_feats=self.max_new_feats-X_new.shape[1])
            ], axis=1)

            # Apply frequency filter on new columns
            highest_freq = X_new.apply(lambda col: col.value_counts().iloc[0])>=self.min_count
            highest_freq.index[highest_freq.values]
            X_new = X_new.loc[: , highest_freq.index[highest_freq.values]]

            if X_new.shape[1] >= self.max_new_feats:
                break

        self.new_col_set = [c for c in X_new.columns if c not in X.columns]
        
        # X_new = DropDuplicatesFeatureGenerator().fit_transform(X_new)
        self.cat_transformer = CategoryFeatureGenerator(minimum_cat_count=1)
        X_new = self.cat_transformer.fit_transform(X_new)

        self.new_col_set = [c for c in X_new.columns if c not in X.columns]

        if self.add_freq or self.only_freq:
            # TODO: Unclear whether there is a more efficient way to do this
            cat_freq = FrequencyFeatureGenerator(fillna=self.fillna, log=self.log)
            candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X_new[self.new_col_set])
            if len(candidate_cols) > 0:
                self.cat_freq = FrequencyFeatureGenerator(candidate_cols=candidate_cols, fillna=self.fillna, log=self.log).fit(X_new[candidate_cols], y)
        return self
    

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X)
    
        features_out = list(X_out.columns)

        # type_group_map_special = {R_CATEGORY: features_out} # TODO: Find out whether that is needed
        return X_out, dict()

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if len(self.new_col_set) == 0:
            return X_in.copy()
        # TODO: Optimize inference time
        X = X_in.copy()

        X_out = pd.DataFrame(index=X.index)
        if len(self.new_col_set) > 0:
            for degree in range(2, self.max_order+1):
                col_set_use = [col for col in self.new_col_set if col.count('_&_')+1 == degree]
                if len(col_set_use) > 0:
                    X_degree = self.combine_predefined(X, col_set_use)
                    X_out = pd.concat([X_out, X_degree], axis=1)
            
            X_out = self.cat_transformer.transform(X_out)

            if self.add_freq or self.only_freq:
                X_out = self.cat_freq.transform(X_out)
            if self.only_freq:
                X_out = X_out.drop(self.new_col_set, axis=1, errors='ignore')
        return pd.concat([X, X_out], axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            # valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        )