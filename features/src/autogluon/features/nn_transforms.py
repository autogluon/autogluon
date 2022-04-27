import copy
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from autogluon.common.features.types import R_FLOAT, R_DATETIME, S_TEXT_SPECIAL

logger = logging.getLogger(__name__)


class ContinuousNormalizer:

    def __init__(self, cont_columns) -> None:
        self.cont_columns = cont_columns
        self.stats = None

    def fit(self, X):
        self.stats = (X[self.cont_columns].mean(), X[self.cont_columns].std())

    def transform(self, X):
        cont_mean, cont_std = self.stats
        X[self.cont_columns] = (X[self.cont_columns] - cont_mean) / cont_std
        return X


class MissingFiller:

    def __init__(self, feature_metadata) -> None:
        self.columns_fills = None
        self.feature_metadata = feature_metadata

    def fit_transform(self, X):
        nullable_numeric_features = self.feature_metadata.get_features(valid_raw_types=[R_FLOAT, R_DATETIME], invalid_special_types=[S_TEXT_SPECIAL])
        self.columns_fills = dict()
        for c in nullable_numeric_features:  # No need to do this for int features, int can't have null
            self.columns_fills[c] = X[c].mean()
        return self.transform(X)

    def transform(self, X):
        return self._fill_missing(X, self.columns_fills)

    def _fill_missing(self, df: pd.DataFrame, columns_fills) -> pd.DataFrame:
        if columns_fills:
            df = df.fillna(columns_fills, inplace=False, downcast=False)
        else:
            df = df.copy()
        return df


class CategoricalFeaturesFilter:

    @staticmethod
    def filter(X, cat_columns, max_unique_categorical_values=10000):
        num_cat_cols_og = len(cat_columns)
        if cat_columns:
            try:
                X_stats = X[cat_columns].describe(include='all').T.reset_index()
                cat_cols_to_drop = list(X_stats[(X_stats['unique'] > max_unique_categorical_values) | (X_stats['unique'].isna())]['index'].values)
            except:
                cat_cols_to_drop = []
            if len(cat_cols_to_drop) != 0:
                cat_cols_to_drop = set(cat_cols_to_drop)
                cat_columns = [col for col in cat_columns if (col not in cat_cols_to_drop)]
        num_cat_cols_use = len(cat_columns)
        logger.log(15, f'Using {num_cat_cols_use}/{num_cat_cols_og} categorical features')
        return cat_columns


class TargetScaler:

    def __init__(self, problem_type, y_scaler=None) -> None:
        self.problem_type = problem_type
        if y_scaler is None:
            if problem_type == 'regression':
                self.y_scaler = StandardScaler()
            elif problem_type == 'quantile':
                self.y_scaler = MinMaxScaler()
            else:
                self.y_scaler = None
        else:
            self.y_scaler = copy.deepcopy(y_scaler)

    def fit_transform(self, y, y_val):
        if self.problem_type in ['regression', 'quantile'] and self.y_scaler is not None:
            y_norm = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1))
            y_val_norm = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)) if y_val is not None else None
            logger.log(0, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            y_norm = y
            y_val_norm = y_val
        return y_norm, y_val_norm

    def inverse_transform(self, preds):
        if self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        return preds
