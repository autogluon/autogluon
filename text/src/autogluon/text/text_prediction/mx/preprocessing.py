import numpy as np
import os
import pandas as pd
import functools
import collections
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.preprocessing import get_trimmed_lengths
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from autogluon.features import CategoryFeatureGenerator
from .. import constants as _C
from ..utils import parallel_transform

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def base_preprocess_cfg():
    cfg = CfgNode()
    cfg.text = CfgNode()
    cfg.text.merge = True                     # Whether we will merge different text columns
                                              # or treat them independently.
    cfg.text.max_length = 512                 # The maximum possible length.
    cfg.text.auto_max_length = True           # Try to automatically shrink the maximal length
                                              # based on the statistics of the dataset.
    cfg.categorical = CfgNode()
    cfg.categorical.minimum_cat_count = 100   # The minimal number of data per categorical group
    cfg.categorical.maximum_num_cat = 20      # The minimal number of data per categorical group
    cfg.categorical.convert_to_text = False   # Whether to convert the feature to text

    cfg.numerical = CfgNode()
    cfg.numerical.convert_to_text = False     # Whether to convert the feature to text
    cfg.numerical.impute_strategy = 'mean'    # Whether to use mean to fill in the missing values.
    cfg.numerical.scaler_with_mean = True     # Whether to normalize with mean
    cfg.numerical.scaler_with_std = True      # Whether to normalize with std
    return cfg


def tokenize_data(data: pd.Series, tokenizer):
    out = []
    if data is not None:
        for idx, ele in data.iteritems():
            if ele is None:
                out.append(np.ones((0,), dtype=np.int32))
            else:
                out.append(np.array(tokenizer.encode(ele, int), dtype=np.int32))
    return out


class MultiModalTextFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column_types, label_column, tokenizer,
                 logger=None, cfg=None):
        self._column_types = column_types
        self._label_column = label_column
        cfg = base_preprocess_cfg().clone_merge(cfg)
        self._cfg = cfg
        self._generators = dict()
        self._logger = logger
        for col_name, col_type in self._column_types.items():
            if col_name == self._label_column:
                continue
            if col_type == _C.TEXT:
                continue
            elif col_type == _C.CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    minimum_cat_count=cfg.categorical.minimum_cat_count,
                    maximum_num_cat=cfg.categorical.maximum_num_cat)
                self._generators[col_name] = generator
            elif col_type == _C.NUMERICAL:
                generator = Pipeline(
                    [('imputer', SimpleImputer()),
                     ('scaler', StandardScaler(with_mean=cfg.numerical.scaler_with_mean,
                                               with_std=cfg.numerical.scaler_with_std))]
                )
                self._generators[col_name] = generator
        self._label_generator = None
        self._tokenizer = tokenizer
        self._fit_called = False
        self._ignore_columns_set = set()

    @property
    def cfg(self):
        return self._cfg

    def fit_transform(self, X, y):
        """Fit and Transform the dataframe

        Parameters
        ----------
        X
            The feature dataframe
        y
            The label series

        Returns
        -------
        processed_X
            The processed X dataframe
        (processed_y)
            The processed Y series
        """
        self._fit_called = True
        text_data_l = []
        categorical_data_l = []
        numerical_data_l = []
        for col_name in sorted(X.columns):
            col_type = self._column_types[col_name]
            if col_type == _C.NULL:
                self._ignore_columns_set.add(col_name)
                continue
            col_value = X[col_name]
            if col_type == _C.TEXT:
                processed_col_value = parallel_transform(
                    df=col_value,
                    chunk_processor=functools.partial(tokenize_data,
                                                      tokenizer=self._tokenizer))
                text_data_l.append((col_name, processed_col_value))
            elif col_type == _C.CATEGORICAL:
                if self.cfg.categorical.convert_to_text:
                    processed_data = col_value.apply(lambda ele: '' if ele is None else str(ele))
                    text_data_l.append((col_name, processed_data))
                else:
                    processed_data = col_value.astype('category')
                    processed_data =\
                        self._generators[col_name].fit_transform(
                            pd.DataFrame(processed_data)).iloc[:, 0]
                    if len(processed_data.unique()) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    categorical_data_l.append((col_name, processed_data))
            elif col_type == _C.NUMERICAL:
                processed_data = pd.to_numeric(col_value)
                if self.cfg.numerical.convert_to_text:
                    processed_data = col_value.apply('{:.3f}'.format)
                    text_data_l.append((col_name, processed_data.to_numpy()))
                else:
                    processed_data = self._generators[col_name]\
                        .fit_transform(processed_data.to_numpy().expand_dims(axis=-1))[:, -1]
                if len(processed_data.unique()) == 1:
                    self._ignore_columns_set.add(col_name)
                    continue
                numerical_data_l.append((col_name, processed_data))
        if self._column_types[self._label_column] == _C.CATEGORICAL:
            if self._label_generator is None:
                self._label_generator = LabelEncoder().fit(y)
            y = self._label_generator.transform(y)
        elif self._column_types[self._label_column] == _C.NUMERICAL:
            y = pd.to_numeric(y)
        else:
            raise NotImplementedError(f'Type of label column is not supported. '
                                      f'Label column type={self._label_column}')
        # Return the processed features and labels
        return text_data_l, categorical_data_l, numerical_data_l, y

    def transform(self, X_df, y_df=None):
        """"

        """
        pass
