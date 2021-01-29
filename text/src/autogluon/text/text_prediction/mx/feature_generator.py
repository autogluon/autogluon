import numpy as np
import pandas as pd
import functools
import collections
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.preprocessing import get_trimmed_lengths
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from autogluon.features import CategoryFeatureGenerator
from .. import constants as _C
from ..utils import parallel_transform


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
    for ele in data:
        out.append(np.array(tokenizer.encode(ele, int)))
    return out


class MultiModalTextModelFeatureTransform(TransformerMixin, BaseEstimator):
    def __init__(self, column_types, label_column, tokenizer, cfg=None):
        self._column_types = column_types
        self._label_column = label_column
        self._cfg = cfg
        self._generators = dict()
        for col_name, col_type in self._column_types:
            if col_type == _C.TEXT:
                continue
            elif col_type == _C.CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    minimum_cat_count=cfg.categorical.minimum_cat_count,
                    maximum_num_cat=cfg.categorical.maximum_num_cat)
                self._generators[col_name] = generator
            elif col_type == _C.NUMERICAL:
                generator = Pipeline([SimpleImputer(),
                                      StandardScaler(with_mean=cfg.numerical.scaler_with_mean,
                                                     with_std=cfg.numerical.scaler_with_std)])
                self._generators[col_name] = generator
        self._tokenizer = tokenizer
        self._fit_called = False
        self._ignore_columns_set = set()

    @property
    def cfg(self):
        return self._cfg

    @property
    def label_generator(self):
        return self._generators[self._label_column]

    def get_feature_batchify(self):
        """Get the batchify function of features

        Returns
        -------
        batchify_fn
        """

    def get_label_batchify(self):
        """Get the batchify function of labels

        Returns
        -------
        batchify_fn
        """

    def fit_transform(self, X, y):
        """Fit + Transform the dataframe

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
        processed_y
            The processed Y series
        """
        self._fit_called = True
        text_data_dict = collections.OrderedDict()
        categorical_data_dict = collections.OrderedDict()
        numerical_data_dict = collections.OrderedDict()
        for col_name in sorted(X.columns):
            col_type = self._column_types[col_name]
            if col_type == _C.NULL:
                self._ignore_columns_set.add(col_name)
                continue
            col_value = X[col_name]
            if col_type == _C.TEXT:
                text_data_dict[col_name] = parallel_transform(
                    df=col_value,
                    processing_fn=functools.partial(tokenize_data, tokenizer=self._tokenizer))
            elif col_type == _C.CATEGORICAL:
                if self.cfg.categorical.convert_to_text:
                    processed_data = col_value.apply(str)
                    text_data_dict[col_name] = processed_data
                else:
                    processed_data = col_value.astype('category')
                    processed_data =\
                        self._generators[col_type].fit_transform(
                            pd.DataFrame(processed_data)).iloc[:, 0]
                    if processed_data.

    def transform(self, X_df, y_df=None):
        """"

        """


    def __getstate__(self):
