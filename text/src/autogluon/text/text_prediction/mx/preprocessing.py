import numpy as np
import os
import pandas as pd
import functools
import collections
from mxnet.gluon.data import ArrayDataset
from autogluon_contrib_nlp.utils.config import CfgNode
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


class MultiModalTextFeatureProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, column_types, label_column, tokenizer,
                 logger=None, cfg=None):
        self._column_types = column_types
        self._label_column = label_column
        cfg = base_preprocess_cfg().clone_merge(cfg)
        self._cfg = cfg
        self._feature_generators = dict()
        self._label_generator = None
        self._logger = logger
        for col_name, col_type in self._column_types.items():
            if col_name == self._label_column:
                continue
            if col_type == _C.TEXT:
                continue
            elif col_type == _C.CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    cat_order='count',
                    minimum_cat_count=cfg.categorical.minimum_cat_count,
                    maximum_num_cat=cfg.categorical.maximum_num_cat)
                self._feature_generators[col_name] = generator
            elif col_type == _C.NUMERICAL:
                generator = Pipeline(
                    [('imputer', SimpleImputer()),
                     ('scaler', StandardScaler(with_mean=cfg.numerical.scaler_with_mean,
                                               with_std=cfg.numerical.scaler_with_std))]
                )
                self._feature_generators[col_name] = generator
        if self._column_types[label_column] == _C.CATEGORICAL:
            self._label_generator = LabelEncoder()

        self._tokenizer = tokenizer
        self._fit_called = False

        # Some columns will be ignored
        self._ignore_columns_set = set()
        self._text_feature_names = []
        self._categorical_feature_names = []
        self._categorical_num_categories = []
        self._numerical_feature_names = []

    @property
    def text_feature_names(self):
        return self._text_feature_names

    @property
    def categorical_feature_names(self):
        return self._categorical_feature_names

    @property
    def numerical_feature_names(self):
        return self._numerical_feature_names

    @property
    def categorical_num_categories(self):
        """We will always include the unknown category"""
        return self._categorical_num_categories

    @property
    def cfg(self):
        return self._cfg

    @property
    def label_maps(self):
        return self._label_maps

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
            The processed X data
        (processed_y)
            The processed Y data
        """
        if self._fit_called:
            raise RuntimeError('Fit has been called. Please create a new preprocessor and call '
                               'fit again!')
        self._fit_called = True
        text_features = []
        categorical_features = []
        numerical_features = []
        for col_name in sorted(X.columns):
            col_type = self._column_types[col_name]
            col_value = X[col_name]
            if col_type == _C.NULL:
                self._ignore_columns_set.add(col_name)
                continue
            elif col_type == _C.TEXT:
                processed_col_value = parallel_transform(
                    df=col_value,
                    chunk_processor=functools.partial(tokenize_data,
                                                      tokenizer=self._tokenizer))
                text_features.append(processed_col_value)
                self._text_feature_names.append(col_name)
            elif col_type == _C.CATEGORICAL:
                if self.cfg.categorical.convert_to_text:
                    # Convert categorical column as text column
                    processed_data = col_value.apply(lambda ele: '' if ele is None else str(ele))
                    if len(np.unique(processed_data)) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    processed_data = parallel_transform(
                        df=processed_data,
                        chunk_processor=functools.partial(tokenize_data, tokenizer=self._tokenizer))
                    text_features.append(processed_data)
                    self._text_feature_names.append(col_name)
                else:
                    processed_data = col_value.astype('category')
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(
                        pd.DataFrame({col_name: processed_data})).iloc[:, 0]
                    if len(np.unique(processed_data)) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    num_categories = len(generator.category_map[col_name]) + 1
                    self._categorical_num_categories.append(num_categories)
                    processed_data = processed_data.to_numpy()
                    processed_data = np.nan_to_num(processed_data, True, num_categories - 1)
                    categorical_features.append(processed_data)
                    self._categorical_feature_names.append(col_name)
            elif col_type == _C.NUMERICAL:
                processed_data = pd.to_numeric(col_value)
                if len(processed_data.unique()) == 1:
                    self._ignore_columns_set.add(col_name)
                    continue
                if self.cfg.numerical.convert_to_text:
                    processed_data = col_value.apply('{:.3f}'.format)
                    processed_data = parallel_transform(
                        df=processed_data,
                        chunk_processor=functools.partial(tokenize_data, tokenizer=self._tokenizer))
                    text_features.append(processed_data)
                    self._text_feature_names.append(col_name)
                else:
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(
                        np.expand_dims(processed_data.to_numpy(), axis=-1))[:, 0]
                    numerical_features.append(processed_data)
                    self._numerical_feature_names.append(col_name)
            else:
                raise NotImplementedError(f'Type of the column is not supported currently. '
                                          f'Received {col_name}={col_type}.')
        if len(numerical_features) > 0:
            numerical_features = np.stack(numerical_features, axis=-1)
        if self._column_types[self._label_column] == _C.CATEGORICAL:
            y = self._label_generator.fit_transform(y)
        elif self._column_types[self._label_column] == _C.NUMERICAL:
            y = pd.to_numeric(y).to_numpy()
        else:
            raise NotImplementedError(f'Type of label column is not supported. '
                                      f'Label column type={self._label_column}')
        # Wrap the processed features and labels into a training dataset
        dataset = ArrayDataset(text_features + categorical_features + [numerical_features, y])
        return dataset

    def transform(self, X_df, y_df=None):
        """"

        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit_transform before calling ' \
                                 'preprocessor.transform.'
        pass

