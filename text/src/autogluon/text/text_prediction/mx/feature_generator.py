import numpy as np
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.preprocessing import get_trimmed_lengths
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from autogluon.features import CategoryMemoryMinimizeFeatureGenerator
from .. import constants as _C


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
    return cfg


class OrdinalMergeRaresHandleUnknownEncoder(_BaseEncoder):
    """Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    max_levels : int, default=None
        One less than the maximum number of categories to keep (max_levels = 2 means we keep 3 distinct categories).
        Infrequent categories are grouped together and mapped to the highest int
        Unknown categories encountered at test time are mapped to another extra category. Embedding layers should be able to take in max_levels + 1 categories!

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    infrequent_indices_: list of arrays of shape(n_infrequent_categories)
        ``infrequent_indices_[i]`` contains a list of indices in
        ``categories_[i]`` corresponsing to the infrequent categories.

    """

    def __init__(self, categories='auto', dtype=np.float64, max_levels=None):
        self.categories = categories
        self.dtype = dtype
        self.max_levels = max_levels

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        X = np.array(
            X).tolist()  # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        self._fit(X, handle_unknown='ignore')

        self.categories_as_sets_ = [set(categories) for categories in self.categories_]
        # new level introduced to account for unknown categories, always = 1 + total number of categories seen during training
        self.categories_unknown_level_ = [min(len(categories), self.max_levels) for categories in
                                          self.categories_]
        self.categories_len_ = [len(categories) for categories in self.categories_]
        return self

    def transform(self, X):
        """Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.

        """
        X_og_array = np.array(X)  # original X array before transform
        X = X_og_array.tolist()  # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        X_int, _ = self._transform(X,
                                   handle_unknown='ignore')  # will contain zeros for 0th category as well as unknown values.

        for i in range(X_int.shape[1]):
            X_col_data = X_og_array[:, i]
            cat_set = self.categories_as_sets_[i]
            unknown_elements = np.array([cat not in cat_set for cat in X_col_data.tolist()])
            X_int[unknown_elements, i] = self.categories_unknown_level_[
                i]  # replace entries with unknown categories with feature_i_numlevels + 1 value. Do NOT modify self.categories_

        return X_int.astype(self.dtype, copy=False)

    def inverse_transform(self, X):
        """Convert the data back to the original representation.
            In case unknown categories are encountered (all zeros in the one-hot encoding), ``None`` is used to represent this category.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.

        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        for i in range(n_features):
            possible_categories = np.append(self.categories_[i], None)
            labels = X[:, i].astype('int64', copy=False)
            X_tr[:, i] = self.categories_[i][labels]

        return X_tr


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
                generator = Pipeline([
                    OrdinalEncoder(dtype=np.int32)
                ])
            elif col_type == _C.NUMERICAL:
                generator =
        self._tokenizer = tokenizer

    def fit_transform(self, data_df):
        """Fit + Transform the dataframe

        Parameters
        ----------
        data_df
            The data frame

        Returns
        -------

        """


    def fit(self, data_df):
        """

        Parameters
        ----------
        data_df

        Returns
        -------

        """


    def transform(self, data_df):
        """"

        """

