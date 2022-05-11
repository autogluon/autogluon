""" 
Variant of the sklearn OneHotEncoder and OrdinalEncoder that can handle unknown classes at test-time 
as well as binning of infrequent categories to limit the overall number of categories considered.
Unknown categories are returned as None in inverse transforms. Always converts input list X to list of the same type elements first (string typically)
"""
from numbers import Integral

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


__all__ = [
    'OneHotMergeRaresHandleUnknownEncoder',
    'OrdinalMergeRaresHandleUnknownEncoder'
]


def _encode_numpy(values, uniques=None, encode=False, check_unknown=True):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        if encode:
            uniques, encoded = np.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            # unique sorts
            return np.unique(values)
    if encode:
        if check_unknown:
            diff = _encode_check_unknown(values, uniques)
            if diff:
                raise ValueError("y contains previously unseen labels: %s"
                                 % str(diff))
        encoded = np.searchsorted(uniques, values)
        return uniques, encoded
    else:
        return uniques


def _encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques


def _encode(values, uniques=None, encode=False, check_unknown=True):
    """Helper function to factorize (find uniques) and encode values.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.
    Parameters
    ----------
    values : array
        Values to factorize or encode.
    uniques : array, optional
        If passed, uniques are not determined from passed values (this
        can be because the user specified categories, or because they
        already have been determined in fit).
    encode : bool, default False
        If True, also encode the values into integer codes based on `uniques`.
    check_unknown : bool, default True
        If True, check for values in ``values`` that are not in ``unique``
        and raise an error. This is ignored for object dtype, and treated as
        True in this case. This parameter is useful for
        _BaseEncoder._transform() to avoid calling _encode_check_unknown()
        twice.
    Returns
    -------
    uniques
        If ``encode=False``. The unique values are sorted if the `uniques`
        parameter was None (and thus inferred from the data).
    (uniques, encoded)
        If ``encode=True``.
    """
    if values.dtype == object:
        try:
            res = _encode_python(values, uniques, encode)
        except TypeError:
            raise TypeError("argument must be a string or number")
        return res
    else:
        return _encode_numpy(values, uniques, encode,
                             check_unknown=check_unknown)


def _encode_check_unknown(values, uniques, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    Parameters
    ----------
    values : array
        Values to check for unknowns.
    uniques : array
        Allowed uniques values.
    return_mask : bool, default False
        If True, return a mask of the same shape as `values` indicating
        the valid values.
    Returns
    -------
    diff : list
        The unique values present in `values` and not in `uniques` (the
        unknown values).
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.
    """
    if values.dtype == object:
        uniques_set = set(uniques)
        diff = list(set(values) - uniques_set)
        if return_mask:
            if diff:
                valid_mask = np.array([val in uniques_set for val in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff
    else:
        unique_values = np.unique(values)
        diff = list(np.setdiff1d(unique_values, uniques, assume_unique=True))
        if return_mask:
            if diff:
                valid_mask = np.in1d(values, uniques)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff


class _BaseEncoder(BaseEstimator, TransformerMixin):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """
    
    def _check_X(self, X):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, eg for the `categories_` attribute.
        
        """
        if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, force_all_finite=False)
            if (not hasattr(X, 'dtype')
                    and np.issubdtype(X_temp.dtype, np.str_)):
                X = check_array(X, dtype=object)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = True
        
        n_samples, n_features = X.shape
        X_columns = []
        
        for i in range(n_features):
            Xi = self._get_feature(X, feature_idx=i)
            Xi = check_array(Xi, ensure_2d=False, dtype=None,
                             force_all_finite=needs_validation)
            X_columns.append(Xi)
        
        return X_columns, n_samples, n_features
    
    def _get_feature(self, X, feature_idx):
        if hasattr(X, 'iloc'):
            # pandas dataframes
            return X.iloc[:, feature_idx]
        # numpy arrays, sparse arrays
        return X[:, feature_idx]
    
    def _fit(self, X, handle_unknown='error'):
        X_list, n_samples, n_features = self._check_X(X)
        
        if self.categories != 'auto':
            if len(self.categories) != n_features:
                raise ValueError("Shape mismatch: if categories is an array,"
                                 " it has to be of shape (n_features,).")
        
        if self.max_levels is not None:
            if (not isinstance(self.max_levels, Integral) or
                    self.max_levels <= 0):
                raise ValueError("max_levels must be None or a strictly "
                                 "positive int, got {}.".format(
                                     self.max_levels))
        
        self.categories_ = []
        self.infrequent_indices_ = []
        
        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == 'auto':
                cats = _encode(Xi)
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype != object:
                    if not np.all(np.sort(cats) == cats):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")
                if handle_unknown == 'error':
                    diff = _encode_check_unknown(Xi, cats)
                    if diff:
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
            self.categories_.append(cats)
            
            if self.max_levels is not None:
                infrequent_indices = self._find_infrequent_category_indices(Xi)
            else:
                infrequent_indices = np.array([])
            self.infrequent_indices_.append(infrequent_indices)
    
    def _find_infrequent_category_indices(self, Xi):
        # TODO: this is using unique on X again. Ideally we should integrate
        # this into _encode()
        _, counts = np.unique(Xi, return_counts=True)
        return np.argsort(counts)[:-self.max_levels]
    
    def _transform(self, X, handle_unknown='error'):
        X_list, n_samples, n_features = self._check_X(X)
        
        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)
        
        if n_features != len(self.categories_):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features."
                .format(len(self.categories_,), n_features)
            )
        
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _encode_check_unknown(Xi, self.categories_[i],
                                                     return_mask=True)
            
            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (self.categories_[i].dtype.kind in ('U', 'S')
                            and self.categories_[i].itemsize > Xi.itemsize):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    else:
                        Xi = Xi.copy()
                    
                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _encode_check_unknown was
            # already called above.
            _, encoded = _encode(Xi, self.categories_[i], encode=True,
                                 check_unknown=False)
            X_int[:, i] = encoded
        
        # We need to take care of infrequent categories here. We want all the
        # infrequent categories to end up in a specific column, after all the
        # frequent ones. Let's say we have 4 categories with 2 infrequent
        # categories (and 2 frequent categories): we want the value in X_int
        # for the infrequent categories to be 2 (third and last column), and
        # the values for the frequent ones to be 0 and 1. The piece of code
        # below performs this mapping.
        # TODO: maybe integrate this part with the one above
        self._infrequent_mappings = {}
        huge_int = np.iinfo(X_int.dtype).max
        for feature_idx in range(n_features):
            if self.infrequent_indices_[feature_idx].size > 0:
                mapping = np.arange(len(self.categories_[feature_idx]))
                # Trick: set the infrequent cats columns to a very big int and
                # encode again.
                for ordinal_cat in self.infrequent_indices_[feature_idx]:
                    mapping[ordinal_cat] = huge_int
                _, mapping = _encode_numpy(mapping, encode=True)
                
                # update X_int and save mapping for later (for dropping logic)
                X_int[:, feature_idx] = mapping[X_int[:, feature_idx]]
                self._infrequent_mappings[feature_idx] = mapping
        
        return X_int, X_mask
    
    def _more_tags(self):
        return {'X_types': ['categorical']}


class OneHotMergeRaresHandleUnknownEncoder(_BaseEncoder):
    """Encode categorical integer features as a one-hot numeric array.
    
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter)
    
    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.
    
    Always uses handle_unknown='ignore' which maps unknown test-time categories to all zeros vector.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values, default='auto'.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

    drop : 'first' or a list/array of shape (n_features,), default=None.
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped. If ``drop[i]`` is an infrequent category, an
          error is raised: it is only possible to drop all of the infrequent
          categories, not just one of them.
        - 'infrequent' : drop the infrequent categories column (see
          ``max_levels`` parameter).

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=float
        Desired dtype of output.
        
    max_levels : int, default=None
        One less than the maximum number of categories to keep (max_levels = 2 means we keep 3 distinct categories). 
        Infrequent categories are grouped together and mapped into a single column, which counts as extra category.
        Unknown categories encountered at test time are mapped to all zeros vector.
    
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
        be dropped for each feature. None if all the transformed features will
        be retained.

    infrequent_indices_: list of arrays of shape(n_infrequent_categories)
        ``infrequent_indices_[i]`` contains a list of indices in
        ``categories_[i]`` corresponding to the infrequent categories.

    """
    
    def __init__(self, categories='auto', drop=None, sparse=True,
                 dtype=np.float64, max_levels=None):
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = 'ignore'
        self.drop = drop
        self.max_levels = max_levels
    
    def _validate_keywords(self):
        if self.handle_unknown not in ('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', "
                   "got {0}.".format(self.handle_unknown))
            raise ValueError(msg)
        # If we have both dropped columns and ignored unknown
        # values, there will be ambiguous cells. This creates difficulties
        # in interpreting the model.
        if self.drop is not None and self.handle_unknown != 'error':
            raise ValueError(
                "`handle_unknown` must be 'error' when the drop parameter is "
                "specified, as both would create categories that are all "
                "zero.")
    
    def _compute_drop_idx(self):
        if self.drop is None:
            return None
        elif (isinstance(self.drop, str) and
                self.drop in ('first', 'infrequent')):
            return np.zeros(len(self.categories_), dtype=np.int_)
        elif not isinstance(self.drop, str):
            try:
                self.drop = np.asarray(self.drop, dtype=object)
                droplen = len(self.drop)
            except (ValueError, TypeError):
                msg = ("Wrong input for parameter `drop`. Expected "
                       "'first', None or array of objects, got {}")
                raise ValueError(msg.format(type(self.drop)))
            if droplen != len(self.categories_):
                msg = ("`drop` should have length equal to the number "
                       "of features ({}), got {}")
                raise ValueError(msg.format(len(self.categories_),
                                            len(self.drop)))
            missing_drops = [(i, val) for i, val in enumerate(self.drop)
                             if val not in self.categories_[i]]
            if any(missing_drops):
                msg = ("The following categories were supposed to be "
                       "dropped, but were not found in the training "
                       "data.\n{}".format(
                           "\n".join(
                                ["Category: {}, Feature: {}".format(c, v)
                                    for c, v in missing_drops])))
                raise ValueError(msg)
            return np.array([np.where(cat_list == val)[0][0]
                             for (val, cat_list) in
                             zip(self.drop, self.categories_)], dtype=np.int_)
        else:
            msg = ("Wrong input for parameter `drop`. Expected "
                   "'first', None or array of objects, got {}")
            raise ValueError(msg.format(type(self.drop)))
    
    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.
    
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
    
        Returns
        -------
        self
        """
        X = np.array(X).tolist() # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        self._validate_keywords()
        self._fit(X, handle_unknown=self.handle_unknown)
        self.drop_idx_ = self._compute_drop_idx()
        # check if user wants to manually drop a feature that is
        # infrequent: this is not allowed
        if self.drop is not None and not isinstance(self.drop, str):
            for feature_idx, (infrequent_indices, drop_idx) in enumerate(
                    zip(self.infrequent_indices_, self.drop_idx_)):
                if drop_idx in infrequent_indices:
                    raise ValueError(
                        "Category {} of feature {} is infrequent and thus "
                        "cannot be dropped. Use drop='infrequent' "
                        "instead.".format(
                            self.categories_[feature_idx][drop_idx],
                            feature_idx
                        )
                    )
        return self
    
    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to fit(X).transform(X) but more convenient.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.
        """
        X = np.array(X).tolist() # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        self._validate_keywords()
        return super().fit_transform(X, y)
    
    def transform(self, X):
        """Transform X using one-hot encoding.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        
        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.
        """
        X = np.array(X).tolist() # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        check_is_fitted(self, 'categories_')
        # validation of X happens in _check_X called by _transform
        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown)
        n_samples, n_features = X_int.shape
        
        # n_columns indicates, for each feature, how many columns are used in
        # X_trans. By default this corresponds to the number of categories, but
        # will differ if we drop some of them, or if there are infrequent
        # categories (all mapped to the same column)
        n_columns = [len(cats) for cats in self.categories_]
        for feature_idx in range(n_features):
            n_infrequent = self.infrequent_indices_[feature_idx].size
            if n_infrequent > 0:
                # still add 1 for the infrequent column
                n_columns[feature_idx] += 1 - n_infrequent
            if self.drop is not None:
                # if drop is not None we always drop one column in general,
                # except when drop is 'infrequent' and there is no infrequent
                # category.
                n_columns[feature_idx] -= 1
                if (isinstance(self.drop, str) and self.drop == 'infrequent'
                        and n_infrequent == 0):
                    n_columns[feature_idx] += 1  # revert decrement from above
        
        if self.drop is not None:
            to_drop = self.drop_idx_.copy()
            if isinstance(self.drop, str):
                if self.drop == 'infrequent':
                    for feature_idx in range(n_features):
                        if self.infrequent_indices_[feature_idx].size > 0:
                            # drop the infrequent column (i.e. the last one)
                            to_drop[feature_idx] = n_columns[feature_idx]
                        else:
                            # no infrequent category, use special marker -1
                            # so that no dropping happens for this feature
                            to_drop[feature_idx] = -1
            else:
                # self.drop is an array of categories. we need to remap the
                # dropped indexes if some of the categories are infrequent.
                # see _transform() for details about the mapping.
                for feature_idx in range(n_features):
                    if self.infrequent_indices_[feature_idx].size > 0:
                        mapping = self._infrequent_mappings[feature_idx]
                        to_drop[feature_idx] = mapping[to_drop[feature_idx]]
            
            # We remove all the dropped categories from mask, and decrement
            # all categories that occur after them to avoid an empty column.
            to_drop = to_drop.reshape(1, -1)
            keep_cells = (X_int != to_drop) | (to_drop == -1)
            X_mask &= keep_cells
            X_int[(X_int > to_drop) & (to_drop != -1)] -= 1
        
        mask = X_mask.ravel()
        n_values = np.array([0] + n_columns)
        feature_indices = np.cumsum(n_values)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]
        
        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if not self.sparse:
            return out.toarray()
        else:
            return out
    
    def inverse_transform(self, X):
        """Convert the back data to the original representation.
        
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.
        
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
        if self.drop is None:
            n_transformed_features = sum(len(cats)
                                         for cats in self.categories_)
        else:
            n_transformed_features = sum(len(cats) - 1
                                         for cats in self.categories_)
        
        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if X.shape[1] != n_transformed_features:
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))
        
        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        j = 0
        found_unknown = {}
        
        for i in range(n_features):
            if self.drop is None:
                cats = self.categories_[i]
            else:
                cats = np.delete(self.categories_[i], self.drop_idx_[i])
            n_categories = len(cats)
        
            # Only happens if there was a column with a unique
            # category. In this case we just fill the column with this
            # unique category value.
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
                j += n_categories
                continue
            sub = X[:, j:j + n_categories]  # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats[labels]
            if self.handle_unknown == 'ignore':
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                # ignored unknown categories: we have a row of all zero
                if unknown.any():
                    found_unknown[i] = unknown
            # drop will either be None or handle_unknown will be error. If
            # self.drop is not None, then we can safely assume that all of
            # the nulls in each column are the dropped value
            elif self.drop is not None:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    X_tr[dropped, i] = self.categories_[i][self.drop_idx_[i]]
            
            j += n_categories
        
        # if ignored are found: potentially need to upcast result to
        # insert None values
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)
        
            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None
        
        return X_tr
    
    def get_feature_names(self, input_features=None):
        """Return feature names for output features.
    
        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.
        
        Returns
        -------
        output_feature_names : array of string, length n_output_features
        
        """
        check_is_fitted(self, 'categories_')
        cats = self.categories_
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))
        
        feature_names = []
        for i in range(len(cats)):
            names = [
                input_features[i] + '_' + str(t) for t in cats[i]]
            if self.drop is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)
        
        return np.array(feature_names, dtype=object)


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
        X = np.array(X).tolist()  # converts all elements in X to the same type (i.e. cannot mix floats, ints, and str)
        self._fit(X, handle_unknown='ignore')

        self.categories_as_sets_ = [set(categories) for categories in self.categories_]
        # new level introduced to account for unknown categories, always = 1 + total number of categories seen during training
        self.categories_unknown_level_ = [min(len(categories), self.max_levels) for categories in self.categories_]
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
        X_int, _ = self._transform(X, handle_unknown='ignore')  # will contain zeros for 0th category as well as unknown values.

        for i in range(X_int.shape[1]):
            X_col_data = X_og_array[:, i]
            cat_set = self.categories_as_sets_[i]
            unknown_elements = np.array([cat not in cat_set for cat in X_col_data.tolist()])
            X_int[unknown_elements, i] = self.categories_unknown_level_[i]  # replace entries with unknown categories with feature_i_numlevels + 1 value. Do NOT modify self.categories_

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
