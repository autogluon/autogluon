"""
Code adapted from skrub==0.6.2
"""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from ._sklearn_compat import validate_data


def _mask_inf(X):
    """Replace infinite values with NaN and return their sign."""
    if (mask_inf := np.isinf(X)).any():
        sign = np.sign(X)
        X = np.where(mask_inf, np.nan, X)
        # 0 when X is finite, 1 when X is +inf, -1 when X is -inf
        mask_inf = mask_inf.astype(X.dtype) * sign

    return X, mask_inf


def _set_zeros(X, zero_cols):
    """Set the finite values of the specified columns to zero."""
    mask = np.isfinite(X)
    mask[:, ~zero_cols] = False
    X[mask] = 0.0
    return X


def _soft_clip(X, max_absolute_value, mask_inf):
    """Apply a soft clipping to the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be clipped.
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.
    mask_inf : array-like, shape (n_samples, n_features)
        A mask indicating the positions of infinite values in the input data and their
        signs.

    Returns
    -------
    X_clipped : array-like, shape (n_samples, n_features)
        The clipped version of the input.
    """
    X = X / np.sqrt(1 + (X / max_absolute_value) ** 2)
    X = np.where(mask_inf == 1, max_absolute_value, X)
    X = np.where(mask_inf == -1, -max_absolute_value, X)
    return X


class _MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """A variation of scikit-learn MinMaxScaler.

    A simple min-max scaler that centers the median to zero and scales
    the data to the range [-2, 2].

    scikit-learn MinMaxScaler computes the following::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    This scaler computes the following::

        X_std = (X - median) / (X.max(axis=0) - X.min(axis=0) + eps)
        X_scaled = X_std * (max - min) + min

    where we set min = 0 and max = 2.
    """

    def fit(self, X, y=None):
        del y
        eps = np.finfo("float32").tiny
        self.median_ = np.nanmedian(X, axis=0)
        self.scale_ = 2 / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + eps)
        return self

    def transform(self, X):
        check_is_fitted(self, ["median_", "scale_"])
        return self.scale_ * (X - self.median_)


class SquashingScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    r"""Perform robust centering and scaling followed by soft clipping.

    When features have large outliers, smooth clipping prevents the outliers from
    affecting the result too strongly, while robust scaling prevents the outliers from
    affecting the inlier scaling. Infinite values are mapped to the corresponding
    boundaries of the interval. NaN values are preserved.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.

    quantile_range : tuple of float, default=(0.25, 0.75)
        The quantiles used to compute the scaling factor. The first value is the lower
        quantile and the second value is the upper quantile. The default values are the
        25th and 75th percentiles, respectively. The quantiles are used to compute the
        scaling factor for the robust scaling step. The quantiles are computed from the
        finite values in the input column. If the two quantiles are equal, the scaling
        factor is computed from the 0th and 100th percentiles (i.e., the minimum and
        maximum values of the finite values in the input column).

    Notes
    -----
    This transformer is applied to each column independently. It uses two stages:

    1. The first stage centers the median of the data to zero and multiplies the data by a
       scaling factor determined from quantiles of the distribution, using
       scikit-learn's :class:`~sklearn.preprocessing.RobustScaler`. It also handles
       edge-cases in which the two quantiles are equal by following-up with a
       :class:`~sklearn.preprocessing.MinMaxScaler`.
    2. The second stage applies a soft clipping to the transformed data to limit the
       data to the interval ``[-max_absolute_value, max_absolute_value]`` in an
       injective way.

    Infinite values will be mapped to the corresponding boundaries of the interval. NaN
    values will be preserved.

    The formula for the transform is:

    .. math::

        \begin{align*}
            a &:= \begin{cases}
                1/(q_{\beta} - q_{\alpha}) &\text{if} \quad q_{\beta} \neq q_{\alpha} \\
                2/(q_1 - q_0) &\text{if}\quad q_{\beta} = q_{\alpha} \text{ and } q_1
                \neq q_0 \\ 0 & \text{otherwise}
            \end{cases} \\ z &:= a.(x - q_{1/2}), \\ x_{\text{out}} &:= \frac{z}{\sqrt{1
            + (z/B)^2}},
        \end{align*}

    where:

    - :math:`x` is a value in the input column.
    - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
    - :math:`B` is max_abs_value
    - :math:`\alpha` is the lower quantile
    - :math:`\beta` is the upper quantile.

    References
    ----------
    This method has been introduced as the robust scaling and smooth clipping transform
    in `Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data
    (Holzmüller et al., 2024) <https://arxiv.org/abs/2407.04491>`_.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from skrub import SquashingScaler

    In the general case, this scale uses a RobustScaler:

    >>> X = pd.DataFrame(dict(col=[np.inf, -np.inf, 3, -1, np.nan, 2]))
    >>> SquashingScaler(max_absolute_value=3).fit_transform(X)
    array([[ 3.        ],
           [-3.        ],
           [ 0.49319696],
           [-1.34164079],
           [        nan],
           [ 0.        ]])

    When quantile ranges are equal, this scaler uses a customized MinMaxScaler:

    >>> X = pd.DataFrame(dict(col=[0, 1, 1, 1, 2, np.nan]))
    >>> SquashingScaler().fit_transform(X)
    array([[-0.9486833],
           [ 0.       ],
           [ 0.       ],
           [ 0.       ],
           [ 0.9486833],
           [       nan]])

    Finally, when the min and max are equal, this scaler fills the column with zeros:

    >>> X = pd.DataFrame(dict(col=[1, 1, 1, np.nan]))
    >>> SquashingScaler().fit_transform(X)
    array([[ 0.],
           [ 0.],
           [ 0.],
           [nan]])
    """  # noqa: E501

    def __init__(
        self,
        max_absolute_value=3.0,
        quantile_range=(25.0, 75.0),
    ):
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        """Fit the transformer to a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        self : SquashingScaler
            The fitted transformer.
        """
        del y

        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the transformer and transform a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out: numpy array, shape (n_samples, n_features)
            The transformed version of the input.
        """
        del y

        if not (
            isinstance(self.max_absolute_value, numbers.Number)
            and np.isfinite(self.max_absolute_value)
            and self.max_absolute_value > 0
        ):
            raise ValueError(
                f"Got max_absolute_value={self.max_absolute_value!r}, but expected a positive finite number."
            )
        X = validate_data(
            self,
            X=X,
            reset=True,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        # To use sklearn scalers, we need to convert np.inf to np.nan. However, we need
        # to replace the original ±np.inf with ±max_absolute_value in the final output.
        # mask_inf is a 2D array containing the sign of the np.inf in the input.
        X, mask_inf = _mask_inf(X)

        # For each column, we apply 1 out of 3 scaling methods:
        # If the max is equal to the min, then we fill the column with zeros.
        zero_cols = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)

        # If the two quantiles defined by quantile_range have the same values, we
        # use a customized MinMaxScaler. We remove from this selection columns that
        # are already selected as zero cols (i.e. columns that have the same min and max
        # also have the same quantile_range values).
        quantiles = np.nanpercentile(X, self.quantile_range, axis=0)
        minmax_cols = quantiles[0, :] == quantiles[1, :]
        minmax_cols = minmax_cols & ~zero_cols

        # Otherwise (general case), we use a RobustScaler.
        robust_cols = ~(minmax_cols | zero_cols)

        # Copying the input since we change the values in place.
        X_tr = X.copy()
        if robust_cols.any():
            self.robust_scaler_ = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=self.quantile_range,
                copy=True,
            )
            X_tr[:, robust_cols] = self.robust_scaler_.fit_transform(X[:, robust_cols])
        else:
            self.robust_scaler_ = None
        self.robust_cols_ = robust_cols

        if minmax_cols.any():
            self.minmax_scaler_ = _MinMaxScaler()
            X_tr[:, minmax_cols] = self.minmax_scaler_.fit_transform(X[:, minmax_cols])
        else:
            self.minmax_scaler_ = None
        self.minmax_cols_ = minmax_cols

        if zero_cols.any():
            X_tr = _set_zeros(X_tr, zero_cols)
        self.zero_cols_ = zero_cols

        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out: numpy array of shape (n_samples, n_features)
            The transformed version of the input.
        """
        check_is_fitted(self, ["robust_scaler_", "minmax_scaler_"])

        X = validate_data(
            self,
            X=X,
            reset=False,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        X, mask_inf = _mask_inf(X)

        X_tr = X.copy()
        if self.robust_cols_.any():
            X_tr[:, self.robust_cols_] = self.robust_scaler_.transform(X[:, self.robust_cols_])
        if self.minmax_cols_.any():
            X_tr[:, self.minmax_cols_] = self.minmax_scaler_.transform(X[:, self.minmax_cols_])
        if self.zero_cols_.any():
            # if the scale is 0, we set the values to 0
            X_tr = _set_zeros(X_tr, self.zero_cols_)

        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)
