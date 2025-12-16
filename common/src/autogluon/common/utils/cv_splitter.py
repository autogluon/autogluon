from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import KBinsDiscretizer

from .warning_filter import warning_filter

logger = logging.getLogger(__name__)


# TODO: Add binned stratification support for regression in train/val split (non CV)
class CVSplitter:
    def __init__(
        self,
        splitter_cls=None,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int | None = 0,
        stratify: bool = False,
        shuffle: bool = False,
        bin: bool = False,
        n_bins: int | None = None,
        groups: pd.Series = None,
    ):
        """
        Wrapper around splitter objects to perform KFold splits.
        Supports regression stratification via the `bin` and `n_bins` argument.

        Parameters
        ----------
        splitter_cls, default None
            The class to use for splitting.
            If None, will automatically be determined based off of `stratify`, `groups`, and `n_repeats`.
        n_splits : int, default 5
            The number of splits to perform.
            Ignored if `groups` is specified.
        n_repeats: int, default 1
            The number of repeated splits to perform.
            Ignored if `groups` is specified.
        random_state : int, default 0
            The seed to use when splitting the data.
        stratify : bool, default False
            If True, will stratify the splits on `y`.
        bin : bool, default False
            If True and `stratify` is True, will bin `y` into `n_bins` bins for stratification.
            Should only be used for regression and quantile tasks.
        n_bins : int, default None
            The number of bins to use when `bin` is True.
            If None, defaults to `np.floor(n_samples / n_splits)`.
        groups : pd.Series, default None
            If specified, splitter_cls will default to LeaveOneGroupOut.

        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle
        self.bin = bin
        self.n_bins = n_bins
        self.groups = groups
        if splitter_cls is None:
            splitter_cls = self._get_splitter_cls()
        self._splitter = self._get_splitter(splitter_cls)

    def _get_splitter_cls(self):
        if self.groups is not None:
            num_groups = len(self.groups.unique())
            if self.n_repeats != 1:
                raise AssertionError(
                    f"n_repeats must be 1 when split groups are specified. (n_repeats={self.n_repeats})"
                )
            self.n_splits = num_groups
            splitter_cls = LeaveOneGroupOut
            # pass
        elif self.stratify:
            splitter_cls = RepeatedStratifiedKFold
        else:
            splitter_cls = RepeatedKFold
        return splitter_cls

    def _get_splitter(self, splitter_cls) -> BaseCrossValidator:
        if splitter_cls == LeaveOneGroupOut:
            return splitter_cls()
        elif splitter_cls in [RepeatedKFold, RepeatedStratifiedKFold]:
            return splitter_cls(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        elif splitter_cls in [KFold, StratifiedKFold]:
            assert self.n_repeats == 1
            return splitter_cls(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        else:
            raise AssertionError(f"{splitter_cls} is not supported as a valid `splitter_cls` input to CVSplitter.")

    def split(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        splitter = self._splitter
        if isinstance(splitter, (RepeatedStratifiedKFold, StratifiedKFold)):
            if self.bin:
                if self.n_bins is None:
                    n_splits = splitter.get_n_splits()
                    n_samples = len(y)

                    # ensure at least n_splits samples per bin
                    n_bins = int(np.floor(n_samples / n_splits))
                else:
                    n_bins = self.n_bins

                if n_bins > 1:
                    k_bins_discretizer = KBinsDiscretizer(
                        n_bins=n_bins, encode="ordinal", random_state=self.random_state
                    )
                    y_bin = k_bins_discretizer.fit_transform(y.to_frame())[:, 0]
                    y = pd.Series(data=y_bin, index=y.index, name=y.name)
                else:
                    if isinstance(splitter, StratifiedKFold):
                        splitter_cls = KFold
                    else:
                        splitter_cls = RepeatedKFold
                    # Don't stratify, can't bin!
                    splitter = self._get_splitter(splitter_cls=splitter_cls)

            # FIXME: There is a bug in sklearn that causes an incorrect ValueError if performing stratification and all classes have fewer than n_splits samples.
            #  This is hacked by adding a dummy class with n_splits samples, performing the kfold split, then removing the dummy samples from all resulting indices.
            #  This is very inefficient and complicated and ideally should be fixed in sklearn.
            with warning_filter():
                try:
                    out = [[train_index, test_index] for train_index, test_index in splitter.split(X, y)]
                except:
                    y_dummy = pd.concat([y, pd.Series([-1] * self.n_splits)], ignore_index=True)
                    X_dummy = pd.concat([X, X.head(self.n_splits)], ignore_index=True)
                    invalid_index = set(list(y_dummy.tail(self.n_splits).index))
                    out = [[train_index, test_index] for train_index, test_index in splitter.split(X_dummy, y_dummy)]
                    len_out = len(out)
                    for i in range(len_out):
                        train_index, test_index = out[i]
                        out[i][0] = [index for index in train_index if index not in invalid_index]
                        out[i][1] = [index for index in test_index if index not in invalid_index]
            return out
        else:
            return [[train_index, test_index] for train_index, test_index in splitter.split(X, y, groups=self.groups)]
