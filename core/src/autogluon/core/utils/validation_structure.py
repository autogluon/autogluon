from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationStructure:
    """Declarative description of a dataset's validation-relevant structure.

    Users specify only the semantic columns; fold counts, interval blocking, and
    clamping are derived. Consumed by the learner (bagged path: fold-id labels routed
    through the existing ``groups`` channel) and the trainer (holdout path: a
    group-disjoint or temporally-forward train/validation split).

    Parameters
    ----------
    group_on : str | list[str], optional
        Column(s) identifying groups that must not span both a training and a
        validation split (e.g. a customer or session id).
    time_on : str, optional
        Column encoding time (datetime or numeric). Validation splits become
        contiguous time blocks; the non-bagged holdout is the latest block.
    stratify_on : str, optional
        Column to stratify splits on. Defaults to the label for classification when
        combined with ``group_on``; ignored for time-based splits (stratification is
        not meaningful across a temporal cut).

    Specifying both ``group_on`` and ``time_on`` is not supported.
    """

    group_on: str | list[str] | None = None
    time_on: str | None = None
    stratify_on: str | None = None

    def __post_init__(self):
        if self.group_on is not None and self.time_on is not None:
            raise NotImplementedError("Specifying both `group_on` and `time_on` is not supported.")
        if self.group_on is None and self.time_on is None and self.stratify_on is None:
            raise ValueError("ValidationStructure requires at least one of `group_on`, `time_on`, `stratify_on`.")

    @classmethod
    def from_input(cls, value: ValidationStructure | dict | None) -> ValidationStructure | None:
        if value is None or isinstance(value, ValidationStructure):
            return value
        if isinstance(value, dict):
            invalid = set(value) - {"group_on", "time_on", "stratify_on"}
            if invalid:
                raise ValueError(
                    f"Invalid `validation_structure` keys: {sorted(invalid)}. "
                    "Valid keys: ['group_on', 'time_on', 'stratify_on']"
                )
            return cls(**value)
        raise ValueError(f"`validation_structure` must be a dict or ValidationStructure, got: {type(value)}")

    # ── column access helpers ────────────────────────────────────────────────────

    def _group_values(self, X: pd.DataFrame) -> pd.Series:
        columns = [self.group_on] if isinstance(self.group_on, str) else list(self.group_on)
        for column in columns:
            if column not in X.columns:
                raise KeyError(f"`group_on` column {column!r} not found in the training data.")
            if X[column].isna().any():
                raise ValueError(f"`group_on` column {column!r} contains NaN values.")
        if len(columns) == 1:
            return X[columns[0]]
        return X[columns].astype(str).agg("||".join, axis=1)

    def _time_values(self, X: pd.DataFrame) -> pd.Series:
        if self.time_on not in X.columns:
            raise KeyError(f"`time_on` column {self.time_on!r} not found in the training data.")
        values = X[self.time_on]
        if values.isna().any():
            raise ValueError(f"`time_on` column {self.time_on!r} contains NaN values.")
        if pd.api.types.is_datetime64_any_dtype(values):
            values = values.astype("int64")
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"`time_on` column {self.time_on!r} must be datetime or numeric.")
        return values

    def _stratify_values(self, X: pd.DataFrame, y: pd.Series) -> pd.Series | None:
        if self.stratify_on is None:
            return None
        if self.stratify_on in X.columns:
            return X[self.stratify_on].astype("category")
        raise KeyError(f"`stratify_on` column {self.stratify_on!r} not found in the training data.")

    # ── bagged path ──────────────────────────────────────────────────────────────

    def fold_ids(self, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int = 0) -> pd.Series:
        """Per-row fold labels honoring the declared structure.

        Rows sharing a label form one validation fold (consumed as ``groups`` by the
        bagging ``CVSplitter``, i.e. leave-one-fold-out). Time produces contiguous
        blocks that never split ties; groups are assigned to folds group-disjointly
        (stratified when a stratification signal exists). The number of distinct
        labels may be lower than ``n_splits`` when the data cannot support it.
        """
        assert n_splits >= 2
        if self.time_on is not None:
            labels = _time_blocks(self._time_values(X), n_blocks=n_splits)
        elif self.group_on is not None:
            labels = _group_folds(
                groups=self._group_values(X),
                stratify=self._resolve_stratify_for_folds(X, y),
                n_splits=n_splits,
                random_state=random_state,
            )
        else:
            labels = _stratified_folds(self._stratify_values(X, y), n_splits=n_splits, random_state=random_state)
        n_folds = labels.nunique()
        if n_folds < n_splits:
            logger.log(
                20,
                f"validation_structure: data supports only {n_folds} folds (requested {n_splits}).",
            )
        return pd.Series(labels, index=X.index, name="__fold_id__")

    def _resolve_stratify_for_folds(self, X: pd.DataFrame, y: pd.Series) -> pd.Series | None:
        stratify = self._stratify_values(X, y)
        if stratify is None and y.nunique() <= max(20, int(np.sqrt(len(y)))):
            # No explicit stratification column: fall back to the label when it looks
            # categorical (mirrors AutoGluon's default label-stratified splitting).
            stratify = y
        return stratify

    # ── holdout path ─────────────────────────────────────────────────────────────

    def holdout_split_indices(
        self, X: pd.DataFrame, y: pd.Series, holdout_frac: float, random_state: int = 0
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A single structure-aware ``(train_idx, val_idx)`` positional split.

        Group structure yields a group-disjoint split; time structure yields a
        *forward* holdout (the latest contiguous block — a random fold would leak
        future information into training). Returns None when only ``stratify_on`` is
        set: AutoGluon's default label-stratified holdout needs no correction.
        """
        if self.time_on is None and self.group_on is None:
            return None
        if self.time_on is not None:
            values = self._time_values(X).to_numpy()
            order = np.argsort(values, kind="stable")
            cut = len(X) - max(1, int(round(len(X) * holdout_frac)))
            # never split rows with identical time stamps across the boundary
            while 0 < cut < len(X) and values[order[cut - 1]] == values[order[cut]]:
                cut -= 1
            if cut == 0:
                raise ValueError(
                    f"`time_on` column {self.time_on!r} cannot produce a non-empty forward holdout "
                    f"(all rows in the holdout block share the earliest time value)."
                )
            return order[:cut], order[cut:]

        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=random_state)
        train_idx, val_idx = next(splitter.split(X, y, groups=self._group_values(X)))
        return train_idx, val_idx


def _time_blocks(values: pd.Series, n_blocks: int) -> pd.Series:
    """Contiguous, tie-preserving time blocks of near-equal row counts (labels 0..k-1)."""
    counts = values.value_counts().sort_index()
    n_rows = int(counts.sum())
    target = n_rows / n_blocks
    block_of_value = {}
    block, filled = 0, 0
    for value, count in counts.items():
        # close the current block once it reached its share, keeping later blocks feasible
        if filled >= target * (block + 1) and block < n_blocks - 1:
            block += 1
        block_of_value[value] = block
        filled += int(count)
    return values.map(block_of_value)


def _group_folds(groups: pd.Series, stratify: pd.Series | None, n_splits: int, random_state: int) -> pd.Series:
    """Group-disjoint fold labels via (Stratified)GroupKFold test-fold membership."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        raise ValueError(f"`group_on` needs at least 2 distinct groups, found {groups.nunique()}.")
    if stratify is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_target = stratify
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_target = np.zeros(len(groups))
    labels = np.full(len(groups), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(groups)), split_target, groups=groups)):
        labels[test_idx] = fold
    assert (labels >= 0).all()
    return pd.Series(labels, index=groups.index)


def _stratified_folds(stratify: pd.Series, n_splits: int, random_state: int) -> pd.Series:
    """Fold labels stratified on an explicit column (plain IID otherwise handled by AutoGluon)."""
    from sklearn.model_selection import StratifiedKFold

    n_splits = min(n_splits, int(stratify.value_counts().min()))
    if n_splits < 2:
        raise ValueError(f"`stratify_on` minority value occurs fewer than 2 times; cannot build stratified folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = np.full(len(stratify), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(stratify)), stratify)):
        labels[test_idx] = fold
    assert (labels >= 0).all()
    return pd.Series(labels, index=stratify.index)
