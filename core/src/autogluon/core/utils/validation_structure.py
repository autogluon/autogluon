from __future__ import annotations

import logging
from dataclasses import dataclass, fields

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
    group_time_on: str | None = None

    #: At or below this many group instances (distinct groups, or rows when ungrouped) the
    #: tiny-data regime applies: more folds and repeats, since each fold is small and noisy.
    max_instances_for_tiny_data: int = 500
    tiny_data_num_folds: int = 5
    tiny_data_num_repeats: int = 5
    default_num_folds: int = 8
    default_num_repeats: int = 1

    def __post_init__(self):
        if self.group_on is not None and self.time_on is not None:
            raise NotImplementedError(
                "Specifying both `group_on` and `time_on` is not supported. To express data that is "
                "both grouped and temporal, use `group_time_on` (groups ordered by time)."
            )
        if self.group_time_on is not None and (self.group_on is not None or self.time_on is not None):
            raise ValueError("`group_time_on` is mutually exclusive with `group_on` / `time_on`.")
        if all(v is None for v in (self.group_on, self.time_on, self.stratify_on, self.group_time_on)):
            raise ValueError(
                "ValidationStructure requires at least one of `group_on`, `time_on`, `stratify_on`, `group_time_on`."
            )

    @classmethod
    def from_input(cls, value: ValidationStructure | dict | None) -> ValidationStructure | None:
        if value is None or isinstance(value, ValidationStructure):
            return value
        if isinstance(value, dict):
            valid = {f.name for f in fields(cls)}
            invalid = set(value) - valid
            if invalid:
                raise ValueError(
                    f"Invalid `validation_structure` keys: {sorted(invalid)}. Valid keys: {sorted(valid)}"
                )
            return cls(**value)
        raise ValueError(f"`validation_structure` must be a dict or ValidationStructure, got: {type(value)}")

    # ── split-count policy ───────────────────────────────────────────────────────

    def resolve_num_splits(
        self, X: pd.DataFrame, num_folds: int | None = None, num_repeats: int | None = None
    ) -> tuple[int, int]:
        """Fold/repeat counts to use for this data, derived when not given.

        Counts *group instances* rather than rows (a grouped task's effective sample size is
        its number of groups): at or below :attr:`max_instances_for_tiny_data` the tiny-data
        regime applies, otherwise the defaults. Explicit user values always win.
        """
        n_instances = self._num_group_instances(X)
        if num_folds is not None and num_repeats is not None:
            return num_folds, num_repeats
        if n_instances <= self.max_instances_for_tiny_data:
            folds, repeats = self.tiny_data_num_folds, self.tiny_data_num_repeats
            logger.log(
                20,
                f"validation_structure: tiny data ({n_instances} <= {self.max_instances_for_tiny_data} "
                f"instances): using num_bag_folds={folds}, num_bag_sets={repeats}.",
            )
        else:
            folds, repeats = self.default_num_folds, self.default_num_repeats
        return (num_folds if num_folds is not None else folds, num_repeats if num_repeats is not None else repeats)

    def _num_group_instances(self, X: pd.DataFrame) -> int:
        """Effective sample size: distinct groups when grouped, else row count."""
        if self.group_on is not None:
            return int(self._group_values(X).nunique())
        if self.group_time_on is not None:
            return int(self._group_time_values(X).nunique())
        return len(X)

    # ── explicit splits (the bagged path) ────────────────────────────────────────

    def custom_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None = None,
        num_repeats: int | None = None,
        random_state: int = 0,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
        """Explicit ``(train_idx, val_idx)`` splits honoring the declared structure.

        Returns ``(splits, num_folds, num_repeats)`` with ``len(splits) == num_folds *
        num_repeats``; the returned counts may be clamped below what was requested (see
        below), so callers must adopt them rather than assume their originals held.

        Explicit splits (rather than per-row group labels routed through the ``groups``
        channel) are what make repeated grouped cross-validation possible: the ``groups``
        channel forces leave-one-group-out with ``n_repeats == 1``.

        Clamping rules, each of which also forces ``num_repeats = 1`` because the resulting
        partition is deterministic and repeating it would only duplicate work:

        * temporal (``time_on`` / ``group_time_on``): blocks are contiguous in time, so
          there is exactly one valid partition;
        * fewer groups than folds: folds drop to the group count;
        * a stratification value rarer than the fold count: folds drop to that count.
        """
        num_folds, num_repeats = self.resolve_num_splits(X, num_folds, num_repeats)
        num_folds = max(2, num_folds)
        num_repeats = max(1, num_repeats)

        if self.time_on is not None or self.group_time_on is not None:
            labels = self._temporal_labels(X, n_blocks=num_folds)
            splits = _splits_from_labels(labels)
            return splits, len(splits), 1

        if self.group_on is not None:
            groups = self._group_values(X)
            n_groups = int(groups.nunique())
            if n_groups < num_folds:
                logger.log(
                    20,
                    f"validation_structure: {n_groups} groups < num_folds ({num_folds}); "
                    f"setting num_folds={n_groups} and num_repeats=1.",
                )
                num_folds, num_repeats = n_groups, 1
            stratify = self._resolve_stratify_for_folds(X, y)
            if stratify is not None:
                minority = int(stratify.value_counts().min())
                if minority < num_folds:
                    logger.log(
                        20,
                        f"validation_structure: rarest stratification value occurs {minority} times "
                        f"< num_folds ({num_folds}); setting num_folds={minority} and num_repeats=1.",
                    )
                    num_folds, num_repeats = max(2, minority), 1

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for repeat in range(num_repeats):
            labels = self.fold_ids(X, y, n_splits=num_folds, random_state=random_state + repeat)
            repeat_splits = _splits_from_labels(labels)
            if len(repeat_splits) != num_folds:
                # a splitter produced fewer folds than asked; adopt what the data supports
                num_folds = len(repeat_splits)
                splits, num_repeats = [], 1
                splits.extend(repeat_splits)
                break
            splits.extend(repeat_splits)
        _validate_splits(splits, n_rows=len(X), stratify=self._stratify_values(X, y))
        return splits, num_folds, num_repeats

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

    def _group_time_values(self, X: pd.DataFrame) -> pd.Series:
        """Group ids for ``group_time_on`` (groups are the unit ordered in time)."""
        if self.group_time_on not in X.columns:
            raise KeyError(f"`group_time_on` column {self.group_time_on!r} not found in the training data.")
        values = X[self.group_time_on]
        if values.isna().any():
            raise ValueError(f"`group_time_on` column {self.group_time_on!r} contains NaN values.")
        return values

    def _temporal_labels(self, X: pd.DataFrame, n_blocks: int) -> pd.Series:
        """Contiguous time-block labels; ``group_time_on`` blocks whole groups in time order.

        ``time_on`` blocks rows by their time value (ties never split). ``group_time_on``
        treats each group as an indivisible unit ordered by its first appearance, so a block
        boundary never cuts through a group: the result is both group-disjoint and forward
        in time.
        """
        if self.group_time_on is not None:
            groups = self._group_time_values(X)
            # order groups by first appearance, then block the ordered group sequence
            order = {g: i for i, g in enumerate(groups.drop_duplicates())}
            return _time_blocks(groups.map(order), n_blocks=n_blocks)
        return _time_blocks(self._time_values(X), n_blocks=n_blocks)

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
        if self.time_on is not None or self.group_time_on is not None:
            labels = self._temporal_labels(X, n_blocks=n_splits)
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
        if self.time_on is None and self.group_on is None and self.group_time_on is None:
            return None
        if self.group_time_on is not None:
            # whole groups, ordered by first appearance: the latest groups form the holdout
            labels = self._temporal_labels(X, n_blocks=max(2, int(round(1 / max(holdout_frac, 1e-9))))).to_numpy()
            last = labels.max()
            val_idx = np.flatnonzero(labels == last)
            train_idx = np.flatnonzero(labels != last)
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(f"`group_time_on` column {self.group_time_on!r} cannot produce a forward holdout.")
            return train_idx, val_idx
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


def _splits_from_labels(labels: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-label-out positional ``(train_idx, val_idx)`` splits, in label order."""
    values = np.asarray(labels)
    splits = []
    for label in sorted(pd.unique(values)):
        val_idx = np.flatnonzero(values == label)
        train_idx = np.flatnonzero(values != label)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def _validate_splits(splits: list[tuple[np.ndarray, np.ndarray]], n_rows: int, stratify: pd.Series | None) -> None:
    """Assert the splits are usable: non-empty, positional, and stratification-preserving."""
    if not splits:
        raise ValueError("validation_structure produced no usable validation splits.")
    for train_idx, val_idx in splits:
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("validation_structure produced an empty train or validation split.")
        if train_idx.max(initial=-1) >= n_rows or val_idx.max(initial=-1) >= n_rows:
            raise ValueError("validation_structure produced out-of-range split indices.")
    if stratify is not None:
        expected = set(pd.unique(stratify.astype(str)))
        for _, val_idx in splits:
            present = set(pd.unique(stratify.astype(str).to_numpy()[val_idx]))
            if not present:
                raise ValueError("validation_structure produced a validation split with no rows.")
            missing = expected - present
            if missing and len(expected) == 2:
                # binary stratification with an absent class makes the fold unscorable
                logger.warning(
                    f"validation_structure: a validation fold is missing stratification value(s) {sorted(missing)}."
                )


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
        raise ValueError("`stratify_on` minority value occurs fewer than 2 times; cannot build stratified folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = np.full(len(stratify), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(stratify)), stratify)):
        labels[test_idx] = fold
    assert (labels >= 0).all()
    return pd.Series(labels, index=stratify.index)
