from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from autogluon.common.constants import BINARY, MULTICLASS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationStructure:
    """Declarative description of a dataset's validation-relevant structure.

    Users specify only the semantic columns; fold counts, interval blocking, and clamping
    are derived. The learner resolves the bagged splits (:meth:`custom_splits`) and the
    trainer the non-bagged holdout (:meth:`holdout_split_indices`), both on the raw cleaned
    frame, before feature transformation can alter the columns named here.

    Parameters
    ----------
    group_on : str | list[str], optional
        Column(s) identifying groups that must not span both a training and a
        validation split (e.g. a customer or session id).
    time_on : str, optional
        Column encoding time (datetime or numeric). Validation splits become
        contiguous time blocks; the non-bagged holdout is the latest block.
    stratify_on : str, optional
        Column to stratify splits on (may name the label rather than a feature). Defaults
        to the label for classification when combined with ``group_on``; for time-based
        splits it only influences how the contiguous blocks are assigned to fold indices,
        never the block boundaries themselves.
    size_validation_on_groups : bool, default False
        If True, the automatically selected validation method (fold count, repeats, extra
        holdout, stack depth) is chosen from the number of groups rather than the number of
        rows. Rows within a group are not independent, so the group count is often the sample
        size that matters -- a task with 4,672 rows across 68 groups is large by rows and small
        by groups -- but which applies is a judgement about the data, so this is opt-in. Has no
        effect unless ``group_on`` or ``group_time_on`` is set.
    group_time_on : str, optional
        Column identifying groups that are *also* ordered in time (e.g. a session id whose
        sessions arrive in sequence). Whole groups are blocked in time order, so the splits
        are simultaneously group-disjoint and forward in time, and the non-bagged holdout is
        the latest groups. Use this for data that is both grouped and temporal; combining
        ``group_on`` with ``time_on`` is not supported, as their semantics would be ambiguous.
    temporal_forward_only : bool, default False
        Restrict temporal validation to forward-chaining (an expanding window): fold *i*
        validates time block *i+1* and trains only on the blocks before it, so a model is never
        trained on data from after the window it is scored on.

        The default (False) is leave-one-block-out: every block is validated exactly once and
        trained on for every other fold, which covers all rows but trains most folds partly on
        the future. Forward-chaining removes that lookahead at two costs:

        1. **The earliest block is never validated.** It is training data only, so those rows
           get no out-of-fold prediction. Callers that consume out-of-fold predictions must
           handle the gap -- see :meth:`uncovered_rows`.
        2. **Folds train on less data.** Fold 0 sees one block; only the last sees nearly all.

        Requires at least 3 blocks (2 folds plus the initial training block), and has no effect
        unless ``time_on`` or ``group_time_on`` is set.
    """

    group_on: str | list[str] | None = None
    time_on: str | None = None
    stratify_on: str | None = None
    group_time_on: str | None = None
    size_validation_on_groups: bool = False
    temporal_forward_only: bool = False

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
        if self.temporal_forward_only and self.time_on is None and self.group_time_on is None:
            raise ValueError(
                "`temporal_forward_only` requires `time_on` or `group_time_on`; there is no time order to chain along."
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

    def _can_split_group_table(self, X: pd.DataFrame, y: pd.Series, problem_type: str | None = None) -> bool:
        """Whether the folds can be built from the group table rather than from the samples.

        True only when a stratification signal exists *and* is constant within every group:
        the group table holds one row per group, so it can only carry one stratification
        value per group. Splitting it then balances the stratification across folds without
        letting group sizes skew it.

        This is therefore always False without a stratification signal -- notably for
        regression, which does not stratify -- and such tasks fall through to the
        sample-level ``GroupKFold`` / ``StratifiedGroupKFold`` path. That matches the
        grouped-validation protocol this mirrors, whose group-table branch likewise requires
        a stratification column.
        """
        stratify = self._resolve_stratify_for_folds(X, y, problem_type)
        if stratify is None:
            return False
        groups = self._group_values(X)
        return bool(pd.Series(np.asarray(stratify)).groupby(np.asarray(groups)).nunique().max() == 1)

    def rare_stratification_values(
        self, X: pd.DataFrame, y: pd.Series, min_units: int = 2, problem_type: str | None = None
    ) -> dict[Any, int]:
        """Stratification values spanning fewer than ``min_units`` independent units.

        The independent unit is the group where grouping is declared, and the row otherwise.
        The distinction matters because the guarantee callers want from "at least two members"
        is that *every* training split retains one of them, and only two members in two
        different folds deliver it. Two rows inside one group are one unit: they always land in
        the same fold, so the fold validating them trains on none of that value. Duplicating
        rows -- what ``augment_rare_classes`` does -- cannot fix that, which is why the count
        has to be taken in units rather than rows.

        Returns ``{value: n_units}`` for the offending values only, empty when there is no
        stratification signal.
        """
        stratify = self._resolve_stratify_for_folds(X, y, problem_type)
        if stratify is None:
            return {}
        values = np.asarray(stratify)
        units = np.asarray(self._group_values(X)) if self.group_on is not None else np.arange(len(values))
        units_per_value = pd.Series(units).groupby(values, observed=True).nunique()
        return {value: int(count) for value, count in units_per_value.items() if count < min_units}

    def max_scorable_folds(self, X: pd.DataFrame, y: pd.Series, problem_type: str | None = None) -> int | None:
        """Fold ceiling that keeps every validation fold scorable, or None when unbounded.

        AutoGluon scores every bagged child on its own validation fold
        (``fold_model.val_score = fold_model.score_with_y_pred_proba(...)``), and a binary metric
        such as ``roc_auc`` is *undefined* on a fold holding one class. For binary problems a fold
        that misses a class therefore fails the fit rather than degrading it, which makes the
        ceiling load-bearing: it is the number of independent units — groups where grouping is
        declared, rows otherwise — holding the rarer class, since past that some fold necessarily
        receives none of it.

        Multiclass is unbounded. ``log_loss`` is defined with a class absent from a fold, so there
        the cost is a narrower per-fold estimate rather than a failure.
        """
        if problem_type != BINARY or y is None:
            return None
        values = np.asarray(y)
        units = np.asarray(self._group_values(X)) if self.group_on is not None else np.arange(len(values))
        units_per_class = pd.Series(units).groupby(values, observed=True).nunique()
        return int(units_per_class.min()) if len(units_per_class) else None

    def num_group_instances(self, X: pd.DataFrame) -> int | None:
        """Number of distinct groups, or None when this structure declares no grouping.

        The independent-unit count for grouped data, which callers may use as the sample size
        for size-dependent decisions (see :attr:`size_validation_on_groups`).
        """
        if self.group_on is None and self.group_time_on is None:
            return None
        return self._num_group_instances(X)

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
        problem_type: str | None = None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
        """Explicit ``(train_idx, val_idx)`` splits honoring the declared structure.

        Returns ``(splits, num_folds, num_repeats)`` with ``len(splits) == num_folds *
        num_repeats``; the returned counts may be clamped below what was requested (see
        below), so callers must adopt them rather than assume their originals held. The
        requested counts are otherwise honored as given -- deriving them from data size is
        the caller's decision, not this class's.

        Explicit splits (rather than per-row group labels routed through the ``groups``
        channel) are what make repeated grouped cross-validation possible: the ``groups``
        channel forces leave-one-group-out with ``n_repeats == 1``.

        These clamps are the *feasibility* half of choosing a validation method: the caller's
        requested counts are the policy (see AutoGluon's validation size curves), and this
        reduces them only where the data cannot support what was asked. It never raises them,
        so a caller may treat the request as an upper bound.

        Clamping rules, each of which also forces ``num_repeats = 1`` because the resulting
        partition is deterministic and repeating it would only duplicate work:

        * temporal (``time_on`` / ``group_time_on``): blocks are contiguous in time, so
          there is exactly one valid partition;
        * fewer groups than folds: folds drop to the group count;
        * a stratification value rarer than the fold count: folds drop to that count.
        """
        num_folds = max(2, num_folds if num_folds is not None else 8)
        num_repeats = max(1, num_repeats if num_repeats is not None else 1)

        if self.time_on is not None or self.group_time_on is not None:
            if self.temporal_forward_only:
                # One more block than folds: the earliest is training data for every fold and is
                # itself never validated.
                labels = self._temporal_labels(X, n_blocks=num_folds + 1)
                splits = _forward_chaining_splits(labels)
                if len(splits) < 2:
                    # One fold is a single forward holdout, not cross-validation; bagging needs >= 2.
                    raise ValueError(
                        f"`temporal_forward_only` needs at least 3 time blocks to form 2 folds, but "
                        f"{self.time_on or self.group_time_on!r} supports only {labels.nunique()} "
                        f"(yielding {len(splits)} fold(s)). Use the default temporal splits, or a "
                        f"non-bagged holdout, for data this coarse in time."
                    )
                uncovered = len(X) - sum(len(val_idx) for _, val_idx in splits)
                logger.log(
                    20,
                    f"validation_structure: forward-chaining over {len(splits) + 1} time blocks -> "
                    f"{len(splits)} folds; the earliest block ({uncovered} rows) is never validated, so "
                    f"those rows get no out-of-fold prediction.",
                )
                return splits, len(splits), 1
            labels = self._temporal_labels(X, n_blocks=num_folds)
            # Route the block labels through GroupKFold rather than emitting them in
            # chronological order: the partition set is the same either way, but this
            # matches the fold ordering the benchmark protocol produces.
            splits = _splits_from_labels(
                _group_folds(
                    groups=labels,
                    stratify=self._stratify_values(X, y),
                    n_splits=int(labels.nunique()),
                    random_state=random_state,
                )
            )
            if num_repeats > 1:
                logger.log(
                    20,
                    f"validation_structure: a temporal partition is deterministic, so repeats add "
                    f"nothing; num_repeats reduced from {num_repeats} to 1.",
                )
            if len(splits) != num_folds:
                logger.log(
                    20,
                    f"validation_structure: {self.time_on or self.group_time_on!r} supports "
                    f"{len(splits)} time blocks (requested {num_folds} folds).",
                )
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
            # Binary only: a fold whose validation rows hold one class cannot be scored at all, so
            # the fold count has to respect how many groups carry the rarer class (see
            # `max_scorable_folds`). Repeats are left alone -- fewer folds does not make a repeat
            # any less valid, and dropping them was never what kept the fit working.
            ceiling = self.max_scorable_folds(X, y, problem_type)
            if ceiling is not None and ceiling < num_folds:
                logger.log(
                    20,
                    f"validation_structure: the rarer class spans {ceiling} group(s) < num_folds "
                    f"({num_folds}); setting num_folds={max(2, ceiling)} so every fold stays scorable.",
                )
                num_folds = max(2, ceiling)
            # Multiclass keeps the requested fold count: log_loss tolerates a class missing from a
            # fold, so the cost is a narrower per-fold estimate, not a failed fit.
            rare = self.rare_stratification_values(X, y, min_units=num_folds, problem_type=problem_type)
            if rare:
                logger.log(
                    20,
                    f"validation_structure: stratification value(s) {sorted(map(str, rare))} span fewer "
                    f"than num_folds ({num_folds}) groups, so some folds validate on none of them; "
                    f"keeping num_folds={num_folds}.",
                )

        if self.group_on is not None and self._can_split_group_table(X, y, problem_type):
            # Every group carries one stratification value: split the *group table* (one row
            # per group) and expand back to rows, so group sizes do not skew stratification.
            splits = _per_group_splits(
                groups=self._group_values(X),
                stratify=self._resolve_stratify_for_folds(X, y),
                n_splits=num_folds,
                n_repeats=num_repeats,
                random_state=random_state,
            )
            _validate_splits(splits, n_rows=len(X), stratify=self._stratify_values(X, y))
            return splits, num_folds, num_repeats

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for repeat in range(num_repeats):
            labels = self.fold_ids(
                X, y, n_splits=num_folds, random_state=random_state + repeat, problem_type=problem_type
            )
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

    def uncovered_rows(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:
        """Positional indices of rows no fold validates, i.e. rows that get no out-of-fold prediction.

        Empty for every configuration except ``temporal_forward_only``, whose earliest time block
        is training data only. Callers that consume out-of-fold predictions should exclude these
        rows: a bagged model leaves them at the accumulator's initial value rather than marking
        them missing, so downstream they read as a prediction of 0 rather than as absent.

        ``**kwargs`` are forwarded to :meth:`custom_splits` (``num_folds``, ``random_state``, ...)
        so the answer matches the splits that call produces.
        """
        splits, _, _ = self.custom_splits(X, y, **kwargs)
        covered = np.concatenate([val_idx for _, val_idx in splits]) if splits else np.array([], dtype=int)
        return np.setdiff1d(np.arange(len(X)), covered)

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
        if y is not None and getattr(y, "name", None) == self.stratify_on:
            # stratifying on the label itself: it is not a feature column
            return y.astype("category")
        raise KeyError(f"`stratify_on` column {self.stratify_on!r} is neither a feature column nor the label.")

    # ── bagged path ──────────────────────────────────────────────────────────────

    def fold_ids(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int = 0, problem_type: str | None = None
    ) -> pd.Series:
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
                stratify=self._resolve_stratify_for_folds(X, y, problem_type),
                n_splits=n_splits,
                random_state=random_state,
            )
        else:
            # Binary needs both classes in every fold to be scorable at all; here the independent
            # unit is the row, so this is the rarer class's row count.
            ceiling = self.max_scorable_folds(X, y, problem_type)
            if ceiling is not None and ceiling < n_splits:
                logger.log(
                    20,
                    f"validation_structure: the rarer class has {ceiling} row(s) < n_splits "
                    f"({n_splits}); setting n_splits={max(2, ceiling)} so every fold stays scorable.",
                )
                n_splits = max(2, ceiling)
            labels = _stratified_folds(self._stratify_values(X, y), n_splits=n_splits, random_state=random_state)
        n_folds = labels.nunique()
        if n_folds < n_splits:
            logger.log(
                20,
                f"validation_structure: data supports only {n_folds} folds (requested {n_splits}).",
            )
        return pd.Series(labels, index=X.index, name="__fold_id__")

    def _resolve_stratify_for_folds(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str | None = None
    ) -> pd.Series | None:
        """The stratification signal for fold construction, or None to not stratify.

        With no explicit ``stratify_on``, the label is used for classification, mirroring
        AutoGluon's default label-stratified splitting so that declaring ``group_on`` does
        not silently cost the stratification a user gets for free. This needs the caller's
        ``problem_type``: guessing it from the label's cardinality misjudges both directions
        -- high-cardinality multiclass looks continuous, and a coarse numeric target looks
        categorical -- and the callers all know the answer already.
        """
        stratify = self._stratify_values(X, y)
        if stratify is None and problem_type in (BINARY, MULTICLASS):
            stratify = y
        return stratify

    # ── holdout path ─────────────────────────────────────────────────────────────

    def holdout_split_indices(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        holdout_frac: float,
        random_state: int = 0,
        problem_type: str | None = None,
        min_cls_count_train: int = 1,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A single structure-aware ``(train_idx, val_idx)`` positional split.

        Group structure yields a group-disjoint split; time structure yields a
        *forward* holdout (the latest contiguous block — a random fold would leak
        future information into training). Returns None when only ``stratify_on`` is
        set: AutoGluon's default label-stratified holdout needs no correction.

        ``min_cls_count_train`` is the per-class row count the training side must keep, the
        same guarantee ``generate_train_test_split`` enforces on the unstructured path. It is
        honoured by moving whole groups across the boundary, never single rows, so the split
        stays group-disjoint; a forward holdout cannot be repaired that way and is reported
        instead (moving validation rows into training would move the time boundary).
        """
        if self.time_on is None and self.group_on is None and self.group_time_on is None:
            return None
        if self.time_on is not None or self.group_time_on is not None:
            # Forward holdout: the latest block of the *same* blocking the bagged path uses,
            # so cut placement and tie handling are defined in exactly one place and the
            # holdout boundary always coincides with a bagged fold boundary. The block count
            # follows from holdout_frac; the realised size is the nearest block boundary to
            # it, which tracks the request more closely than cutting at the fraction and
            # walking the boundary back over ties (that can only move one way, so a large
            # tie block at the boundary overshoots badly).
            n_blocks = max(2, int(round(1 / max(holdout_frac, 1e-9))))
            labels = self._temporal_labels(X, n_blocks=n_blocks).to_numpy()
            val_idx = np.flatnonzero(labels == labels.max())
            train_idx = np.flatnonzero(labels != labels.max())
            if len(train_idx) == 0 or len(val_idx) == 0:
                column = self.time_on if self.time_on is not None else self.group_time_on
                raise ValueError(f"column {column!r} cannot produce a non-empty forward holdout.")
            self._warn_untrained_values(X, y, train_idx, problem_type, repairable=False)
            return train_idx, val_idx

        # Grouped holdout: one fold of the same split the bagged path would build, so the
        # holdout inherits the group-disjointness *and* the stratification handling rather
        # than re-deriving them (GroupShuffleSplit cannot stratify at all). The fold count
        # is chosen so a single fold is ~holdout_frac of the data; with coarse grouping the
        # realised size can differ, because whole groups cannot be subdivided.
        n_splits = max(2, int(round(1 / max(holdout_frac, 1e-9))))
        splits, _, _ = self.custom_splits(
            X, y, num_folds=n_splits, num_repeats=1, random_state=random_state, problem_type=problem_type
        )
        train_idx, val_idx = splits[0]
        return self._repair_grouped_holdout(X, y, train_idx, val_idx, min_cls_count_train, problem_type)

    def _repair_grouped_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        min_cls_count_train: int,
        problem_type: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move whole groups from validation to training until every value clears the floor.

        Whole groups, because moving individual rows would place one group on both sides of the
        split and undo the group-disjointness the structure exists to provide. The smallest
        qualifying group moves first, so the validation side gives up as little as possible.
        """
        stratify = self._resolve_stratify_for_folds(X, y, problem_type)
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        if stratify is None or min_cls_count_train < 1:
            return train_idx, val_idx

        values, groups = np.asarray(stratify), np.asarray(self._group_values(X))
        moved: list[Any] = []
        train, val = train_idx, val_idx
        for value in pd.unique(values):
            while int((values[train] == value).sum()) < min_cls_count_train:
                on_val = groups[val][values[val] == value]
                if len(on_val) == 0:
                    break  # no group left to take it from; reported below
                sizes = pd.Series(groups[val]).value_counts()
                group = min(pd.unique(on_val), key=lambda g: (int(sizes.get(g, 0)), str(g)))
                whole_group = np.flatnonzero(groups == group)
                train = np.union1d(train, whole_group)
                val = np.setdiff1d(val, whole_group)
                moved.append(group)

        if moved and len(val) == 0:
            # Coarse grouping can make the repair cost the entire holdout. The original split is
            # imperfect but usable, so keep it: an unlearnable rare value degrades the fit, while
            # an empty holdout has nothing to validate on at all.
            logger.warning(
                f"validation_structure: keeping every stratification value in training would consume "
                f"the whole {len(val_idx)}-row holdout, so the split is left as it is."
            )
            self._warn_untrained_values(X, y, train_idx, problem_type, repairable=True)
            return train_idx, val_idx
        if moved:
            logger.log(
                20,
                f"validation_structure: moved {len(moved)} whole group(s) into training so every "
                f"stratification value keeps at least {min_cls_count_train} training row(s); "
                f"holdout is now {len(val)} rows (was {len(val_idx)}).",
            )
        self._warn_untrained_values(X, y, train, problem_type, repairable=True)
        return train, val

    def _warn_untrained_values(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        problem_type: str | None,
        repairable: bool,
    ) -> None:
        """Report stratification values the training side of a holdout cannot learn."""
        stratify = self._resolve_stratify_for_folds(X, y, problem_type)
        if stratify is None:
            return
        values = np.asarray(stratify)
        missing = sorted(set(pd.unique(values)) - set(pd.unique(values[np.asarray(train_idx)])), key=str)
        if missing:
            reason = (
                "too few rows of it exist to keep any in training"
                if repairable
                else "a forward holdout cannot borrow from the future to correct this"
            )
            logger.warning(
                f"validation_structure: stratification value(s) {[str(v) for v in missing]} are absent "
                f"from the training split, so nothing can be learned about them ({reason})."
            )


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


def _forward_chaining_splits(labels: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window positional splits: fold *i* validates block *i+1*, trains on blocks <= *i*.

    With ``n`` blocks this yields ``n - 1`` folds. The earliest block appears only in training
    sets, so it is never validated -- unlike leave-one-block-out, the returned splits do not
    cover every row. No fold ever trains on a row later than its validation window.
    """
    values = np.asarray(labels)
    ordered = sorted(pd.unique(values))
    splits = []
    for i, label in enumerate(ordered[1:], start=1):
        val_idx = np.flatnonzero(values == label)
        train_idx = np.flatnonzero(np.isin(values, ordered[:i]))
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
        stratify_str = stratify.astype(str).to_numpy()
        expected = set(pd.unique(stratify_str))
        untrainable: set[str] = set()
        n_untrainable_folds = 0
        n_unscorable_folds = 0
        for train_idx, val_idx in splits:
            if not len(val_idx):
                raise ValueError("validation_structure produced a validation split with no rows.")
            missing_from_train = expected - set(pd.unique(stratify_str[train_idx]))
            if missing_from_train:
                untrainable |= missing_from_train
                n_untrainable_folds += 1
            if expected - set(pd.unique(stratify_str[val_idx])):
                n_unscorable_folds += 1
        if untrainable:
            # The serious direction: such a fold's model never sees the value at all.
            logger.warning(
                f"validation_structure: {n_untrainable_folds} of {len(splits)} fold(s) train on none of "
                f"stratification value(s) {sorted(untrainable)}, so those folds cannot learn them."
            )
        if n_unscorable_folds:
            # Checked for every problem type, not just binary: a multiclass fold missing a value
            # scores on an incomplete set just as a binary one does, and metrics such as roc_auc
            # are undefined outright when a fold holds a single value.
            logger.warning(
                f"validation_structure: {n_unscorable_folds} of {len(splits)} validation fold(s) are "
                f"missing at least one stratification value, so their per-fold scores cover fewer "
                f"values than the full data. The out-of-fold predictions still cover every row."
            )


def _split_target(values: np.ndarray | pd.Series) -> pd.Series:
    """A stratification target :class:`CVSplitter` can also apply its rare-class workaround to.

    That workaround appends a sentinel class, which cannot be ordered against string labels,
    so non-numeric values are encoded to integer codes in order of first appearance. sklearn
    derives that same encoding internally (``np.unique(y_idx, return_inverse=True)`` ranks
    classes by first occurrence), so the folds are identical either way.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number):
        values = pd.factorize(values, sort=False)[0]
    return pd.Series(values)


def _per_group_splits(
    groups: pd.Series, stratify: pd.Series | None, n_splits: int, n_repeats: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated (stratified) K-fold over the group table, expanded to row indices.

    Groups are numbered by first appearance (``pd.factorize``) so the group table's row
    order — and therefore the splitter's output — does not depend on group sizes or values.

    Splitting goes through :class:`CVSplitter` rather than sklearn directly. sklearn's
    ``StratifiedKFold`` refuses to split at all when *every* class holds fewer than
    ``n_splits`` members, which the group table hits whenever the groups-per-class count
    falls below the fold count — while the row counts behind those groups are ample. That
    refusal is not a real constraint: two groups of a class in two different folds already
    leave every training split with a member of it. ``CVSplitter`` works around the sklearn
    behaviour and otherwise builds the same ``Repeated{,Stratified}KFold``, so the splits are
    unchanged wherever sklearn succeeds today.
    """
    from .cv_splitter import CVSplitter

    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)

    group_target = None
    if stratify is not None:
        # each group carries one stratification value: read it off the group's first row
        _, first_row_of_group = np.unique(codes, return_index=True)
        group_target = np.asarray(stratify)[first_row_of_group]

    splitter = CVSplitter(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        stratify=group_target is not None,
    )
    target = _split_target(group_target) if group_target is not None else pd.Series(np.zeros(n_groups))

    splits = []
    for _, val_groups in splitter.split(X=None, y=target):
        # expand group membership to rows with one boolean lookup instead of per-group scans
        is_val_group = np.zeros(n_groups, dtype=bool)
        is_val_group[np.asarray(val_groups, dtype=int)] = True
        val_mask = is_val_group[codes]
        splits.append((np.flatnonzero(~val_mask), np.flatnonzero(val_mask)))
    return splits


def _time_blocks(values: pd.Series, n_blocks: int) -> pd.Series:
    """Contiguous, tie-preserving time blocks of near-equal row counts (labels 0..k-1).

    Splits the *observed* time values (not equal-width spans) into contiguous intervals:
    identical time values always land in the same block, larger values are always later,
    and each cut is placed where the cumulative row count is closest to that block's share
    of the total. Fewer blocks than requested are produced when there are too few distinct
    time values to support them.
    """
    counts = values.value_counts(dropna=False).sort_index()
    n_unique = len(counts)
    if n_unique < 2:
        raise ValueError("`time_on` needs at least 2 distinct time values to form validation blocks.")
    n_blocks = min(n_blocks, n_unique)

    weights = counts.to_numpy()
    cumulative = np.cumsum(weights)
    total = int(weights.sum())

    # Greedy contiguous partition of the sorted unique values: each cut goes where the
    # cumulative weight lands closest to the target, bounded so the remaining blocks
    # still have at least one unique value each.
    cut_positions: list[int] = []
    start = 0
    for block in range(1, n_blocks):
        target = block * total / n_blocks
        max_j = n_unique - (n_blocks - block) - 1
        candidates = np.arange(start, max_j + 1)
        j = int(candidates[np.argmin(np.abs(cumulative[candidates] - target))])
        cut_positions.append(j)
        start = j + 1

    labels_for_unique = np.empty(n_unique, dtype=int)
    prev = 0
    for block, cut in enumerate(cut_positions):
        labels_for_unique[prev : cut + 1] = block
        prev = cut + 1
    labels_for_unique[prev:] = len(cut_positions)

    return values.map(pd.Series(labels_for_unique, index=counts.index))


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
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_target = np.zeros(len(groups))
    labels = np.full(len(groups), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(groups)), split_target, groups=groups)):
        labels[test_idx] = fold
    assert (labels >= 0).all()
    return pd.Series(labels, index=groups.index)


def _stratified_folds(stratify: pd.Series, n_splits: int, random_state: int) -> pd.Series:
    """Fold labels stratified on an explicit column (plain IID otherwise handled by AutoGluon).

    The fold count is bounded by the row count, *not* by the rarest value's count. k-fold
    needs only two members of a value, in two different folds, for every training split to
    retain one of them, so a value occurring twice supports any number of folds. sklearn
    refuses to split when every value is rarer than ``n_splits``, which :class:`CVSplitter`
    works around.

    A value occurring exactly once is the case this cannot fix: the fold that validates it
    trains on none of it. That is what AutoGluon's ``augment_rare_classes`` duplicates
    upstream for metrics that need every class, so it is rejected here rather than silently
    yielding a fold that cannot learn the value.
    """
    from sklearn.model_selection import StratifiedKFold

    from .cv_splitter import CVSplitter

    # Counted on the values rather than the Series: a categorical dtype reports its unused
    # categories as zero-count entries, which would read as an absent value.
    minority = int(pd.Series(np.asarray(stratify)).value_counts().min())
    if minority < 2:
        raise ValueError("`stratify_on` minority value occurs fewer than 2 times; cannot build stratified folds.")
    n_splits = min(n_splits, len(stratify))
    if n_splits < 2:
        raise ValueError(f"`stratify_on` needs at least 2 rows to build stratified folds, found {len(stratify)}.")
    splitter = CVSplitter(
        splitter_cls=StratifiedKFold,
        n_splits=n_splits,
        n_repeats=1,
        random_state=random_state,
        stratify=True,
        shuffle=True,
    )
    labels = np.full(len(stratify), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(X=None, y=_split_target(stratify))):
        labels[test_idx] = fold
    assert (labels >= 0).all()
    return pd.Series(labels, index=stratify.index)
