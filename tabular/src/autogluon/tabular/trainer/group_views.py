"""Group views: two views of every base model on data whose label is constant within each group.

When the rows of a group share one label (for example several measurements of the same patient, each row
labelled with the patient's diagnosis), a row is one noisy observation of a group-level target. Two views
of each base model use that:

- the model's row predictions pooled within each group, so every row of a group gets the group's prediction
  (`GroupViews.pooling`: by default the geometric mean of the class probabilities, renormalised, and the mean
  for regression; several strategies give one pooled view each);
- the same model fit on the collapsed table, one row per group (numeric features averaged, the others
  set to their most frequent value in the group, plus the log of the group's row count), its prediction
  for a group copied to the group's rows.

Neither view depends on the order of the rows within a group.

The weighted ensemble chooses among both views of every base model, or among one kind of view when
`GroupViews.views` names only that one (the collapsed views alone skip the fits on the rows). The group a row
belongs to travels as
`GROUP_KEY_COLUMN`, a feature of special type `S_GROUP_KEY` holding integer codes computed from the raw
`group_on` columns of the frame being processed (so unseen test groups stay distinct). Models other than the
ones defined here leave that feature out.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Callable

import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, R_INT, S_BOOL, S_GROUP_KEY, S_STACK
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel

logger = logging.getLogger(__name__)

GROUP_KEY_COLUMN = "__ag_group_key__"
"""The feature that carries each row's group, as integer codes within the frame."""

GROUP_COLLAPSED_SUFFIX = "_GroupCollapsed"
"""Appended to a base model's name for its collapsed view (e.g. `LightGBM_GroupCollapsed_BAG_L1`)."""

GROUP_LOG_ROWS_COLUMN = "__ag_group_log_rows__"
"""The collapsed table's feature holding the log of each group's row count."""

GROUP_VIEWS_PROBLEM_TYPES = (BINARY, MULTICLASS, REGRESSION)

GROUP_VIEW_KINDS = ("pooled", "collapsed")
"""The views of `GroupViews.views`: every bagged model's row predictions pooled within groups, and the model fit on
the collapsed table."""

_LOG_CLIP = 1e-15

GroupPooling = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""A custom pooling strategy, `pooling(pred, groups) -> pooled` (see `GroupViews.pooling`)."""

GROUP_POOLED_SEPARATOR = "@"
"""Joins a model's name and a pooling strategy's into the name of the model's view pooled by that strategy, when
`GroupViews.pooling` lists several (e.g. `LightGBM_BAG_L1@median`)."""


def group_key_feature_metadata() -> FeatureMetadata:
    """The metadata of `GROUP_KEY_COLUMN`, to join with the training features' metadata."""
    return FeatureMetadata(
        type_map_raw={GROUP_KEY_COLUMN: R_INT}, type_group_map_special={S_GROUP_KEY: [GROUP_KEY_COLUMN]}
    )


def group_codes(X: pd.DataFrame, group_on: list[str]) -> np.ndarray:
    """Integer codes of the groups of the rows of `X` (one code per distinct value of the `group_on` columns).

    Codes are only meaningful within `X`: two frames number their groups independently.
    """
    missing = [column for column in group_on if column not in X.columns]
    if missing:
        raise KeyError(
            f"`group_views` pools and collapses predictions within groups, so the data needs the `group_on` "
            f"column(s) {missing}."
        )
    for column in group_on:
        if X[column].isna().any():
            raise ValueError(f"`group_on` column {column!r} contains NaN values, so some rows have no group.")
    if len(group_on) == 1:
        values = X[group_on[0]]
    else:
        values = X[group_on].astype(str).agg("||".join, axis=1)
    codes, _ = pd.factorize(values, sort=False)
    return codes.astype(np.int64)


def label_constant_within_groups(y: pd.Series, codes: np.ndarray) -> bool:
    """Whether every group's rows share one label (rows with a missing label are ignored)."""
    y = pd.Series(np.asarray(y))
    known = y.notna().to_numpy()
    if not known.any():
        return False
    return bool(y[known].groupby(codes[known]).nunique().max() <= 1)


def _group_mean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Each row's value replaced by the mean over its group (`values` is 1-d or 2-d over rows)."""
    groups, inverse = np.unique(codes, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(groups)).astype(np.float64)
    if values.ndim == 1:
        return (np.bincount(inverse, weights=values, minlength=len(groups)) / counts)[inverse]
    out = np.empty_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = (np.bincount(inverse, weights=values[:, j], minlength=len(groups)) / counts)[inverse]
    return out


def _group_mode(column: pd.Series, group: np.ndarray, n_groups: int) -> pd.Series:
    """Each group's most frequent non-missing value of `column` (ties: the smallest value; missing if none)."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        codes, uniques = column.cat.codes.to_numpy().astype(np.int64), None
    else:
        codes, uniques = pd.factorize(column, sort=True)
    known = codes >= 0
    counts = pd.DataFrame({"g": group[known], "c": codes[known]}).value_counts().reset_index(name="n")
    best = counts.sort_values(["g", "n", "c"], ascending=[True, False, True]).drop_duplicates("g")
    mode = np.full(n_groups, -1, dtype=np.int64)
    mode[best["g"].to_numpy()] = best["c"].to_numpy()
    if uniques is None:
        return pd.Series(pd.Categorical.from_codes(mode, dtype=column.dtype), name=column.name)
    values = pd.Series(np.asarray(uniques).take(np.maximum(mode, 0)), name=column.name)
    if (mode < 0).any():
        values = values.where(mode >= 0)
    return values


def _pool_geometric_mean(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The geometric mean over each group's rows: the exponentiated mean of the log-predictions."""
    return np.exp(_group_mean(np.log(np.clip(pred, _LOG_CLIP, 1)), groups))


def _pool_median(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The median over each group's rows, per column."""
    return pd.DataFrame(pred).groupby(groups, sort=False).transform("median").to_numpy()


GROUP_POOLING: dict[str, GroupPooling] = {
    "geometric_mean": _pool_geometric_mean,
    "mean": _group_mean,
    "median": _pool_median,
}
"""The built-in pooling strategies of `GroupViews.pooling`, by name."""


def default_group_pooling(problem_type: str) -> str:
    """The pooling strategy `GroupViews.pooling=None` stands for."""
    return "mean" if problem_type == REGRESSION else "geometric_mean"


def group_pooling_function(pooling: str | GroupPooling | None, problem_type: str) -> GroupPooling:
    """The function of a pooling strategy (see `GroupViews.pooling`), checked against `problem_type`."""
    if pooling is None:
        pooling = default_group_pooling(problem_type)
    if callable(pooling):
        return pooling
    if pooling not in GROUP_POOLING:
        raise ValueError(
            f"`group_views` pooling must be one of {sorted(GROUP_POOLING)} or a callable, got {pooling!r}."
        )
    if pooling == "geometric_mean" and problem_type == REGRESSION:
        raise ValueError(
            "`group_views` pooling 'geometric_mean' pools class probabilities, so it does not apply to regression: "
            "use 'mean', 'median' or a callable."
        )
    return GROUP_POOLING[pooling]


def group_pooling_strategies(pooling: str | GroupPooling | Sequence[str | GroupPooling] | None) -> list:
    """`GroupViews.pooling` as a list of strategies (`[None]` for the problem type's default)."""
    if isinstance(pooling, (list, tuple)):
        return list(pooling)
    return [pooling]


def group_pooling_name(pooling: str | GroupPooling) -> str:
    """The name of a pooling strategy: a built-in's own, or the callable's `__name__`."""
    if isinstance(pooling, str):
        return pooling
    return getattr(pooling, "__name__", type(pooling).__name__)


def pool_within_groups(
    pred: np.ndarray, codes: np.ndarray, problem_type: str, pooling: str | GroupPooling | None = None
) -> np.ndarray:
    """One model's predictions pooled within each group, by `pooling` (see `GroupViews.pooling`).

    Binary class probabilities are the positive class's, as a 1-d array, and are pooled as both classes'.
    Pooled class probabilities are floored at 1e-15 and renormalised.
    """
    pool = group_pooling_function(pooling, problem_type)
    pred = np.asarray(pred)
    if problem_type == BINARY:
        values = np.stack([1 - pred, pred], axis=1)
    else:
        values = pred.reshape(len(pred), -1)
    values = values.astype(np.float64)
    pooled = np.asarray(pool(values, codes), dtype=np.float64)
    if pooled.shape != values.shape:
        raise ValueError(
            f"`group_views` pooling must return an array of the shape of its input {values.shape}, got {pooled.shape}."
        )
    if problem_type != REGRESSION:
        pooled = np.maximum(pooled, _LOG_CLIP)
        pooled /= pooled.sum(axis=1, keepdims=True)
    if problem_type == BINARY:
        pooled = pooled[:, 1]
    return pooled.reshape(pred.shape).astype(pred.dtype)


@dataclass(frozen=True)
class GroupViews:
    """[EXPERIMENTAL] `group_views` of `TabularPredictor.fit` (which documents the views), as a dataclass.

    The equivalent dict works too: `group_views={"pooling": "median"}`. `group_views=True` is `GroupViews()`
    and `group_views="auto"` is `GroupViews(enabled="auto")`.

    Parameters
    ----------
    enabled : bool | str, default True
        True: on, raising if the fit does not allow it. "auto": on when the data and fit allow it. False: off.
    pooling : str | callable | list, default None
        How the weighted ensemble pools each model's row predictions within a group: "geometric_mean",
        "mean", "median" or a callable `pooling(pred, groups)`. None: "geometric_mean" for classification and
        "mean" for regression. A list of these gives every model one pooled view per strategy, each a separate
        input to the weighted ensemble, named `<model>@<strategy>` (a callable's `__name__`).
    views : str | list[str], default ("pooled", "collapsed")
        The views to fit (`GROUP_VIEW_KINDS`): "pooled", every bagged model fit on the rows with its predictions
        pooled within groups, and "collapsed", every bagged model also fit on the collapsed table. "collapsed"
        alone fits each bagged model on the collapsed table only, under its `_GroupCollapsed` name, and skips
        its fit on the rows; `pooling` then cannot be set. "pooled" alone fits no collapsed models.
    """

    enabled: bool | str = True
    pooling: str | GroupPooling | Sequence[str | GroupPooling] | None = None
    views: str | Sequence[str] = GROUP_VIEW_KINDS

    def __post_init__(self):
        if self.enabled not in (True, False, "auto"):
            raise ValueError(f"group_views must be True, False or 'auto', got {self.enabled!r}.")
        views = (self.views,) if isinstance(self.views, str) else tuple(self.views)
        if not views or any(view not in GROUP_VIEW_KINDS for view in views) or len(set(views)) < len(views):
            raise ValueError(
                f"`group_views` views must be a non-empty list of distinct views among {list(GROUP_VIEW_KINDS)}, "
                f"got {self.views!r}."
            )
        # In `GROUP_VIEW_KINDS` order; a frozen dataclass is hashable only with hashable fields.
        object.__setattr__(self, "views", tuple(view for view in GROUP_VIEW_KINDS if view in views))
        if self.pooling is None:
            return
        if "pooled" not in self.views:
            raise ValueError(
                "`group_views` pooling sets how the pooled views pool, but `views` fits only the collapsed ones."
            )
        strategies = group_pooling_strategies(self.pooling)
        if not strategies:
            raise ValueError("`group_views` pooling must list at least one strategy.")
        for pooling in strategies:
            if not callable(pooling) and pooling not in GROUP_POOLING:
                raise ValueError(
                    f"`group_views` pooling must be one of {sorted(GROUP_POOLING)}, a callable or a list of these, "
                    f"got {pooling!r}."
                )
        names = [group_pooling_name(pooling) for pooling in strategies]
        if len(set(names)) < len(names):
            raise ValueError(
                f"`group_views` pooling strategies name the weighted ensemble's inputs, so their names must be "
                f"distinct, got {names}."
            )
        if isinstance(self.pooling, list):
            # A frozen dataclass is hashable only with hashable fields.
            object.__setattr__(self, "pooling", tuple(self.pooling))

    @classmethod
    def from_input(cls, value: bool | str | dict | GroupViews | None) -> GroupViews | None:
        """`fit`'s `group_views` as a `GroupViews`, or None when off."""
        if value is None or value is False:
            return None
        if value is True or isinstance(value, str):
            value = cls(enabled=value)
        elif isinstance(value, dict):
            invalid = set(value) - {f.name for f in fields(cls)}
            if invalid:
                raise ValueError(
                    f"Invalid `group_views` keys: {sorted(invalid)}. Valid keys: {[f.name for f in fields(cls)]}"
                )
            value = cls(**value)
        elif not isinstance(value, cls):
            raise ValueError(f"group_views must be True, False, 'auto', a dict or a GroupViews, got {value!r}.")
        return None if value.enabled is False else value

    def resolve_pooling(
        self, problem_type: str, check_picklable: bool = True
    ) -> str | GroupPooling | tuple[str | GroupPooling, ...]:
        """The pooling strategy for `problem_type` (a built-in's name, or the callable), or a tuple of several.

        A callable is saved with the predictor, so `check_picklable` checks that up front rather than once
        every model is fit. A list of one strategy is that strategy.
        """
        if self.pooling is None:
            return default_group_pooling(problem_type)
        strategies = group_pooling_strategies(self.pooling)
        for pooling in strategies:
            group_pooling_function(pooling, problem_type)
            if callable(pooling) and check_picklable:
                try:
                    pickle.dumps(pooling)
                except (pickle.PicklingError, AttributeError, TypeError) as err:
                    raise ValueError(
                        "A custom `group_views` pooling is saved with the predictor, so it must be picklable: a "
                        f"function defined at module level, not a lambda or a nested function. Got {pooling!r}."
                    ) from err
        return strategies[0] if len(strategies) == 1 else tuple(strategies)


class GroupCollapsedModel(AbstractModel):
    """`model_base` fit on one row per group, its prediction for a group copied to the group's rows.

    The collapsed table has each numeric feature's group mean, each other feature's most frequent value in
    the group (ties: the smallest value; boolean features count as the latter), and the log of the group's
    row count. The `group_columns` features themselves are left out, being constant within a group. A
    group's label is its rows' most frequent label (classification) or their mean (regression); with the
    label constant within groups, both are simply the group's label.

    Bagged like any other model: the bagged folds are group-disjoint, so each fold collapses its own training
    groups and predicts its validation groups.

    Parameters
    ----------
    model_base : AbstractModel
        The unfitted model to fit on the collapsed table.
    group_columns : list[str]
        The `group_on` columns, left out of the collapsed table.
    hyperparameters : dict, default None
        Overrides of `model_base`'s hyperparameters (refit_full passes the trained values through here).
    """

    uses_group_key = True
    _default_auxiliary_params_extra = dict(drop_unique=False)
    _INNER_NAME = "inner"

    def __init__(self, model_base: AbstractModel, group_columns: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model_base = model_base
        self.group_columns = list(group_columns or [])
        self._inner_type: type[AbstractModel] | None = None
        # Whether the last save wrote the inner model: a bag that keeps no fold models (`save_bag_folds=False`,
        # `refit_folds=True`) saves its fold children with the inner model dropped.
        self._inner_saved = False
        self._mean_features: list[str] | None = None
        self._mode_features: list[str] | None = None

    def get_params(self) -> dict:
        params = super().get_params()
        params["model_base"] = self.model_base
        params["group_columns"] = self.group_columns
        return params

    def _get_default_resources(self) -> tuple[int, float]:
        return self.model_base._get_default_resources()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return self.model_base.get_minimum_resources(is_gpu_available=is_gpu_available)

    def _get_maximum_resources(self) -> dict[str, int | float]:
        return self.model_base._get_maximum_resources()

    def declared_refit_hyperparameters(self) -> dict:
        return self.model_base.declared_refit_hyperparameters()

    def _set_collapse_features(self) -> None:
        features = [f for f in self._features if f != GROUP_KEY_COLUMN and f not in self.group_columns]
        numeric = set(
            self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT], invalid_special_types=[S_BOOL])
        )
        self._mean_features = [f for f in features if f in numeric]
        self._mode_features = [f for f in features if f not in numeric]

    def _collapse(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """The collapsed table of `X` (one row per group, in order of first appearance) and each row's group."""
        group, _ = pd.factorize(X[GROUP_KEY_COLUMN].to_numpy(), sort=False)
        parts = []
        if self._mean_features:
            parts.append(X[self._mean_features].groupby(group, sort=True).mean())
        counts = np.bincount(group)
        if self._mode_features:
            parts.append(pd.concat([_group_mode(X[f], group, len(counts)) for f in self._mode_features], axis=1))
        parts.append(pd.DataFrame({GROUP_LOG_ROWS_COLUMN: np.log(counts).astype(np.float64)}))
        X_collapsed = pd.concat([part.reset_index(drop=True) for part in parts], axis=1)
        return X_collapsed, group

    def _collapse_labels(self, y: pd.Series, group: np.ndarray) -> pd.Series:
        y = np.asarray(y)
        n_groups = group.max() + 1
        if self.problem_type == REGRESSION:
            values = np.bincount(group, weights=y.astype(np.float64), minlength=n_groups) / np.bincount(group)
        else:
            counts = np.zeros((n_groups, self.num_classes if self.num_classes else int(y.max()) + 1))
            np.add.at(counts, (group, y.astype(np.int64)), 1)
            values = counts.argmax(axis=1)
        return pd.Series(values)

    def _collapsed_metadata(self, X_collapsed: pd.DataFrame) -> FeatureMetadata:
        type_map_raw = {f: R_FLOAT for f in self._mean_features}
        type_map_raw.update({f: self._feature_metadata.get_feature_type_raw(f) for f in self._mode_features})
        type_map_raw[GROUP_LOG_ROWS_COLUMN] = R_FLOAT
        type_group_map_special = {}
        for feature in self._mode_features:
            for special in self._feature_metadata.get_feature_types_special(feature):
                type_group_map_special.setdefault(special, []).append(feature)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, sample_weight=None, **kwargs):
        X = self.preprocess(X)
        self._set_collapse_features()
        X_collapsed, group = self._collapse(X)
        y_collapsed = self._collapse_labels(y, group)
        if sample_weight is not None:
            sample_weight = np.bincount(group, weights=np.asarray(sample_weight, dtype=np.float64)) / np.bincount(
                group
            )
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            X_val, group_val = self._collapse(X_val)
            y_val = self._collapse_labels(y_val, group_val)
        else:
            X_val, y_val = None, None
        for key in ("X_test", "y_test", "X_unlabeled", "sample_weight_val"):
            kwargs.pop(key, None)

        params = self.model_base.get_params()
        params["hyperparameters"] = {**(params.get("hyperparameters") or {}), **self.get_hyperparameters_init()}
        params["name"] = self._INNER_NAME
        params["path"] = self.path
        inner: AbstractModel = type(self.model_base)(**params)
        inner.set_contexts(path_context=os.path.join(self.path, self._INNER_NAME))
        inner.fit(
            X=X_collapsed,
            y=y_collapsed,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
            sample_weight=sample_weight,
            feature_metadata=self._collapsed_metadata(X_collapsed),
            num_classes=self.num_classes,
            **kwargs,
        )
        self.model = inner
        self._inner_type = type(inner)
        self.params_trained = copy.deepcopy(inner.params_trained)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        X_collapsed, group = self._collapse(X)
        return self.model.predict_proba(X_collapsed)[group]

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        if path is None:
            path = self.path
        inner = self.model
        if inner is not None:
            inner.save(path=os.path.join(path, self._INNER_NAME), verbose=verbose)
        self._inner_saved = inner is not None
        self.model = None
        try:
            path = super().save(path=path, verbose=verbose)
        finally:
            self.model = inner
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._inner_saved:
            model.model = model._inner_type.load(
                path=os.path.join(path, cls._INNER_NAME), reset_paths=reset_paths, verbose=verbose
            )
        return model

    def convert_to_refit_full_template(self):
        template = super().convert_to_refit_full_template()
        if self.model is not None:
            template.model_base = self.model.convert_to_refit_full_template()
        return template


class GroupPooledWeightedEnsembleModel(WeightedEnsembleModel):
    """`WeightedEnsembleModel` over the group views: every base model's predictions are pooled within groups.

    The base models are the bagged models and their `GroupCollapsedModel` counterparts, so the ensemble
    selects among each model's pooled row predictions and its collapsed predictions (the built-in pooling
    strategies leave the latter unchanged, being constant within a group already). The data passed in carries
    `GROUP_KEY_COLUMN`. The `group_pooling` hyperparameter is the pooling strategy (see `GroupViews.pooling`).

    With several strategies, each of a model's pooled views is an input of its own: a copy of the model's stack
    columns under the prefix `<model>@<strategy>`, pooled by that strategy. A collapsed model is one input, under
    its own name and pooled by the first strategy. `_get_model_weights` sums a model's inputs' weights; the
    children's `model_weights` keep them apart.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: The base model and pooling strategy of each input, by stack column prefix, as fit (refit_full later points
        #: the prefixes at the refit base models).
        self._group_inputs: dict[str, tuple[str, str | GroupPooling | None]] = {}

    def _set_default_params(self):
        self._set_default_param_value("group_pooling", None)
        super()._set_default_params()

    def _is_collapsed(self, model: str) -> bool:
        return (self._base_model_types_inner_dict or {}).get(model) is GroupCollapsedModel

    def _inputs(self, base_model_names: list[str]) -> list[tuple[str, str, str | GroupPooling | None]]:
        """The `(stack column prefix, base model, pooling strategy)` of every input of the ensemble."""
        strategies = group_pooling_strategies(self.params["group_pooling"])
        inputs = []
        for model in base_model_names:
            if len(strategies) == 1 or self._is_collapsed(model):
                inputs.append((model, model, strategies[0]))
            else:
                inputs += [
                    (f"{model}{GROUP_POOLED_SEPARATOR}{group_pooling_name(pooling)}", model, pooling)
                    for pooling in strategies
                ]
        return inputs

    def _with_input_columns(self, X: pd.DataFrame, inputs: list[tuple[str, str]]) -> pd.DataFrame:
        """`X` plus the stack columns of each `(prefix, base model)` input, copied from the base model's columns,
        where `X` has those but not the input's."""
        copies = {}
        for prefix, model in inputs:
            if prefix == model:
                continue
            columns, _ = self.set_stack_columns(stack_column_prefix_lst=[prefix])
            model_columns, _ = self.set_stack_columns(stack_column_prefix_lst=[model])
            if columns[0] not in X.columns and all(column in X.columns for column in model_columns):
                copies.update({column: X[model_column] for column, model_column in zip(columns, model_columns)})
        if not copies:
            return X
        return pd.concat([X, pd.DataFrame(copies, index=X.index)], axis=1)

    def _preprocess_set_features(self, X: pd.DataFrame, feature_metadata: FeatureMetadata = None):
        # The inputs are features of the ensemble. They are set before the limits on base models apply, so
        # every base model's are added here, and `_update_feature_metadata` keeps those of the models it keeps.
        inputs = [(prefix, model) for prefix, model, _ in self._inputs(self.base_model_names)]
        super()._preprocess_set_features(X=self._with_input_columns(X, inputs), feature_metadata=feature_metadata)

    def _update_feature_metadata(self, X: pd.DataFrame, feature_metadata: FeatureMetadata) -> FeatureMetadata:
        types_inner = self._base_model_types_inner_dict
        if types_inner is not None:
            # `max_base_models_per_type` counts a collapsed model under the type it collapses, as a type of its
            # own, rather than every collapsed model as one type. Without its model on the rows (`views` of the
            # collapsed views alone), each collapsed model is a type of its own.
            self._base_model_types_inner_dict = {
                model: (
                    (GroupCollapsedModel, types_inner.get(model.replace(GROUP_COLLAPSED_SUFFIX, "", 1), model))
                    if model_type is GroupCollapsedModel
                    else model_type
                )
                for model, model_type in types_inner.items()
            }
        try:
            feature_metadata = super()._update_feature_metadata(X=X, feature_metadata=feature_metadata)
        finally:
            self._base_model_types_inner_dict = types_inner
        inputs = self._inputs(self.base_model_names)
        self._group_inputs = {prefix: (model, pooling) for prefix, model, pooling in inputs}
        prefixes = [prefix for prefix, _, _ in inputs]
        if prefixes == self.stack_column_prefix_lst:
            return feature_metadata
        model_columns = self.stack_columns
        self.stack_column_prefix_lst = prefixes
        self.stack_column_prefix_to_model_map = {prefix: model for prefix, model, _ in inputs}
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=prefixes)
        feature_metadata = feature_metadata.remove_features(
            features=[column for column in model_columns if column not in self.stack_columns]
        )
        new_columns = [column for column in self.stack_columns if column not in model_columns]
        return feature_metadata.join_metadata(
            FeatureMetadata(
                type_map_raw={column: R_FLOAT for column in new_columns},
                type_group_map_special={S_STACK: new_columns},
            )
        )

    def _fit(self, X, y, **kwargs):
        super()._fit(X, y, **kwargs)
        if self._child_has_weights() and len(group_pooling_strategies(self.params["group_pooling"])) > 1:
            weights = sorted(super()._get_model_weights().items(), key=lambda item: item[1], reverse=True)
            logger.log(20, f"\tGroup view weights: {{{', '.join(f'{k!r}: {round(v, 3)}' for k, v in weights)}}}")
        return self

    def _get_model_weights(self) -> dict:
        """The base models' weights, each the sum of the weights of the model's inputs."""
        weights = {}
        for prefix, weight in super()._get_model_weights().items():
            model = self._group_inputs[prefix][0] if prefix in self._group_inputs else prefix
            weights[model] = weights.get(model, 0) + weight
        return weights

    def preprocess(self, X, preprocess_nonadaptive: bool = True, model=None, **kwargs):
        if GROUP_KEY_COLUMN not in X.columns:
            raise KeyError(f"{self.name} pools predictions within groups, so its input needs `{GROUP_KEY_COLUMN}`.")
        codes = X[GROUP_KEY_COLUMN].to_numpy()
        X = self._with_input_columns(X, list(self.stack_column_prefix_to_model_map.items()))
        X = super().preprocess(X, preprocess_nonadaptive=False, **kwargs)
        X = self._pool(X, codes)
        if preprocess_nonadaptive:
            X = super().preprocess(X, preprocess_nonadaptive=True, model=model, compute_base_preds=False)
        return X

    def _pool(self, X: pd.DataFrame, codes: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        k = self.num_pred_cols_per_model
        for i, prefix in enumerate(self.stack_column_prefix_lst):
            columns = self.stack_columns[i * k : (i + 1) * k]
            values = X[columns].to_numpy()
            if k == 1:
                values = values[:, 0]
            pooled = pool_within_groups(values, codes, self.problem_type, self._group_inputs[prefix][1])
            X[columns] = pooled.reshape(len(X), k)
        return X
