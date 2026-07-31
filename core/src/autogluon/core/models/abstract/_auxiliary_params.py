"""Typed schema for model auxiliary params (``params_aux``).

:class:`AuxiliaryParams` is the single source of truth for the auxiliary-param key space:

- Its *materialized* fields seed ``AbstractModel._get_default_auxiliary_params`` (via
  :meth:`AuxiliaryParams.base_defaults`), replacing the historical literal dict.
- Its *wrapper-only* fields are the params consumed by wrapper code without being present
  in the defaults dict (the constraint params of ``_ag_params_common()`` and friends);
  they carry the same fallback defaults the consuming ``params_aux.get(...)`` calls used.
- :meth:`AuxiliaryParams.known_keys` drives the ``FitHelper.verify_model`` validation of
  ``_default_auxiliary_params_extra`` declarations.

Models keep declaring overrides as plain (naturally partial) dicts via
``_default_auxiliary_params_extra``; user overrides keep flowing in as ``ag_args_fit``
dicts. This dataclass is the *merged runtime view*: ``AbstractModel.aux_params`` builds it
from ``params_aux``, giving wrapper code typed, typo-proof attribute access. Keys that are
not fields (model-private params registered via ``_ag_params()``) land in ``extra``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

#: Field metadata marking params that are consumed by wrapper code but intentionally
#: absent from the `_get_default_auxiliary_params` defaults dict.
_WRAPPER_ONLY = {"materialized": False}


@dataclass
class AuxiliaryParams:
    """Typed view of a model's auxiliary params. See the module docstring."""

    # -- materialized defaults: seed `AbstractModel._get_default_auxiliary_params` --
    # Ratio of memory usage allowed by the model. Values > 1.0 have an increased risk of causing OOM errors.
    # Used in memory checks during model training to avoid OOM errors.
    max_memory_usage_ratio: float = 1.0
    # GPU counterpart of max_memory_usage_ratio: ratio of available VRAM the model is allowed to use.
    # Only checked for models that define a GPU memory estimate (see `_estimate_gpu_memory_usage`).
    # If None, GPU memory checks are skipped.
    max_gpu_memory_usage_ratio: float | None = 1.0
    # Ratio of given time_limit to use during fit(). If time_limit == 10 and max_time_limit_ratio=0.3,
    # time_limit would be changed to 3.
    max_time_limit_ratio: float = 1.0
    # max time_limit value during fit(). If the provided time_limit is greater than this value, it will be
    # replaced by max_time_limit. Occurs after max_time_limit_ratio is applied.
    max_time_limit: float | None = None
    # min time_limit value during fit(). If the provided time_limit is less than this value, it will be
    # replaced by min_time_limit. Occurs after max_time_limit is applied.
    min_time_limit: float = 0
    # If a feature's raw type is not in this list, it is pruned.
    valid_raw_types: list[str] | None = None
    # If a feature has a special type not in this list, it is pruned.
    valid_special_types: list[str] | None = None
    # List, drops any features in `self.feature_metadata.type_group_map_special[type]` for type in
    # `ignored_type_group_special`. | Currently undocumented in task.
    ignored_type_group_special: list[str] | None = None
    # List, drops any features in `self.feature_metadata.type_group_map_raw[type]` for type in
    # `ignored_type_group_raw`. | Currently undocumented in task.
    ignored_type_group_raw: list[str] | None = None
    # Kwargs for `autogluon.tabular.features.feature_metadata.FeatureMetadata.get_features()`.
    # Overrides valid_raw_types, valid_special_types, ignored_type_group_special and
    # ignored_type_group_raw. | Currently undocumented in task.
    get_features_kwargs: dict | None = None
    # If not None, applies an additional feature filter to the result of get_feature_kwargs.
    # This should be reserved for users and be None by default. | Currently undocumented in task.
    get_features_kwargs_extra: dict | None = None
    # If not None, calculates `self.predict_1_time` at end of fit call by predicting on this many rows of data.
    predict_1_batch_size: int | None = None
    # Temperature scaling parameter that is set post-fit if calibrate=True during TabularPredictor.fit()
    # on the model with the best validation score and eval_metric="log_loss".
    temperature_scalar: float | None = None

    # -- wrapper-only params: consumed by wrapper code, absent from the defaults dict --
    # (fit-time constraints; see `_ag_params_common` / `_validate_fit_args`)
    # If specified, raises an AssertionError at fit time if len(X) < min_rows.
    min_rows: int | None = field(default=None, metadata=_WRAPPER_ONLY)
    # If specified, raises an AssertionError at fit time if len(X) > max_rows.
    max_rows: int | None = field(default=None, metadata=_WRAPPER_ONLY)
    # If specified, raises an AssertionError at fit time if len(X.columns) > max_features.
    max_features: int | None = field(default=None, metadata=_WRAPPER_ONLY)
    # If specified, raises an AssertionError at fit time if self.num_classes > max_classes.
    max_classes: int | None = field(default=None, metadata=_WRAPPER_ONLY)
    # If specified, raises an AssertionError at fit time if self.problem_type not in problem_types.
    problem_types: list[str] | None = field(default=None, metadata=_WRAPPER_ONLY)
    # If True, ignores the values of `min_rows`, `max_rows`, `max_features`, `max_classes` and `problem_types`.
    ignore_constraints: bool = field(default=False, metadata=_WRAPPER_ONLY)
    # If specified, predictions on more than `max_batch_size` rows are computed in chunks of at most
    # `max_batch_size` rows each (see `_predict_proba_batch`), bounding prediction-time memory usage.
    # Models may declare the sentinel "auto" and resolve it to an int during `_fit`.
    max_batch_size: int | str | None = field(default=None, metadata=_WRAPPER_ONLY)
    # Whether to drop features that have only 1 unique value.
    drop_unique: bool = field(default=True, metadata=_WRAPPER_ONLY)

    #: Keys of `params_aux` that are not fields above: model-private params (registered via
    #: the model's `_ag_params()`, e.g. TabPFN's `model_telemetry`) and any user-supplied extras.
    extra: dict[str, Any] = field(default_factory=dict, metadata=_WRAPPER_ONLY)

    @classmethod
    def base_defaults(cls) -> dict[str, Any]:
        """The `_get_default_auxiliary_params` base defaults dict (materialized fields only)."""
        return {f.name: f.default for f in dataclasses.fields(cls) if f.metadata.get("materialized", True)}

    @classmethod
    def known_keys(cls) -> set[str]:
        """All schema field names (materialized and wrapper-only), excluding the `extra` catch-all."""
        return {f.name for f in dataclasses.fields(cls) if f.name != "extra"}

    @classmethod
    def from_dict(cls, params_aux: dict[str, Any]) -> AuxiliaryParams:
        """Build the typed view from a `params_aux` dict; non-field keys land in `extra`."""
        known = cls.known_keys()
        kwargs = {k: v for k, v in params_aux.items() if k in known}
        extra = {k: v for k, v in params_aux.items() if k not in known}
        return cls(**kwargs, extra=extra)
