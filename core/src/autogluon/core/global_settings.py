"""Process-wide AutoGluon settings that belong to no single model class.

The counterpart of a model class's :class:`~autogluon.core.models.abstract.ClassSettings` for
knobs shared by every model in the process, for example the number of pretrained networks the
shared-weights registry keeps loaded. Set them with :func:`set_global_settings` or with
``TabularPredictor.fit(global_settings={...})``, which also records them on the predictor so they
are applied again when it is loaded. Every model snapshots the explicitly set values when it is
constructed and re-applies them where it is fit or loaded, so a fold model fit in a worker process
runs under the settings of the process that launched the fit.

A new setting is a field of :class:`GlobalSettings` (``None`` meaning "not set", so the component
keeps its own default) plus the lines in :func:`_apply` that hand it to the component it configures.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalSettings:
    """Process-wide settings; a field left at ``None`` keeps the default of the component it configures.

    Parameters
    ----------
    shared_weights_capacity : int | None, default None
        Number of pretrained networks the shared-weights registry keeps loaded at once, least
        recently used first out (see
        :mod:`autogluon.core.models.abstract._shared_weights_registry`). ``0`` keeps none, so every
        fit loads its checkpoint. ``None`` uses the registry default: the environment variable
        ``AG_SHARED_WEIGHTS_CAPACITY`` when set, otherwise 2.
    """

    shared_weights_capacity: int | None = None

    def __post_init__(self):
        capacity = self.shared_weights_capacity
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError(f"shared_weights_capacity must be a non-negative int or None, got {capacity!r}.")

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def explicit(self) -> dict[str, Any]:
        """The settings that are set, i.e. not ``None``."""
        return {name: value for name, value in self.to_dict().items() if value is not None}

    def replace(self, **values: Any) -> GlobalSettings:
        """A copy with ``values`` applied; an unknown key raises ``ValueError``."""
        unknown = sorted(set(values) - set(self.field_names()))
        if unknown:
            raise ValueError(f"Unknown global setting(s) {unknown}; valid settings are {self.field_names()}.")
        return dataclasses.replace(self, **values)


_SETTINGS = GlobalSettings()
_SET = False
_LOCK = threading.Lock()


def get_global_settings() -> GlobalSettings:
    """The process-wide settings, defaults until :func:`set_global_settings` is called."""
    return _SETTINGS


def set_global_settings(**values: Any) -> GlobalSettings:
    """Set process-wide settings and apply them to the components they configure.

    Keys are :class:`GlobalSettings` fields; an unknown key or an invalid value raises
    ``ValueError`` and changes nothing. Setting a field to ``None`` restores that component's
    default. A change to a value set earlier in the process is logged, since models fit under the
    old value see the new one from now on.

    Examples
    --------
    >>> from autogluon.core.global_settings import set_global_settings
    >>> set_global_settings(shared_weights_capacity=4)  # keep up to 4 pretrained networks loaded
    GlobalSettings(shared_weights_capacity=4)
    """
    global _SETTINGS, _SET
    with _LOCK:
        current = _SETTINGS
        new = current.replace(**values)
        if _SET and new != current:
            logger.log(
                30,
                f"\tGlobal settings change from {current.explicit()} to {new.explicit()}: "
                f"every model in the process now uses the new values.",
            )
        _apply(old=current, new=new)
        _SETTINGS = new
        _SET = True
        return new


def _apply(*, old: GlobalSettings, new: GlobalSettings) -> None:
    """Hand each setting to the component it configures; a field back at ``None`` restores its default."""
    if new.shared_weights_capacity is not None or old.shared_weights_capacity is not None:
        from .models.abstract import _shared_weights_registry

        _shared_weights_registry.set_capacity(new.shared_weights_capacity)


__all__ = ["GlobalSettings", "get_global_settings", "set_global_settings"]
