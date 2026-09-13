"""Settings a model class owns for the whole process, as opposed to per-config hyperparameters.

A hyperparameter belongs to one config: two configs of the same class can hold different
values without affecting each other. Some knobs cannot work that way, because they steer
state every model of the class in the process shares -- a registry of built networks, a
cache size, a telemetry switch. Setting such a knob on one config would silently change
what every other config of the class sees. A model class declares those knobs as a
:class:`ClassSettings` dataclass in ``class_settings_cls``; ``AbstractModel`` holds one
instance per declaring class, and ``TabularPredictor.fit(model_class_settings=...)`` sets
them once, before any model trains.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassSettings:
    """Base for a model class's process-wide settings; subclasses declare fields with defaults."""

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **values: Any) -> ClassSettings:
        """A copy with ``values`` applied; an unknown key raises ``ValueError``."""
        unknown = sorted(set(values) - set(self.field_names()))
        if unknown:
            raise ValueError(
                f"Unknown {type(self).__name__} setting(s) {unknown}; valid settings are {self.field_names()}."
            )
        return dataclasses.replace(self, **values)
