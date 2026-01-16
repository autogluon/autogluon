from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImportTarget:
    module: str
    qualname: str  # e.g. "MyCustomFE" or "Outer.Inner"


def parse_import_target(s: str) -> ImportTarget:
    # Accept "pkg.mod:Class" (preferred) and "pkg.mod.Class" (optional)
    if ":" in s:
        module, qualname = s.split(":", 1)
        return ImportTarget(module=module, qualname=qualname)
    # dotted fallback: split last token as qualname
    parts = s.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid import target: {s!r}")
    return ImportTarget(module=".".join(parts[:-1]), qualname=parts[-1])


def import_by_target(target: ImportTarget) -> Any:
    mod = importlib.import_module(target.module)
    obj = mod
    for attr in target.qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def resolve_fg_class(
    name: str,
    *,
    registry: dict[str, type] | None = None,
    import_map: dict[str, str] | None = None,
    base_class: type | None = None,  # e.g. FeatureGenerator
) -> type:
    # 1) registry
    if registry and name in registry:
        cls = registry[name]
    else:
        # 2) import_map alias
        if import_map and name in import_map:
            target = parse_import_target(import_map[name])
            cls = import_by_target(target)
        else:
            # 3) direct import target (optional convenience)
            if ":" in name or (name.count(".") >= 1 and name[0].isalpha()):
                target = parse_import_target(name)
                cls = import_by_target(target)
            else:
                raise KeyError(
                    f"Unknown feature generator {name!r}. "
                    f"Known registry keys: {sorted(registry or {})}... "
                    f"Known import_map keys: {sorted(import_map or {})}..."
                )

    if not inspect.isclass(cls):
        raise TypeError(f"Resolved {name!r} to non-class object: {cls!r}")
    if base_class is not None and not issubclass(cls, base_class):
        raise TypeError(f"{cls} is not a subclass of {base_class}")
    return cls
