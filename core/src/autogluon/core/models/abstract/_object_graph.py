"""Walkers over a fitted estimator's object graph for the shared-weights mixin.

The mixin never asks a wrapper where its library keeps the network. It finds the registry payload
inside the fitted estimator by identity, remembers the paths, nulls them for the pickle and puts the
payload back on load; it moves the remaining tensors to the CPU for the pickle and rewrites device
fields when a model changes device. Everything here is torch-lazy: torch is imported inside the
functions, and the module is importable without it.

A path is a tuple of steps, each ``("attr", name)``, ``("key", key)`` or ``("index", i)``. Walks
descend into dicts, lists, tuples and plain objects (anything with a ``__dict__`` that is not a
class, module or function), stop at tensors, arrays, frames, strings and numbers, treat a
``torch.nn.Module`` as one unit, follow every object once (aliases keep aliasing, cycles end) and
give up on a branch deeper than :data:`MAX_DEPTH` or a graph larger than :data:`MAX_NODES`.
"""

from __future__ import annotations

import copy
import types
from typing import TYPE_CHECKING, Any

from ._shared_weights_registry import normalize_device, shallow_copy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import torch

Step = tuple[str, Any]
Path = tuple[Step, ...]

#: Deepest path a walk follows; a deeper branch is left untouched.
MAX_DEPTH = 16
#: Most objects one walk visits; a larger graph is left untouched beyond the budget.
MAX_NODES = 250_000

_SCALAR_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None), range, slice)
_CODE_TYPES = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    types.BuiltinMethodType,
    property,
    staticmethod,
    classmethod,
)
#: Top-level packages whose objects are data, never containers of a network.
_DATA_PACKAGES = frozenset({"numpy", "pandas", "scipy", "pyarrow", "polars", "sklearn.utils._bunch"})
#: Sentinel of the leaf helpers: "not a leaf, keep walking".
_NOT_A_LEAF = object()


def _module_root(obj: Any) -> str:
    module = getattr(type(obj), "__module__", "") or ""
    return module.split(".")[0]


def _is_tensor(obj: Any) -> bool:
    torch = _torch()
    return torch is not None and isinstance(obj, torch.Tensor)


def _is_module(obj: Any) -> bool:
    torch = _torch()
    return torch is not None and isinstance(obj, torch.nn.Module)


def _is_torch_device(obj: Any) -> bool:
    torch = _torch()
    return torch is not None and isinstance(obj, torch.device)


def _torch() -> Any:
    import sys

    return sys.modules.get("torch")


def is_leaf(obj: Any) -> bool:
    """Whether a walk stops at ``obj`` without looking inside it."""
    if isinstance(obj, (_SCALAR_TYPES, _CODE_TYPES)):
        return True
    if _is_tensor(obj) or _is_module(obj) or _is_torch_device(obj):
        return True
    torch = _torch()
    if torch is not None and isinstance(obj, (torch.dtype, torch.Generator)):
        return True
    return _module_root(obj) in _DATA_PACKAGES


def _children(obj: Any) -> Iterator[tuple[Step, Any]]:
    """The ``(step, child)`` pairs a walk descends into; empty for a leaf or an object without ``__dict__``."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            yield ("key", key), value
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield ("index", index), value
    elif hasattr(obj, "__dict__"):
        for name, value in list(vars(obj).items()):
            yield ("attr", name), value


def _with_child(parent: Any, step: Step, value: Any) -> Any:
    """``parent`` with the child at ``step`` replaced: mutable containers in place, a tuple rebuilt."""
    kind, where = step
    if kind == "key":
        parent[where] = value
    elif kind == "index":
        if isinstance(parent, tuple):
            items = list(parent)
            items[where] = value
            return type(parent)(items) if type(parent) is not tuple else tuple(items)
        parent[where] = value
    else:
        try:
            setattr(parent, where, value)
        except (AttributeError, TypeError):  # frozen dataclass
            object.__setattr__(parent, where, value)
    return parent


def _get_child(parent: Any, step: Step) -> Any:
    kind, where = step
    if kind == "key":
        return parent[where]
    if kind == "index":
        return parent[where]
    return getattr(parent, where)


def assign(root: Any, path: Path, value: Any) -> Any:
    """Set ``value`` at ``path`` below ``root``; returns ``root`` (a new tuple when ``root`` itself is one)."""
    if not path:
        return value
    step, rest = path[0], path[1:]
    child = _get_child(root, step)
    new_child = assign(child, rest, value) if rest else value
    if new_child is not child or not rest:
        return _with_child(root, step, new_child)
    return root


def get_at(root: Any, path: Path) -> Any:
    """The object at ``path`` below ``root``."""
    current = root
    for step in path:
        current = _get_child(current, step)
    return current


# --- payloads -------------------------------------------------------------------------------------


def iter_modules(payload: Any) -> Iterator[torch.nn.Module]:
    """Every ``torch.nn.Module`` reachable from ``payload`` (a module, a container or a state dict), each once."""
    seen: set[int] = set()

    def walk(obj: Any, depth: int) -> Iterator[Any]:
        if obj is None or depth > 6 or id(obj) in seen:
            return
        seen.add(id(obj))
        if _is_module(obj):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                yield from walk(value, depth + 1)
        elif not is_leaf(obj) and hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                yield from walk(value, depth + 1)

    yield from walk(payload, 0)


def payload_modules(payload: Any) -> list[torch.nn.Module]:
    """The modules of a payload, in walk order."""
    return list(iter_modules(payload))


def shared_components(payload: Any) -> list[Any]:
    """The objects a detach looks for by identity: the payload itself first, then its modules.

    The order is a function of the payload's structure, so the registry entry of another device
    yields the same indices and a recorded ``(path, index)`` pair points at the matching object.
    """
    components: list[Any] = [payload]
    for module in iter_modules(payload):
        if module is not payload:
            components.append(module)
    return components


def payload_device(payload: Any) -> str:
    """Device type of the first parameter, buffer or tensor found in ``payload``; ``"cpu"`` when none."""
    if _is_tensor(payload):
        return normalize_device(payload.device)
    if isinstance(payload, dict):
        for value in payload.values():
            if _is_tensor(value):
                return normalize_device(value.device)
    for module in iter_modules(payload):
        for tensor in module.parameters():
            return normalize_device(tensor.device)
        for tensor in module.buffers():
            return normalize_device(tensor.device)
    return "cpu"


def check_payload_device(payload: Any, device: str) -> None:
    """Raise ``RuntimeError`` when ``payload`` does not live on device type ``device``."""
    expected = normalize_device(device)
    actual = payload_device(payload)
    if actual != expected:
        raise RuntimeError(
            f"the shared weights live on {actual!r} but the estimator asked for {expected!r}; a shared payload is "
            "never moved with .to(), take the registry entry for the requested device instead"
        )


def mark_identity_under_deepcopy(payload: Any) -> None:
    """Make ``copy.deepcopy(payload)`` return ``payload`` itself (sklearn's ``clone`` deep-copies constructor arguments).

    Applied to container payloads only, never to a bare ``torch.nn.Module``: libraries deep-copy
    modules on purpose to obtain private copies (tabpfn's per-device cache), and those must stay
    copies. Objects that reject new attributes are left alone.
    """
    if payload is None or _is_module(payload) or isinstance(payload, (dict, list, tuple)) or is_leaf(payload):
        return
    try:
        object.__setattr__(payload, "__deepcopy__", lambda memo, _p=payload: _p)
    except (AttributeError, TypeError):
        return


# --- detach for the pickle ------------------------------------------------------------------------


class _Budget:
    def __init__(self) -> None:
        self.nodes = 0

    def take(self) -> bool:
        self.nodes += 1
        return self.nodes <= MAX_NODES


def _cpu_copy_of_module(module: torch.nn.Module) -> torch.nn.Module:
    for tensor in module.parameters():
        if tensor.device.type != "cpu":
            return copy.deepcopy(module).to("cpu")
    for tensor in module.buffers():
        if tensor.device.type != "cpu":
            return copy.deepcopy(module).to("cpu")
    return module


def detach_shared(
    root: Any, shared: Sequence[Any] = (), *, shared_ids: Sequence[int] | None = None
) -> tuple[Any, list[tuple[Path, int]], list[Path]]:
    """A copy of ``root``'s object graph with every reference to a ``shared`` object nulled and every tensor on the CPU.

    Containers on the way to a change are shallow-copied (the live estimator keeps everything);
    aliases stay aliases in the copy. Tensors are replaced by detached CPU copies, an owned module
    with tensors off the CPU by a deep copy on the CPU. Bound methods of the root estimator (the
    instance-level shadows a fit installs) are dropped.

    ``shared_ids`` names the objects by identity instead (the ids recorded when they were alive), so an
    estimator whose registry entry was evicted still pickles without its network.

    Returns:
        ``(copy, paths, moved)``: ``paths`` lists ``(path, index into shared)`` for every nulled
        reference, so :func:`attach_shared` can put another device's payload back; ``moved`` lists
        the paths of tensors and owned modules that were on another device, so :func:`move_to`
        can return them to the model's device after a reload.
    """
    shared_ids = {
        identity: index
        for index, identity in enumerate(shared_ids if shared_ids is not None else [id(obj) for obj in shared])
    }
    paths: list[tuple[Path, int]] = []
    moved: list[Path] = []
    memo: dict[int, Any] = {}
    in_progress: set[int] = set()
    budget = _Budget()

    def transform(obj: Any, path: Path, depth: int) -> Any:
        index = shared_ids.get(id(obj))
        if index is not None:
            paths.append((path, index))
            return None
        leaf = _cpu_leaf(obj)
        if leaf is not _NOT_A_LEAF:
            if leaf is not obj:
                moved.append(path)
            return leaf
        key = id(obj)
        if is_leaf(obj) or depth > MAX_DEPTH or key in in_progress or not budget.take():
            return obj
        if key in memo:
            return memo[key]
        in_progress.add(key)
        try:
            result = _transform_container(obj, path, depth, transform)
        finally:
            in_progress.discard(key)
        memo[key] = result
        return result

    result = transform(root, (), 0)
    if result is root:
        result = shallow_copy(root) if hasattr(root, "__dict__") else result
    if hasattr(result, "__dict__"):
        for name, value in list(vars(result).items()):
            if isinstance(value, types.MethodType) and value.__self__ is root:
                del result.__dict__[name]
    return result, paths, moved


def move_to(root: Any, paths: Sequence[Path], device: str) -> Any:
    """Move the tensor or module at every path to device type ``device`` (tensors replaced, modules in place)."""
    device_type = normalize_device(device)
    for path in paths:
        try:
            value = get_at(root, path)
        except (AttributeError, KeyError, IndexError, TypeError):
            continue
        if _is_module(value):
            value.to(device_type)
        elif _is_tensor(value) and value.device.type != device_type:
            root = assign(root, path, value.to(device_type))
    return root


def _cpu_leaf(obj: Any) -> Any:
    """A CPU copy for a tensor or an owned module, ``_NOT_A_LEAF`` for anything else."""
    if _is_tensor(obj):
        return obj.detach().to("cpu") if obj.device.type != "cpu" else obj
    if _is_module(obj):
        return _cpu_copy_of_module(obj)
    return _NOT_A_LEAF


def _transform_container(obj: Any, path: Path, depth: int, transform: Callable[[Any, Path, int], Any]) -> Any:
    if isinstance(obj, dict):
        return _transform_dict(obj, path, depth, transform)
    if isinstance(obj, (list, tuple)):
        return _transform_sequence(obj, path, depth, transform)
    if hasattr(obj, "__dict__"):
        return _transform_object(obj, path, depth, transform)
    return obj


def _transform_dict(obj: dict, path: Path, depth: int, transform: Callable[[Any, Path, int], Any]) -> Any:
    items = {key: transform(value, (*path, ("key", key)), depth + 1) for key, value in list(obj.items())}
    if all(items[key] is value for key, value in obj.items()):
        return obj
    new = type(obj)() if type(obj) is not dict else {}
    new.update(items)
    return new


def _transform_sequence(obj: list | tuple, path: Path, depth: int, transform: Callable[[Any, Path, int], Any]) -> Any:
    values = [transform(value, (*path, ("index", i)), depth + 1) for i, value in enumerate(obj)]
    if all(new is old for new, old in zip(values, obj, strict=False)):
        return obj
    if isinstance(obj, tuple):
        return tuple(values) if type(obj) is tuple else type(obj)(*values)
    return list(values)


def _transform_object(obj: Any, path: Path, depth: int, transform: Callable[[Any, Path, int], Any]) -> Any:
    changes = {}
    for name, value in list(vars(obj).items()):
        new_value = transform(value, (*path, ("attr", name)), depth + 1)
        if new_value is not value:
            changes[name] = new_value
    if not changes:
        return obj
    new = shallow_copy(obj)
    new.__dict__.update(changes)
    return new


def attach_shared(root: Any, paths: Sequence[tuple[Path, int]], shared: Sequence[Any]) -> Any:
    """Put ``shared[index]`` back at every recorded path of ``root``; returns ``root`` (rebuilt when it is a tuple)."""
    for path, index in paths:
        root = assign(root, path, shared[index])
    return root


def find_shared(root: Any, shared: Sequence[Any]) -> list[tuple[Path, int]]:
    """The ``(path, index)`` pairs where ``root`` references a ``shared`` object, without copying anything."""
    shared_ids = {id(obj): index for index, obj in enumerate(shared)}
    found: list[tuple[Path, int]] = []
    seen: set[int] = set()
    budget = _Budget()

    def walk(obj: Any, path: Path, depth: int) -> None:
        index = shared_ids.get(id(obj))
        if index is not None:
            found.append((path, index))
            return
        if is_leaf(obj) or depth > MAX_DEPTH or id(obj) in seen or not budget.take():
            return
        seen.add(id(obj))
        for step, child in _children(obj):
            walk(child, (*path, step), depth + 1)

    walk(root, (), 0)
    return found


# --- device changes -------------------------------------------------------------------------------


def _matches_device_string(value: str, old_type: str) -> bool:
    return value == old_type or value.startswith(old_type + ":")


def rewrite_devices(root: Any, old: str, new: str, *, skip: Sequence[Any] = ()) -> Any:
    """In place: every device field, owned tensor and owned module recorded on ``old`` now names or lives on ``new``.

    ``torch.device`` values and device strings equal to ``old`` (or ``"old:N"``) become ``new``;
    tensors on ``old`` are replaced by ``.to(new)`` copies; owned modules are moved with
    ``.to(new)``. Objects in ``skip`` (the shared payload and its modules) are never touched or
    entered. Tuples along the way are rebuilt on their parent. Returns ``root`` (rebuilt when it is
    a tuple).
    """
    old_type = normalize_device(old)
    new_type = normalize_device(new)
    if old_type == new_type:
        return root
    torch = _torch()
    skip_ids = {id(obj) for obj in skip}
    seen: set[int] = set()
    budget = _Budget()

    def device_leaf(obj: Any) -> Any:
        """The rewritten value for a device string, device, tensor or owned module; ``_NOT_A_LEAF`` otherwise."""
        if isinstance(obj, str):
            return new_type if _matches_device_string(obj, old_type) else obj
        if _is_torch_device(obj):
            return torch.device(new_type) if obj.type == old_type else obj
        if _is_tensor(obj):
            return obj.to(new_type) if obj.device.type == old_type else obj
        if _is_module(obj):
            obj.to(new_type)
            return obj
        return _NOT_A_LEAF

    def transform(obj: Any, depth: int) -> Any:
        if id(obj) in skip_ids:
            return obj
        leaf = device_leaf(obj)
        if leaf is not _NOT_A_LEAF:
            return leaf
        if is_leaf(obj) or depth > MAX_DEPTH or id(obj) in seen or not budget.take():
            return obj
        seen.add(id(obj))
        for step, child in _children(obj):
            new_child = transform(child, depth + 1)
            if new_child is not child:
                obj = _with_child(obj, step, new_child)
        if isinstance(obj, dict):
            _rewrite_device_keys(obj, old_type, new_type)
        return obj

    return transform(root, 0)


def _rewrite_device_keys(mapping: dict, old_type: str, new_type: str) -> None:
    """Re-key ``torch.device`` keys of ``old_type`` to ``new_type`` in place (tabpfn's per-device caches)."""
    torch = _torch()
    for key in list(mapping):
        if _is_torch_device(key) and key.type == old_type:
            mapping[torch.device(new_type)] = mapping.pop(key)


def tensors_off_device(root: Any, device: str, *, skip: Sequence[Any] = ()) -> list[Path]:
    """Paths of tensors (including module parameters and buffers) in ``root`` that are not on device type ``device``."""
    target = normalize_device(device)
    skip_ids = {id(obj) for obj in skip}
    found: list[Path] = []
    seen: set[int] = set()
    budget = _Budget()

    def walk(obj: Any, path: Path, depth: int) -> None:
        if id(obj) in skip_ids:
            return
        if _is_tensor(obj):
            if obj.device.type != target:
                found.append(path)
            return
        if _is_module(obj):
            for name, tensor in list(obj.named_parameters()) + list(obj.named_buffers()):
                if tensor.device.type != target:
                    found.append((*path, ("attr", name)))
            return
        if is_leaf(obj) or depth > MAX_DEPTH or id(obj) in seen or not budget.take():
            return
        seen.add(id(obj))
        for step, child in _children(obj):
            walk(child, (*path, step), depth + 1)

    walk(root, (), 0)
    return found


__all__ = [
    "MAX_DEPTH",
    "MAX_NODES",
    "Path",
    "Step",
    "assign",
    "attach_shared",
    "check_payload_device",
    "detach_shared",
    "find_shared",
    "get_at",
    "is_leaf",
    "iter_modules",
    "mark_identity_under_deepcopy",
    "move_to",
    "payload_device",
    "payload_modules",
    "rewrite_devices",
    "shared_components",
    "tensors_off_device",
]
