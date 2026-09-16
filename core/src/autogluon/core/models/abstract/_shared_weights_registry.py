"""Process-wide registry of pretrained checkpoint weights shared by foundation-model wrappers.

What is shared
    One immutable object per :class:`WeightsKey` per process, where the key names the checkpoint
    file, the estimator variant, the device type, the parameter dtype and the forward-affecting
    construction flags. The object is usually the built ``torch.nn.Module`` (in-context models
    never mutate their weights during a fit) but can be a state dict or a small container that the
    library's estimator accepts as its network. A bagged fit with 8 fold children and a refit child
    otherwise reads the same checkpoint and builds the same network 9 times, inside the timed fit.

Why it is fair
    Pretrained weights are part of the environment, like an imported library or a CUDA context:
    they are identical for every fit of that checkpoint and a served deployment keeps them resident.
    The first fit of a checkpoint in a process (an untimed warm-up fit, or the first bagged child)
    loads the entry, every later fit takes a hit through :func:`get_or_load`, and every result
    records what happened (:meth:`WeightsEntry.to_metadata`, ``info["shared_weights"]`` of each
    model, ``report()``). A key is derived from the loader's inputs and the device only, never from
    the task's data, so an entry carries no task-specific state into a fit.

What is never shared
    Anything written during a fit: fine-tuned parameters (a fine-tuning model takes a deep copy of
    the entry, ``SharedWeights.copy_per_fit``), fitted context, KV caches, compiled graphs and other
    data-derived state. A configuration under which the library writes into its network (a flag
    that compiles or casts the module) is named in ``SharedWeights.disabled_by`` and loads its own
    network exactly as before.

Loader contract
    A loader is a zero-argument callable that builds the object for one key. The registry runs it
    with the Python, NumPy and torch (CPU and CUDA) random states saved and restored, so building a
    network never advances a global random number generator, and holds its lock meanwhile, so
    concurrent callers of one key share one load. A loader that raises registers nothing and the
    exception propagates to the caller. :mod:`.shared_weights` wraps the library's own network
    loader as this callable; nothing else calls the registry directly.

Model convention
    A model class declares :class:`autogluon.core.models.abstract.shared_weights.SharedWeights`;
    that module's docstring is the how-to (the declaration, the key, the weightless pickle, device
    swaps, the metadata block). This module is the registry only.

Capacity
    The registry keeps :data:`DEFAULT_CAPACITY` entries, least recently used first out. The
    environment variable :data:`CAPACITY_ENV_VAR` (or, for compatibility with earlier deployments,
    :data:`LEGACY_CAPACITY_ENV_VAR`) overrides the default; :func:`set_capacity` sets it in process.

This module imports torch only inside the functions that need it, so importing it keeps
``import autogluon.core.models`` cheap.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._class_settings import ClassSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import torch

logger = logging.getLogger(__name__)

#: Which stage first loaded a registry entry; the provenance the metadata reports.
LoadedBy = Literal["warmup", "fit", "load"]

#: Environment variable overriding the default registry capacity.
CAPACITY_ENV_VAR = "AG_SHARED_WEIGHTS_CAPACITY"
#: Earlier name of :data:`CAPACITY_ENV_VAR`, read when the current one is unset.
LEGACY_CAPACITY_ENV_VAR = "TABARENA_SHARED_WEIGHTS_CAPACITY"
DEFAULT_CAPACITY = 2


def normalize_device(device: str | torch.device | None) -> str:
    """The device type only, for example ``"cuda"`` for ``"cuda:0"``; ``None`` means ``"cpu"``.

    The index carries no information for the registry, so wrappers that spell the device
    ``"cuda"`` share the entry with wrappers that spell it ``"cuda:0"``.
    """
    if device is None:
        return "cpu"
    return str(device).strip().lower().split(":")[0]


@dataclass(frozen=True)
class WeightsKey:
    """Identity of one immutable network in the process registry.

    Parameters
    ----------
    library : str
        Short library name, for example ``"tabpfn"``, ``"tabicl"``, ``"mitra"``.
    checkpoint : str
        Resolved local path (the Hugging Face blob path after symlink resolution) or a stable
        opaque id for sources that are not a single file.
    variant : str
        Estimator discriminator, for example ``"classifier"`` or ``"regressor"``.
    device : str
        Device type from :func:`normalize_device`.
    dtype : str
        Parameter dtype of the stored network, for example ``"float32"`` or ``"bfloat16"``.
    flags : tuple of (str, str)
        Forward-affecting construction knobs as sorted ``(name, value)`` string pairs.
    """

    library: str
    checkpoint: str
    variant: str
    device: str
    dtype: str = "float32"
    flags: tuple[tuple[str, str], ...] = ()

    def replace(self, **changes: Any) -> WeightsKey:
        """A copy with ``changes`` applied; ``device`` is normalized, ``flags`` re-sorted."""
        if "device" in changes:
            changes["device"] = normalize_device(changes["device"])
        if "flags" in changes:
            changes["flags"] = _normalize_flags(changes["flags"])
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form with named fields; ``flags`` becomes a dict."""
        return {
            "library": self.library,
            "checkpoint": self.checkpoint,
            "variant": self.variant,
            "device": self.device,
            "dtype": self.dtype,
            "flags": dict(self.flags),
        }

    def short(self) -> str:
        """Compact log label: library, variant, device, dtype, checkpoint basename and flags."""
        label = f"{self.library}/{self.variant}@{self.device}:{self.dtype}[{Path(self.checkpoint).name}]"
        if self.flags:
            label += "{" + ",".join(f"{name}={value}" for name, value in self.flags) + "}"
        return label


def _normalize_flags(flags: Any) -> tuple[tuple[str, str], ...]:
    """Sorted ``(name, value-as-str)`` pairs from a mapping or an iterable of pairs."""
    items = flags.items() if hasattr(flags, "items") else flags
    return tuple(sorted((str(name), str(value)) for name, value in items))


def make_key(
    library: str,
    checkpoint: str | Path,
    variant: str,
    device: str | torch.device | None,
    *,
    dtype: str = "float32",
    **flags: Any,
) -> WeightsKey:
    """Build a :class:`WeightsKey`, resolving a path checkpoint and normalizing device and flags.

    A ``checkpoint`` that names an existing path is resolved to its absolute target (symlinks
    followed, so a Hugging Face snapshot link and its blob agree); any other string is kept as an
    opaque stable id.
    """
    if isinstance(checkpoint, Path) or os.path.exists(str(checkpoint)):
        checkpoint = str(Path(checkpoint).resolve())
    return WeightsKey(
        library=library,
        checkpoint=str(checkpoint),
        variant=variant,
        device=normalize_device(device),
        dtype=dtype,
        flags=_normalize_flags(flags),
    )


@dataclass
class WeightsEntry:
    """One registered object with its load provenance.

    Parameters
    ----------
    key : WeightsKey
        The registry key.
    value : Any
        The shared object (a module, a state dict or a small container).
    loaded_by : LoadedBy
        Stage that ran the loader.
    load_time_s : float
        Wall-clock seconds the loader took.
    n_bytes : int or None
        Tensor bytes of ``value`` when they could be counted, else ``None``.
    source : dict
        Where the checkpoint came from (repo id, filename, revision, snapshot sha) for metadata
        that must not depend on absolute host paths.
    hits : int
        Registry hits since the entry was loaded or the stats were reset.
    """

    key: WeightsKey
    value: Any
    loaded_by: LoadedBy
    load_time_s: float
    n_bytes: int | None
    source: dict[str, Any] = field(default_factory=dict)
    hits: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """JSON-able record without the object reference.

        ``checkpoint`` is reduced to its basename (for a Hugging Face blob that is the content
        sha256, a stable identifier) so the record carries no absolute host path; ``source`` holds
        the repo id and filename. :meth:`WeightsKey.to_dict` keeps the absolute path and is meant
        for logs and in-process use only.
        """
        return {
            **self.key.to_dict(),
            "checkpoint": Path(self.key.checkpoint).name,
            "loaded_by": self.loaded_by,
            "load_time_s": self.load_time_s,
            "n_bytes": self.n_bytes,
            "source": dict(self.source),
            "hits": self.hits,
        }


_ENTRIES: OrderedDict[WeightsKey, WeightsEntry] = OrderedDict()
_LOCK = threading.RLock()
_CAPACITY: int | None = None


def _fresh_stats() -> dict[str, int]:
    return {"hits": 0, "misses": 0, "evictions": 0}


_STATS: dict[str, int] = _fresh_stats()


def _capacity_from_env() -> int:
    for name in (CAPACITY_ENV_VAR, LEGACY_CAPACITY_ENV_VAR):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Ignoring %s=%r (not an integer); using %d.", name, raw, DEFAULT_CAPACITY)
            return DEFAULT_CAPACITY
    return DEFAULT_CAPACITY


def capacity() -> int:
    """Entries the registry keeps; ``DEFAULT_CAPACITY`` unless an environment variable overrides it."""
    global _CAPACITY
    with _LOCK:
        if _CAPACITY is None:
            _CAPACITY = _capacity_from_env()
        return _CAPACITY


def set_capacity(n: int) -> None:
    """Set the capacity and evict least recently used entries beyond it; ``0`` stores nothing."""
    global _CAPACITY
    with _LOCK:
        _CAPACITY = max(0, int(n))
        _evict_to(_CAPACITY)


def _evict_to(n: int) -> None:
    while len(_ENTRIES) > n:
        _, entry = _ENTRIES.popitem(last=False)
        _STATS["evictions"] += 1
        logger.log(
            30 if entry.loaded_by == "warmup" else 20,
            "Shared weights %s (loaded by %s) evicted (capacity %d); the next fit of that checkpoint loads it again. "
            "Raise %s or release unused entries.",
            entry.key.short(),
            entry.loaded_by,
            n,
            CAPACITY_ENV_VAR,
        )


@contextmanager
def _rng_guard(*, cuda: bool) -> Iterator[None]:
    """Save and restore the Python, NumPy and torch random states around a loader.

    The torch generators are forked with ``torch.random.fork_rng`` over the CPU and, when ``cuda``
    is requested and available, every CUDA device. Torch is imported here when it is importable so
    the fork also covers a loader that imports torch itself.
    """
    py_state = random.getstate()
    np_module = sys.modules.get("numpy")
    np_state = np_module.random.get_state() if np_module is not None else None
    try:
        import torch
    except ImportError:
        torch = None
    try:
        if torch is None:
            yield
        else:
            devices = list(range(torch.cuda.device_count())) if cuda and torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            np_module.random.set_state(np_state)


@contextmanager
def rng_guard(*, cuda: bool | None = None) -> Iterator[None]:
    """Run a block with the Python, NumPy and torch random states saved and restored around it.

    The public form of the guard every registry loader runs under, for callers that build or fit a
    network outside the registry (a warm-up's dummy fit). ``cuda=None`` forks the CUDA generators
    whenever torch is importable and CUDA is available, so a block that creates the CUDA context
    itself is covered too; ``True`` and ``False`` force the choice. Forking the CUDA generators
    reads their state through ``torch.cuda.get_rng_state``, which creates the CUDA context on every
    visible device when none exists yet, so pass ``False`` (or ``torch.cuda.is_initialized()``) for
    a block that must stay off the GPU on a CUDA host.
    """
    if cuda is None:
        try:
            import torch
        except ImportError:
            cuda = False
        else:
            cuda = bool(torch.cuda.is_available())
    with _rng_guard(cuda=cuda):
        yield


def _run_loader(key: WeightsKey, loader: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``loader`` under the RNG guard; returns the value and the load time in seconds."""
    torch_module = sys.modules.get("torch")
    cuda_initialized = bool(torch_module is not None and torch_module.cuda.is_initialized())
    start = time.perf_counter()
    with _rng_guard(cuda=key.device == "cuda" or cuda_initialized):
        value = loader()
    return value, time.perf_counter() - start


def get_or_load(
    key: WeightsKey,
    loader: Callable[[], Any],
    *,
    stage: LoadedBy = "fit",
    source: dict[str, Any] | None = None,
) -> Any:
    """The registered object for ``key``; on a miss run ``loader`` under the RNG guard and register it.

    The lock is held while the loader runs, so concurrent callers of the same key share one load.
    When ``capacity()`` is ``0`` the value is returned without being stored. A raising loader
    registers nothing and its exception propagates.

    Parameters
    ----------
    key : WeightsKey
        Registry key.
    loader : callable
        Zero-argument callable building the object (see the module's loader contract).
    stage : LoadedBy
        Stage recorded as the entry's ``loaded_by`` on a miss.
    source : dict, optional
        Provenance recorded on the entry (repo id, filename, revision, snapshot sha).
    """
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None:
            _ENTRIES.move_to_end(key)
            entry.hits += 1
            _STATS["hits"] += 1
            logger.debug("Shared weights hit for %s (stage %s).", key.short(), stage)
            return entry.value
        return _load_entry(key, loader, stage=stage, source=source).value


def _load_entry(
    key: WeightsKey,
    loader: Callable[[], Any],
    *,
    stage: LoadedBy,
    source: dict[str, Any] | None,
) -> WeightsEntry:
    """The miss path of :func:`get_or_load`; the caller holds the lock.

    Runs ``loader`` under the RNG guard, counts the miss and builds the entry with its real load
    time, byte estimate and provenance. The entry is stored (and the LRU trimmed) only when
    ``capacity()`` is positive; it is returned either way, so a transient entry still reports what
    the loader did.
    """
    value, load_time_s = _run_loader(key, loader)
    _STATS["misses"] += 1
    entry = WeightsEntry(
        key=key,
        value=value,
        loaded_by=stage,
        load_time_s=load_time_s,
        n_bytes=_estimate_bytes(value),
        source=dict(source or {}),
    )
    if capacity() > 0:
        _ENTRIES[key] = entry
        _evict_to(capacity())
    logger.info("Shared weights %s loaded by %s in %.2fs.", key.short(), stage, load_time_s)
    return entry


def contains(key: WeightsKey) -> bool:
    """Whether ``key`` is registered (no LRU touch)."""
    with _LOCK:
        return key in _ENTRIES


def loaded_by(key: WeightsKey) -> LoadedBy | None:
    """Stage that loaded ``key``, or ``None`` when it is not registered."""
    with _LOCK:
        entry = _ENTRIES.get(key)
        return None if entry is None else entry.loaded_by


def peek(key: WeightsKey) -> WeightsEntry | None:
    """The entry for ``key`` without touching the LRU order or the hit counters."""
    with _LOCK:
        return _ENTRIES.get(key)


def release(*, keys: Iterable[WeightsKey] | None = None) -> int:
    """Drop the registry's references to all entries (or to ``keys``); returns how many were dropped.

    Live estimators keep their own reference, so the memory is freed once the last of them is
    gone. When something was dropped, the garbage collector runs and, if CUDA is initialized, the
    caching allocator returns freed blocks to the device.
    """
    with _LOCK:
        if keys is None:
            dropped = len(_ENTRIES)
            _ENTRIES.clear()
        else:
            dropped = 0
            for key in list(keys):
                if _ENTRIES.pop(key, None) is not None:
                    dropped += 1
    if dropped:
        _free_memory()
    return dropped


def _free_memory() -> None:
    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_module.cuda.empty_cache()


def report() -> dict[str, Any]:
    """JSON-able snapshot: ``capacity``, ``stats`` and every entry's metadata."""
    with _LOCK:
        return {
            "capacity": capacity(),
            "stats": dict(_STATS),
            "entries": [entry.to_metadata() for entry in _ENTRIES.values()],
        }


def reset_stats() -> None:
    """Zero the counters (including each entry's ``hits``); the entries themselves stay registered."""
    with _LOCK:
        _STATS.clear()
        _STATS.update(_fresh_stats())
        for entry in _ENTRIES.values():
            entry.hits = 0


def _estimate_bytes(value: Any) -> int | None:
    """Tensor bytes of a module or a mapping of tensors; ``None`` for anything else."""
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return None
    if isinstance(value, torch_module.nn.Module):
        return tensor_bytes(value)
    if isinstance(value, dict) and value and all(isinstance(v, torch_module.Tensor) for v in value.values()):
        return tensor_bytes(value)
    return None


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    import torch

    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, torch.nn.Module):
        yield from obj.parameters()
        yield from obj.buffers()
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    else:
        for item in obj:
            yield from _iter_tensors(item)


def tensor_bytes(obj: torch.nn.Module | dict[str, torch.Tensor] | Iterable[Any]) -> int:
    """Bytes held by the tensors of ``obj``, each underlying storage counted once.

    ``obj`` is a module (parameters and buffers), a state dict, a tensor or an iterable of those.
    Tensors that view the same storage (tied weights, slices of one buffer) are counted once at the
    storage's size, which is what the device actually holds. Tensors without a real storage
    address (a module built under ``torch.device("meta")``, whose storages all report data pointer
    0) are identified by the tensor object instead, so a meta-built module reports the bytes it
    will hold after ``to_empty``.
    """
    seen: set[Any] = set()
    total = 0
    for tensor in _iter_tensors(obj):
        try:
            storage = tensor.untyped_storage()
            data_ptr = storage.data_ptr()
            ident: Any = id(tensor) if data_ptr == 0 else (tensor.device, data_ptr)
            n_bytes = storage.nbytes()
        except (RuntimeError, NotImplementedError, AttributeError):
            ident = id(tensor)
            n_bytes = tensor.numel() * tensor.element_size()
        if ident not in seen:
            seen.add(ident)
            total += n_bytes
    return total


def shallow_copy(obj: Any) -> Any:
    """A new instance of ``type(obj)`` sharing every attribute; ``__init__`` and ``__getstate__`` are not run."""
    cls = type(obj)
    try:
        new = object.__new__(cls)
    except TypeError:  # built-in types such as SimpleNamespace refuse object.__new__
        new = cls.__new__(cls)
    new.__dict__.update(obj.__dict__)
    return new


@dataclass(frozen=True)
class SharedWeightsClassSettings(ClassSettings):
    """Process-wide switch for one wrapper class that shares weights through the registry.

    ``share_weights=False`` makes every model of that class load its own network through the
    library path and pickle it as before. Set through
    ``TabularPredictor.fit(model_class_settings={"<ag_key>": {"share_weights": False}})``; the
    snapshot travels with each model, so fold workers see the same value. ``AbstractTorchModel``
    gives a class that declares ``shared_weights`` without settings of its own this class, and a
    class with its own settings extends it (``TabPFNClassSettings``).
    """

    share_weights: bool = True


__all__ = [
    "CAPACITY_ENV_VAR",
    "DEFAULT_CAPACITY",
    "LEGACY_CAPACITY_ENV_VAR",
    "LoadedBy",
    "SharedWeightsClassSettings",
    "WeightsEntry",
    "WeightsKey",
    "capacity",
    "contains",
    "get_or_load",
    "loaded_by",
    "make_key",
    "normalize_device",
    "peek",
    "release",
    "report",
    "reset_stats",
    "rng_guard",
    "set_capacity",
    "shallow_copy",
    "tensor_bytes",
]
