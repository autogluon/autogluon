"""Shared pretrained weights: the library's own network loader, memoized per process.

Every foundation-model library builds its network inside one call its ``fit`` makes: a method on the
estimator (tabpfn ``_initialize_model_variables``, tabicl ``_load_model``), a module function or a
classmethod (``Tab2D.from_pretrained``). A model class names that call and the inputs that decide
which network it builds::

    class TabICLModel(AbstractTorchModel):
        shared_weights = SharedWeights(
            loader="tabicl.sklearn:TabICLClassifier._load_model",
            key=("checkpoint_version",),
            disabled_by=("kv_cache", "model_path"),
        )

The first call in the process runs the library's code and registers what it produced (the return
value, and for a method the attributes it set on the estimator) in the process-wide registry
(:mod:`._shared_weights_registry`); every later call with the same inputs and device reuses it, so
the eight bagged children and the refit child share one network. The library's own code still runs
on the miss, so a shared fit equals an unshared fit, downloads and cache directories stay the
library's business. The fitted model pickles without the network, as the estimator's class and
attributes (the library's own pickle protocol is not run: it expects the network), and the load puts
the network back. The device of an entry is the one the loader's inputs name (``device_param``) or, for a loader
without a device input (the library builds on the CPU and moves the network afterwards), the fit's
device from ``num_gpus``; a network built elsewhere is moved there once, as the library would.
A fine-tuning model declares ``copy_per_fit``: the registry keeps the pristine network and every
fit trains a deep copy.

:class:`AbstractTorchModel` calls into this module from six places: ``fit`` (installs the wrapped
loader once, records what the fit took), ``__getstate__`` (a pickle without the shared network),
``__setstate__`` and ``load`` (put it back), ``set_device`` (take the entry for the new device),
``prepare_for_inference`` (attach and eval) and ``_get_memory_size``. A wrapper's ``_fit`` needs no
change. Sharing is off for a fit with ``ag_args_fit={"share_pretrained_weights": False}``, for a
class with the ``share_weights`` class setting off, for a configuration named in ``disabled_by``,
and for every model class without a declaration.

What a library has to offer is one loader call whose inputs are estimator parameters. A library that
loads inside ``__init__`` has no such call; its wrapper keeps an adapter until upstream exposes one.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import importlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from autogluon.core.models.abstract import _object_graph as graph
from autogluon.core.models.abstract._shared_weights_registry import (
    WeightsKey,
    contains,
    get_or_load,
    loaded_by,
    make_key,
    normalize_device,
    peek,
    release,
    tensor_bytes,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

LoaderKind = Literal["method", "function"]


# --- the declaration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedWeights:
    """What a model class declares so that its library's network is built once per process.

    Parameters
    ----------
    loader
        The library call that builds the network, as ``"package.module:Class.method"`` for a method
        the estimator's ``fit`` calls, or ``"package.module:function"`` (also ``Class.classmethod``)
        for a function or classmethod that returns the network. Several when the library has one
        estimator class per task (a classifier and a regressor).
    key
        The loader's inputs that decide which network is built: attribute names of the estimator for
        a method, argument names for a function. The device is part of the key by itself. Inputs
        that only change how the network is used (an ensemble size, a seed) stay out of the key.
    disabled_by
        Inputs under which the library writes into or casts its network, so the fit keeps its own
        copy: an input name (disables when truthy) or a predicate over the inputs.
    copy_per_fit
        True for a fine-tuning model: every fit, the one that built the entry included, gets a deep
        copy of the cached network and trains that; the registry keeps the pristine one where the
        loader built it, and the fitted model pickles its own copy.
    device_param
        The input naming the device the network is built on (``"device"`` for most libraries); the
        loader is keyed on its device type. A loader without it, or one given ``None`` or
        ``"auto"``, is keyed on the device the model's ``_fit`` recorded on ``self.device`` before
        the loader ran, else the fit's device (``num_gpus``), else the CPU.
    """

    loader: str | tuple[str, ...]
    key: tuple[str, ...] = ()
    disabled_by: tuple[str | Callable[[Mapping[str, Any]], bool], ...] = ()
    copy_per_fit: bool = False
    device_param: str = "device"

    def loaders(self) -> tuple[str, ...]:
        return (self.loader,) if isinstance(self.loader, str) else tuple(self.loader)

    def allows(self, inputs: Mapping[str, Any]) -> bool:
        """Evaluate ``disabled_by`` on the loader's inputs."""
        for rule in self.disabled_by:
            if isinstance(rule, str):
                if inputs.get(rule):
                    return False
            elif rule(dict(inputs)):
                return False
        return True

    def disabled_examples(self) -> list[dict[str, Any]]:
        """One input dict per name in ``disabled_by`` that must switch sharing off (for tests)."""
        return [{rule: True} for rule in self.disabled_by if isinstance(rule, str)]


def validate_declaration(spec: SharedWeights, cls: type) -> None:
    where = f"{cls.__name__}.shared_weights"
    if not isinstance(spec, SharedWeights):
        raise TypeError(f"{where} must be a SharedWeights, got {type(spec).__name__}")
    for ref in spec.loaders():
        if ":" not in ref:
            raise TypeError(f"{where}.loader entries look like 'package.module:Class.method', got {ref!r}")
    for rule in spec.disabled_by:
        if not (isinstance(rule, str) or callable(rule)):
            raise TypeError(f"{where}.disabled_by entries are input names or predicates, got {rule!r}")


# --- the loader reference -------------------------------------------------------------------------


@dataclass(frozen=True)
class LoaderRef:
    """A parsed ``"package.module:Qualified.name"`` reference to the library call."""

    module: str
    qualname: str

    @classmethod
    def parse(cls, ref: str) -> LoaderRef:
        module, _, qualname = ref.partition(":")
        if not module or not qualname:
            raise ValueError(f"a loader reference looks like 'package.module:Class.method', got {ref!r}")
        return cls(module, qualname)

    @property
    def text(self) -> str:
        return f"{self.module}:{self.qualname}"

    @property
    def library(self) -> str:
        return self.module.split(".")[0]

    def owner_and_name(self) -> tuple[Any, str]:
        """The object holding the attribute (a module or a class) and the attribute name."""
        module = importlib.import_module(self.module)
        parts = self.qualname.split(".")
        owner: Any = module
        for part in parts[:-1]:
            owner = getattr(owner, part)
        return owner, parts[-1]

    def kind(self) -> LoaderKind:
        """``"method"`` for a plain function on a class (the estimator is its first argument), else ``"function"``."""
        owner, name = self.owner_and_name()
        raw = inspect.getattr_static(owner, name)
        if inspect.isclass(owner) and inspect.isfunction(raw):
            return "method"
        return "function"


# --- payloads and the scope -----------------------------------------------------------------------


@dataclass
class MethodPayload:
    """What a method loader produced: its return value and the attributes it set on the estimator."""

    result: Any
    attributes: dict[str, Any]

    def components(self) -> list[Any]:
        return [self, *graph.iter_modules(self.attributes)]


@dataclass
class FunctionPayload:
    """What a function loader returned, with the arguments that rebuild it in another process."""

    result: Any
    arguments: dict[str, Any]

    def components(self) -> list[Any]:
        return [self, *graph.iter_modules(self.result)]


Payload = MethodPayload | FunctionPayload


@dataclass
class FitScope:
    """The fit (or reload) a loader call belongs to: who fits, whether it shares, what it took."""

    model: Any
    enabled: bool | None = None  # decided on the first loader call, once the model's params are initialized
    device: str | None = None
    key: WeightsKey | None = None
    loader: LoaderRef | None = None
    present_before: bool | None = None
    stage: str = "fit"

    def shares(self) -> bool:
        if self.enabled is None:
            self.enabled = bool(self.model._shares_weights())
        return self.enabled


_SCOPE: contextvars.ContextVar[FitScope | None] = contextvars.ContextVar("shared_weights_scope", default=None)


@contextlib.contextmanager
def scope(
    model: Any, *, enabled: bool | None = None, device: str | None = None, stage: str = "fit"
) -> Iterator[FitScope]:
    """Attribute the loader calls made inside the block to ``model``."""
    current = FitScope(model=model, enabled=enabled, device=device, stage=stage)
    token = _SCOPE.set(current)
    try:
        yield current
    finally:
        _SCOPE.reset(token)


# --- installing the memo --------------------------------------------------------------------------

_INSTALLED: dict[str, tuple[LoaderRef, SharedWeights, Callable[..., Any]]] = {}
_ORIGINAL_ATTR = "__shared_weights_original__"


def install(spec: SharedWeights) -> None:
    """Wrap every loader of ``spec`` once per process; a wrapper already in place is left alone."""
    for ref_text in spec.loaders():
        if ref_text in _INSTALLED:
            continue
        ref = LoaderRef.parse(ref_text)
        owner, name = ref.owner_and_name()
        raw = inspect.getattr_static(owner, name)
        kind = ref.kind()
        if isinstance(raw, classmethod):
            original = raw.__func__
            wrapped = _wrap_function(ref, spec, original, bound_class=True)
            setattr(owner, name, classmethod(wrapped))
        elif isinstance(raw, staticmethod):
            original = raw.__func__
            wrapped = _wrap_function(ref, spec, original, bound_class=False)
            setattr(owner, name, staticmethod(wrapped))
        elif kind == "method":
            original = raw
            wrapped = _wrap_method(ref, spec, original)
            setattr(owner, name, wrapped)
        else:
            original = getattr(owner, name)
            wrapped = _wrap_function(ref, spec, original, bound_class=False)
            setattr(owner, name, wrapped)
        _INSTALLED[ref_text] = (ref, spec, original)
        logger.debug("Shared weights: wrapped %s", ref_text)


def installed_original(ref_text: str) -> Callable[..., Any] | None:
    """The library callable a wrapped loader replaced, or ``None`` when it is not wrapped."""
    entry = _INSTALLED.get(ref_text)
    return None if entry is None else entry[2]


def _inputs_of_estimator(estimator: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(estimator, name, None) for name in names}


def _all_inputs_of_estimator(estimator: Any) -> dict[str, Any]:
    try:
        return dict(estimator.get_params(deep=False))
    except Exception:
        return {k: v for k, v in vars(estimator).items() if not k.endswith("_")}


def _device_type(value: Any, fallback: str | None) -> str:
    if isinstance(value, (list, tuple)):
        return "multi"
    if value is None or value == "auto":
        return normalize_device(fallback) if fallback else "cpu"
    return normalize_device(value)


def _scope_device(active: FitScope | None) -> str | None:
    """The device a loader without a device input is keyed on: the fitting model's ``device`` when its ``_fit`` set it, else the fit's."""
    if active is None:
        return None
    return getattr(active.model, "device", None) or active.device


def _key_for(ref: LoaderRef, inputs: Mapping[str, Any], device_type: str) -> WeightsKey:
    encoded = json.dumps({k: _stable(v) for k, v in inputs.items()}, sort_keys=True, default=str)
    return make_key(ref.library, encoded, ref.qualname, device_type)


def _stable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in value.items()}
    return repr(value)


def _file_stamp(candidates: Iterator[Any]) -> dict[str, Any] | None:
    """``(path, size, mtime)`` of the first existing file among ``candidates`` (a checkpoint the loader read)."""
    for value in candidates:
        if isinstance(value, (str, os.PathLike)):
            path = os.fspath(value)
            if os.path.isfile(path):
                stat = os.stat(path)
                return {"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return None


def _stale(key: WeightsKey) -> bool:
    """Whether the entry's checkpoint file changed on disk since it was read (size or modification time)."""
    entry = peek(key)
    stamp = (entry.source or {}).get("file_stamp") if entry is not None else None
    if not stamp:
        return False
    try:
        stat = os.stat(stamp["path"])
    except OSError:
        return True
    return stat.st_size != stamp["size"] or stat.st_mtime_ns != stamp["mtime_ns"]


def _take(
    key: WeightsKey,
    build: Callable[[], Payload],
    *,
    source_paths: Callable[[Payload], Iterator[Any]],
    place: bool = True,
) -> Payload:
    """The registry payload for ``key``: a hit, or one build (RNG-guarded by the registry) with its file stamp.

    With ``place`` a built network is moved to the key's device when the loader built it elsewhere
    (a library that builds on the CPU and moves the network in its ``fit`` afterwards), so the
    entry lives where its key says and a rebuild in another process needs no library call after it.
    """
    active = _SCOPE.get()
    if _stale(key):
        logger.warning("Shared weights: the checkpoint behind %s changed on disk; rebuilding", key.short())
        release(keys=[key])
    present = contains(key)
    holder: dict[str, Any] = {}

    def loader() -> Payload:
        payload = build()
        if place:
            _place(payload, key.device)
        holder["stamp"] = _file_stamp(source_paths(payload))
        return payload

    stage = active.stage if active is not None else "load"
    payload = get_or_load(key, loader, stage=stage, source={"loader": key.variant})
    if "stamp" in holder and holder["stamp"] is not None:
        entry = peek(key)
        if entry is not None:
            entry.source = {**(entry.source or {}), "file_stamp": holder["stamp"]}
    if active is not None and active.key is None:
        active.key = key
        active.present_before = present
    return payload


def _place(payload: Payload, device_type: str) -> None:
    """Move the modules of a freshly built payload to ``device_type`` when the loader left them elsewhere."""
    for module in payload.components()[1:]:
        if graph.payload_device(module) != device_type:
            module.to(device_type)


def _hand_out(payload: Payload, spec: SharedWeights) -> Payload:
    """What a fit gets: the shared payload, or its own deep copy (where the loader built it) for a fine-tuning model."""
    return copy.deepcopy(payload) if spec.copy_per_fit else payload


def _spec_now(ref: LoaderRef, installing: SharedWeights, active: FitScope | None) -> SharedWeights:
    """The declaration that governs a loader call: the fitting model's when it names this loader, else the installing one."""
    if active is not None:
        spec = getattr(active.model, "shared_weights", None)
        if spec is not None and ref.text in spec.loaders():
            return spec
    return installing


def _wrap_method(ref: LoaderRef, installing: SharedWeights, original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(estimator: Any, *args: Any, **kwargs: Any) -> Any:
        active = _SCOPE.get()
        if active is not None and not active.shares():
            return original(estimator, *args, **kwargs)
        spec = _spec_now(ref, installing, active)
        all_inputs = _all_inputs_of_estimator(estimator)
        if not spec.allows(all_inputs):
            return original(estimator, *args, **kwargs)
        device_type = _device_type(
            getattr(estimator, spec.device_param, all_inputs.get(spec.device_param)), _scope_device(active)
        )
        if device_type == "multi":
            return original(estimator, *args, **kwargs)
        key = _key_for(ref, _inputs_of_estimator(estimator, spec.key), device_type)
        if active is not None and active.loader is None:
            active.loader = ref
            active.device = device_type

        def build() -> MethodPayload:
            before = {name: id(value) for name, value in vars(estimator).items()}
            result = original(estimator, *args, **kwargs)
            after = vars(estimator)
            changed = {name: value for name, value in after.items() if name not in before or before[name] != id(value)}
            return MethodPayload(result=result, attributes=changed)

        def source_paths(payload: MethodPayload) -> Iterator[Any]:
            yield from _inputs_of_estimator(estimator, spec.key).values()
            yield from payload.attributes.values()

        payload = _take(key, build, source_paths=source_paths, place=not spec.copy_per_fit)
        given = _hand_out(payload, spec)
        # Every estimator, the one that built included, gets its own plain containers around the shared leaves.
        for name, value in given.attributes.items():
            setattr(estimator, name, _own_container(value))
        return given.result

    wrapped.__name__ = getattr(original, "__name__", "loader")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(wrapped, _ORIGINAL_ATTR, original)
    return wrapped


def _wrap_function(
    ref: LoaderRef, installing: SharedWeights, original: Callable[..., Any], *, bound_class: bool
) -> Callable[..., Any]:
    signature = inspect.signature(original)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        active = _SCOPE.get()
        if active is not None and not active.shares():
            return original(*args, **kwargs)
        spec = _spec_now(ref, installing, active)
        arguments = _call_arguments(signature, args, kwargs, bound_class=bound_class)
        if not spec.allows(arguments):
            return original(*args, **kwargs)
        device_type = _device_type(arguments.get(spec.device_param), _scope_device(active))
        if device_type == "multi":
            return original(*args, **kwargs)
        key = _key_for(ref, {name: arguments.get(name) for name in spec.key}, device_type)
        if active is not None and active.loader is None:
            active.loader = ref
            active.device = device_type

        def build() -> FunctionPayload:
            return FunctionPayload(
                result=original(*args, **kwargs), arguments={k: _stable(v) for k, v in arguments.items()}
            )

        payload = _take(key, build, source_paths=lambda p: iter(p.arguments.values()), place=not spec.copy_per_fit)
        return _hand_out(payload, spec).result

    wrapped.__name__ = getattr(original, "__name__", "loader")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(wrapped, _ORIGINAL_ATTR, original)
    return wrapped


def _call_arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any], *, bound_class: bool
) -> dict[str, Any]:
    """The call's arguments by parameter name, defaults applied, ``**kwargs`` flattened, ``cls`` dropped.

    A ``*args`` parameter is kept under its own name only when the call filled it; the rebuild
    passes every entry back as a keyword, which such a call cannot take.
    """
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments: dict[str, Any] = {}
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if bound_class and index == 0:
            continue
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            if value:
                arguments[name] = list(value)
        else:
            arguments[name] = value
    return arguments


def _own_container(value: Any) -> Any:
    """A per-estimator copy of a plain container whose leaves (modules, tensors) stay shared."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


# --- what AbstractTorchModel calls ----------------------------------------------------------------


@dataclass
class SharedState:
    """The pickled record of a fit that shared: which entry, where it sat in the estimator, what the pickle moved."""

    key: WeightsKey
    loader: str
    paths: list[tuple[graph.Path, int]] = field(default_factory=list)
    moved: list[graph.Path] = field(default_factory=list)
    arguments: dict[str, Any] | None = None  # a function loader's call, to rebuild in another process
    present_before_fit: bool | None = None
    device: str | None = None
    component_ids: list[int] = field(default_factory=list)  # identities of the payload and its modules while attached
    attributes: tuple[str, ...] = ()  # what a method loader set on the fresh estimator at fit time


def fit_device(num_gpus: float | None) -> str | None:
    """The device type a fit with ``num_gpus`` runs on: ``"cuda"`` for a GPU allocation, ``"cpu"`` for none, ``None`` unknown."""
    if num_gpus is None:
        return None
    return "cuda" if num_gpus > 0 else "cpu"


def fit(model: Any, run: Callable[[], Any], *, device: str | None = None) -> Any:
    """Run ``run`` (the base ``fit``) with the model's loaders wrapped and record what the fit took.

    ``device`` is the fit's device type (see :func:`fit_device`); a loader whose inputs name no
    device is keyed on it.
    """
    spec = model.shared_weights
    if spec is None:
        return run()
    # The wrappers are installed up front (idempotent, and inert without an enabled scope); whether
    # this fit shares is decided on the first loader call, once ``fit`` has initialized the model's
    # auxiliary parameters.
    install(spec)
    with scope(model, device=device) as current:
        out = run()
    payload = peek(current.key) if current.key is not None else None
    if payload is None:
        # Nothing to record: the fit opted out or no loader ran, or the entry left the registry before the
        # fit ended (capacity 0). The model owns its network like an unshared fit.
        model._shared_state = None
        return out
    model._shared_state = SharedState(
        key=current.key,
        loader=current.loader.text,
        arguments=payload.value.arguments if isinstance(payload.value, FunctionPayload) else None,
        present_before_fit=current.present_before,
        device=current.device,
        component_ids=[id(part) for part in payload.value.components()],
        attributes=tuple(payload.value.attributes) if isinstance(payload.value, MethodPayload) else (),
    )
    return out


def shares(model: Any) -> bool:
    """Whether a fit of ``model`` takes its network from the registry (declaration, aux params, class setting)."""
    if model.shared_weights is None:
        return False
    aux = model.aux_params
    if aux.save_pretrained_weights or not getattr(aux, "share_pretrained_weights", True):
        return False
    settings = type(model).get_class_settings()
    return settings is None or bool(getattr(settings, "share_weights", True))


def components(model: Any) -> list[Any]:
    """The registry objects a fitted, shared model references, or ``[]``."""
    state = model._shared_state
    if state is None:
        return []
    entry = peek(state.key)
    if entry is None:
        return []
    return entry.value.components()


def owns_network(model: Any) -> bool:
    """Whether the fitted estimator holds its own network: an unshared fit, or a fine-tuning model's copy.

    Such a model pickles, moves and reports its network as a model without a declaration does; the
    registry entry it copied from is not part of its state.
    """
    state = model._shared_state
    return state is None or bool(model.shared_weights is not None and model.shared_weights.copy_per_fit)


def attached(model: Any) -> bool:
    state = model._shared_state
    return state is None or not state.paths


def _rebuild_from_attributes(cls: type, attributes: dict[str, Any]) -> Any:
    """Unpickle a weightless estimator: its class and attributes, without running the library's ``__setstate__``."""
    estimator = cls.__new__(cls)
    estimator.__dict__.update(attributes)
    return estimator


class _Attributes:
    """Pickles an estimator copy as its class and attributes, bypassing the library's pickle protocol.

    The copy has no network, which is outside what a library's own ``__getstate__`` or
    ``__setstate__`` expects (one exports the network's configuration, another reloads it); the
    weightless pickle is the estimator's attributes, tensors on the CPU, and the load puts the shared
    network back through :func:`ensure_network`.
    """

    def __init__(self, estimator: Any) -> None:
        self.cls = type(estimator)
        self.attributes = dict(vars(estimator))

    def __reduce__(self):
        return _rebuild_from_attributes, (self.cls, self.attributes)


def getstate(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    """The model's state for the pickle: the estimator without the shared network, its other tensors on the CPU."""
    shared = model._shared_state
    if owns_network(model) or model.model is None or shared.paths or not shared.component_ids:
        return state
    estimator, paths, moved = graph.detach_shared(model.model, shared_ids=shared.component_ids)
    record = copy.copy(shared)
    record.paths = paths
    record.moved = moved
    state["model"] = _Attributes(estimator) if hasattr(estimator, "__dict__") else estimator
    state["_shared_state"] = record
    return state


def ensure_network(model: Any, device: str | None = None) -> None:
    """Put the shared network back into a reloaded estimator on ``device`` (the fit device by default).

    The weightless pickle holds the estimator's attributes (the library's own ``__setstate__`` does
    not run, see :class:`_Attributes`), so the recorded paths are set here. A missing entry (another
    process, another device) is rebuilt through the wrapped loader: a method loader re-runs on the
    estimator, a function loader re-runs with its recorded arguments.
    """
    shared = model._shared_state
    if owns_network(model) or not shared.paths or model.model is None:
        return
    device_type = normalize_device(device or shared.device or "cpu")
    if device_type == "cuda" and not _cuda_available():
        device_type = "cpu"
    if shared.device is not None and normalize_device(shared.device) != device_type:
        model.model = graph.rewrite_devices(model.model, shared.device, device_type)
        # The rewrite re-keys the per-device caches the recorded paths step through (tabpfn keeps its
        # network under its ``torch.device``), so the paths follow it before the attach below.
        indices = [index for _, index in shared.paths]
        rewritten = graph.rewrite_path_devices([path for path, _ in shared.paths], shared.device, device_type)
        shared.paths = list(zip(rewritten, indices, strict=True))
        shared.moved = graph.rewrite_path_devices(shared.moved, shared.device, device_type)
    if shared.moved:
        model.model = graph.move_to(model.model, shared.moved, device_type)
    key = shared.key.replace(device=device_type)
    ref = LoaderRef.parse(shared.loader)
    install(model.shared_weights)
    with scope(model, enabled=True, device=device_type, stage="load"):
        payload = peek(key)
        if payload is None or _stale(key):
            _rebuild(model, ref, shared, device_type)
            payload = peek(key)
        _complete_attributes(payload.value, shared, model.model)
        parts = payload.value.components()
    model.model = graph.attach_shared(model.model, shared.paths, parts)
    shared.paths = []
    shared.moved = []
    shared.key = key
    shared.device = device_type
    shared.component_ids = [id(part) for part in parts]


def _complete_attributes(payload: Payload, shared: SharedState, estimator: Any) -> None:
    """Add to a payload built on a reloaded estimator the attributes the fit-time build recorded but this one missed.

    The build records what the loader set by identity. On a fresh estimator that is everything; on a
    reloaded one, a value the loader re-sets to the same singleton (a dtype, a bool, ``None``) looks
    unchanged and would be missing for the next estimator that hits this entry. The estimator holds the
    loader's value either way.
    """
    if not isinstance(payload, MethodPayload):
        return
    present = vars(estimator)
    for name in shared.attributes:
        if name not in payload.attributes and name in present:
            payload.attributes[name] = present[name]


def _rebuild(model: Any, ref: LoaderRef, shared: SharedState, device_type: str) -> None:
    owner, name = ref.owner_and_name()
    if ref.kind() == "method":
        getattr(model.model, name)()
        return
    original = installed_original(ref.text)
    arguments = dict(shared.arguments or {})
    if model.shared_weights.device_param in arguments:
        arguments[model.shared_weights.device_param] = device_type
    wrapped = getattr(owner, name)
    if original is None:
        raise RuntimeError(f"{ref.text} is not wrapped; cannot rebuild the shared network")
    wrapped(**arguments)


def set_device(model: Any, device: str) -> bool:
    """Move a shared model to ``device`` by swapping registry entries; returns False for a model that owns its network."""
    shared = model._shared_state
    if owns_network(model) or model.model is None:
        return False
    device_type = normalize_device(device)
    if not attached(model):
        ensure_network(model, device_type)
        return True
    if shared.device is not None and normalize_device(shared.device) == device_type:
        return True
    parts = components(model)
    paths = graph.find_shared(model.model, parts)
    model.model = graph.attach_shared(model.model, paths, [None] * len(parts))
    shared.paths = paths
    ensure_network(model, device_type)
    return True


def shared_modules(model: Any) -> list[torch.nn.Module]:
    """The shared modules of an attached model (``components()`` minus the payload record itself)."""
    if owns_network(model) or not attached(model):
        return []
    return components(model)[1:]


def prepare_for_inference(model: Any) -> None:
    if owns_network(model):
        return
    ensure_network(model)
    modules = shared_modules(model)
    for module in modules:
        module.eval()
    if modules and graph.payload_device(modules) == "cuda":
        import torch

        torch.cuda.synchronize()


def memory_size(model: Any) -> int | None:
    """Pickled size plus the shared tensors' bytes, or ``None`` for a model that owns its network."""
    if owns_network(model) or not attached(model):
        return None
    return model._get_pickled_size() + tensor_bytes(shared_modules(model))


def info(model: Any) -> dict[str, Any] | None:
    """``info["shared_weights"]``: the key without host paths and the load provenance, or ``None`` for an unshared fit."""
    shared = model._shared_state
    if shared is None:
        return None
    key = shared.key
    entry = peek(key)
    stamp = (entry.source or {}).get("file_stamp") if entry is not None else None
    return {
        **key.to_dict(),
        "inputs": json.loads(key.checkpoint),
        "checkpoint": os.path.basename(stamp["path"]) if stamp else None,
        "loaded_by": loaded_by(key),
        "present_before_fit": shared.present_before_fit,
    }


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


#: The keys of ``get_info()["shared_weights"]``.
INFO_KEYS: tuple[str, ...] = (
    "library",
    "checkpoint",
    "variant",
    "device",
    "dtype",
    "flags",
    "inputs",
    "loaded_by",
    "present_before_fit",
)

__all__ = [
    "INFO_KEYS",
    "FitScope",
    "FunctionPayload",
    "LoaderRef",
    "MethodPayload",
    "SharedState",
    "SharedWeights",
    "attached",
    "components",
    "ensure_network",
    "fit",
    "fit_device",
    "getstate",
    "info",
    "install",
    "installed_original",
    "memory_size",
    "owns_network",
    "prepare_for_inference",
    "scope",
    "set_device",
    "shared_modules",
    "shares",
    "validate_declaration",
]
