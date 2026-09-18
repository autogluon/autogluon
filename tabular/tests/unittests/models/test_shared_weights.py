"""Shared pretrained weights through a memoized library loader, on a fake library and a fake torch model.

The fake library below stands for tabicl, tabpfn or any library that builds its network inside one
call its ``fit`` makes: ``_Estimator._load_model`` (a method loader) and ``build_network`` (a function
loader). The fake models declare them the way a real wrapper does; nothing in their ``_fit`` knows
about sharing.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

import autogluon.core.models.abstract._shared_weights_registry as registry
from autogluon.core.models.abstract import _object_graph as graph
from autogluon.core.models.abstract import shared_weights as sw
from autogluon.core.models.abstract.shared_weights import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

torch = pytest.importorskip("torch")

# --- the fake library -------------------------------------------------------------------------------

BUILDS: list[tuple] = []


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(6.0).reshape(2, 3) / 10)
            self.linear.bias.zero_()


def build_network(checkpoint: str, device: str = "cpu", *, precision: str = "float32") -> _Net:
    """A module-level loader (like LimiX's ``load_model`` or Causilo's ``load_pretrained_model``)."""
    BUILDS.append(("function", checkpoint, device, precision))
    return _Net().to(device)


def build_network_on_cpu(checkpoint: str) -> _Net:
    """A loader without a device input: the library builds on the CPU and moves the network in its ``fit``."""
    BUILDS.append(("deviceless", checkpoint))
    return _Net()


class _Estimator:
    """A library estimator that builds its network in ``_load_model``, called by ``fit`` and ``__setstate__``."""

    def __init__(self, checkpoint="default.ckpt", device="cpu", n_estimators=4, kv_cache=False):
        self.checkpoint = checkpoint
        self.device = device
        self.n_estimators = n_estimators
        self.kv_cache = kv_cache
        self.model_ = None

    def get_params(self, deep=False):
        return {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "n_estimators": self.n_estimators,
            "kv_cache": self.kv_cache,
        }

    def _load_model(self) -> int:
        BUILDS.append(("method", self.checkpoint, self.device))
        self.model_ = _Net().to(self.device)
        self.model_path_ = self.checkpoint
        # Like tabpfn's ``InferenceEngine._models``: the network sits in a dict keyed by its ``torch.device``.
        self.caches_ = {torch.device(self.device): self.model_}
        return 7

    def fit(self, X, y):
        self._load_model()
        self.classes_ = np.unique(y)
        self.context_ = torch.zeros(2, device=self.device)
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class _ExportingEstimator(_Estimator):
    """A library whose pickle protocol exports the network's configuration and reloads it in ``__setstate__``."""

    def __getstate__(self):
        return {"exported": True, "config": self.model_.linear.weight.shape}  # needs the network

    def __setstate__(self, saved):
        raise AssertionError("the library's __setstate__ must not run for a weightless pickle")


class _FunctionEstimator:
    """A library estimator whose constructor takes the network a module-level loader returns."""

    def __init__(self, network, n_estimators=4):
        self.network = network
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class Trainer:
    """A fine-tuning library: ``from_pretrained`` returns the base network every fit then trains."""

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", **options: Any) -> _Net:
        BUILDS.append(("classmethod", path, device, options))
        return _Net().to(device)


# --- the fake torch models ---------------------------------------------------------------------------


class _Base(AbstractTorchModel):
    ag_key = "FAKE"
    ag_name = "Fake"
    _supported_problem_types = ["binary", "multiclass"]

    def _predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)

    def get_device(self) -> str:
        try:
            return super().get_device()
        except NotImplementedError:
            return getattr(self.model, "device", "cpu")

    def _set_device(self, device: str):
        self.model.device = device


class MethodModel(_Base):
    """Declares the estimator's ``_load_model`` as the loader."""

    ag_key = "FAKE-METHOD"
    shared_weights = SharedWeights(
        loader=f"{__name__}:_Estimator._load_model",
        key=("checkpoint",),
        disabled_by=("kv_cache",),
    )

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        hps = self._get_model_params()
        self.model = _Estimator(device="cpu", **hps).fit(X, y)


class ProtocolModel(MethodModel):
    """The estimator has its own pickle protocol, which the weightless pickle bypasses."""

    ag_key = "FAKE-PROTOCOL"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        hps = self._get_model_params()
        self.model = _ExportingEstimator(device="cpu", **hps).fit(X, y)


class FunctionModel(_Base):
    """Declares a module-level loader whose result the estimator takes in its constructor."""

    ag_key = "FAKE-FUNCTION"
    shared_weights = SharedWeights(loader=f"{__name__}:build_network", key=("checkpoint", "precision"))

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        hps = self._get_model_params()
        network = build_network(
            hps.get("checkpoint", "default.ckpt"), device="cpu", precision=hps.get("precision", "float32")
        )
        self.model = _FunctionEstimator(network, n_estimators=hps.get("n_estimators", 4)).fit(X, y)


class FineTuneModel(_Base):
    """Declares a classmethod loader with ``copy_per_fit``: every fit trains its own copy of the base network."""

    ag_key = "FAKE-FINETUNE"
    shared_weights = SharedWeights(loader=f"{__name__}:Trainer.from_pretrained", key=("path",), copy_per_fit=True)

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        network = Trainer.from_pretrained("base.safetensors", device="cpu", dtype="bf16")
        with torch.no_grad():
            network.linear.bias.add_(1.0)  # "fine-tuning"
        self.model = _FunctionEstimator(network).fit(X, y)


class PlainModel(_Base):
    """No declaration: the base class never touches its network."""

    ag_key = "FAKE-PLAIN"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        self.model = _Estimator(device="cpu").fit(X, y)


# --- fixtures ----------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean():
    registry.release()
    registry.reset_stats()
    BUILDS.clear()
    yield
    registry.release()
    registry.reset_stats()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((20, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 20))
    return X, y


def _model(cls, tmp_path, name="m", hyperparameters=None):
    return cls(
        path=str(tmp_path / name) + os.sep, name=name, problem_type="binary", hyperparameters=hyperparameters or {}
    )


def _fit(cls, tmp_path, data, name="m", hyperparameters=None):
    model = _model(cls, tmp_path, name, hyperparameters)
    model.fit(X=data[0], y=data[1], num_cpus=1, num_gpus=0)
    return model


# --- tests -------------------------------------------------------------------------------------------


def test_declaration_is_validated_at_class_creation():
    with pytest.raises(TypeError, match="Class.method"):

        class Bad(_Base):
            shared_weights = SharedWeights(loader="no-colon")


def test_method_loader_is_built_once_and_shared_by_later_fits(tmp_path, data):
    first = _fit(MethodModel, tmp_path, data, "a")
    second = _fit(MethodModel, tmp_path, data, "b")
    assert [b[0] for b in BUILDS] == ["method"], BUILDS
    assert first.model.model_ is second.model.model_
    assert first.model.caches_ is not second.model.caches_, "plain containers are per estimator"
    assert first.model.caches_[torch.device("cpu")] is second.model.model_
    assert first._shared_state is not None and first._shared_state.present_before_fit is False
    assert second._shared_state.present_before_fit is True
    assert registry.report()["stats"]["hits"] == 1
    assert first.model.model_path_ == "default.ckpt"


def test_different_key_inputs_build_different_networks(tmp_path, data):
    a = _fit(MethodModel, tmp_path, data, "a", {"checkpoint": "one.ckpt"})
    b = _fit(MethodModel, tmp_path, data, "b", {"checkpoint": "two.ckpt"})
    c = _fit(MethodModel, tmp_path, data, "c", {"checkpoint": "one.ckpt", "n_estimators": 9})
    assert len(BUILDS) == 2
    assert a.model.model_ is not b.model.model_ and a.model.model_ is c.model.model_


def test_opt_outs_run_the_library_unchanged(tmp_path, data):
    disabled = _fit(MethodModel, tmp_path, data, "d", {"kv_cache": True})
    aux = _fit(MethodModel, tmp_path, data, "e", {"ag_args_fit": {"share_pretrained_weights": False}})
    saving = _fit(MethodModel, tmp_path, data, "f", {"ag_args_fit": {"save_pretrained_weights": True}})
    assert len(BUILDS) == 3 and registry.report()["entries"] == []
    for model in (disabled, aux, saving):
        assert model._shared_state is None
        assert model.model.model_ is not None
        assert model._pickles_pretrained_weights() is True
        assert model.get_info()["shared_weights"] is None
    MethodModel.set_class_settings(share_weights=False)
    try:
        off = _fit(MethodModel, tmp_path, data, "g")
        assert off._shared_state is None and len(BUILDS) == 4
    finally:
        MethodModel.set_class_settings(share_weights=True)
    plain = _fit(PlainModel, tmp_path, data, "p")
    assert plain._shared_state is None and plain.get_info().get("shared_weights") is None


def test_pickle_is_weightless_and_the_reload_takes_the_same_network(tmp_path, data):
    model = _fit(MethodModel, tmp_path, data)
    net = model.model.model_
    blob = pickle.dumps(model)
    assert b"linear.weight" not in blob
    assert model.model.model_ is net, "the live model keeps its network"
    assert model._pickles_pretrained_weights() is False

    loaded = pickle.loads(blob)
    assert loaded.model.model_ is None and loaded.model.caches_[torch.device("cpu")] is None
    loaded.predict_proba(data[0])
    assert loaded.model.model_ is net and loaded.model.caches_[torch.device("cpu")] is net
    assert loaded.model.context_.device.type == "cpu"
    assert len(BUILDS) == 1


def test_reload_in_a_fresh_process_rebuilds_through_the_loader(tmp_path, data):
    model = _fit(MethodModel, tmp_path, data)
    blob = pickle.dumps(model)
    registry.release()  # what a new process looks like
    loaded = pickle.loads(blob)
    loaded.prepare_for_inference()
    assert loaded.model.model_ is not None and loaded.model.caches_[torch.device("cpu")] is loaded.model.model_
    assert [b[0] for b in BUILDS] == ["method", "method"]
    assert not loaded.model.model_.training
    again = pickle.loads(pickle.dumps(model))
    again.prepare_for_inference()
    assert again.model.model_ is loaded.model.model_, "the rebuilt entry is shared again"


def test_autogluon_save_and_load(tmp_path, data):
    model = _fit(MethodModel, tmp_path, data)
    model.save(verbose=False)
    loaded = MethodModel.load(model.path, verbose=False)
    loaded.prepare_for_inference()
    assert loaded.model.model_ is model.model.model_
    assert loaded.get_info()["shared_weights"]["loaded_by"] == "fit"


def test_pickle_bypasses_the_library_protocol_that_expects_the_network(tmp_path, data):
    model = _fit(ProtocolModel, tmp_path, data)
    net = model.model.model_
    loaded = pickle.loads(pickle.dumps(model))
    assert type(loaded.model) is _ExportingEstimator and loaded.model.model_ is None
    loaded.prepare_for_inference()
    assert loaded.model.model_ is net and loaded.model.classes_.tolist() == model.model.classes_.tolist()


def test_function_loader_shares_the_returned_network(tmp_path, data):
    a = _fit(FunctionModel, tmp_path, data, "a")
    b = _fit(FunctionModel, tmp_path, data, "b")
    c = _fit(FunctionModel, tmp_path, data, "c", {"precision": "bfloat16"})
    assert a.model.network is b.model.network and c.model.network is not a.model.network
    assert len(BUILDS) == 2
    loaded = pickle.loads(pickle.dumps(a))
    assert loaded.model.network is None
    loaded.predict_proba(data[0])
    assert loaded.model.network is a.model.network
    blob = pickle.dumps(a)
    registry.release()
    fresh = pickle.loads(blob)
    fresh.prepare_for_inference()
    assert fresh.model.network is not None and len(BUILDS) == 3, (
        "a function loader is re-run with its recorded arguments"
    )


def test_copy_per_fit_gives_every_fit_its_own_network(tmp_path, data):
    a = _fit(FineTuneModel, tmp_path, data, "a")
    b = _fit(FineTuneModel, tmp_path, data, "b")
    assert len(BUILDS) == 1
    assert a.model.network is not b.model.network
    assert torch.equal(a.model.network.linear.bias, b.model.network.linear.bias)
    cached = registry.report()["entries"]
    assert len(cached) == 1
    payload = registry.peek(a._shared_state.key).value
    assert payload.result is not a.model.network, "the fit that built the entry trains a copy too"
    assert torch.all(payload.result.linear.bias == 0), "the cached base network is untouched by the fine-tuning"
    assert payload.arguments == {"path": "base.safetensors", "device": "cpu", "dtype": "bf16"}, (
        "keyword options are recorded by name"
    )
    # The fitted model owns its copy: the pickle carries the fine-tuned weights, devices move through
    # `_set_device`, and the memory size is the pickled size.
    assert sw.owns_network(a) and a._pickles_pretrained_weights()
    loaded = pickle.loads(pickle.dumps(a))
    assert torch.equal(loaded.model.network.linear.bias, a.model.network.linear.bias)
    a.set_device("cpu")
    assert a.model.device == "cpu"
    assert a._get_memory_size() == a._get_pickled_size()
    assert a.get_info()["shared_weights"]["present_before_fit"] is False
    assert b.get_info()["shared_weights"]["present_before_fit"] is True


def test_fit_device_keys_a_loader_without_a_device_input(tmp_path, data, monkeypatch):
    """The fit's device (from ``num_gpus``) keys a loader that has no device input, and the built network is placed there."""
    assert sw.fit_device(None) is None and sw.fit_device(0) == "cpu" and sw.fit_device(0.5) == "cuda"
    spec = SharedWeights(loader=f"{__name__}:build_network_on_cpu", key=("checkpoint",))
    sw.install(spec)
    model = _model(MethodModel, tmp_path)
    moved = []
    monkeypatch.setattr(_Net, "to", lambda self, *a, **k: moved.append(a) or self)
    monkeypatch.setattr(graph, "_is_tensor", lambda obj: False)
    monkeypatch.setattr(model, "shared_weights", spec, raising=False)
    with sw.scope(model, enabled=True, device="cuda") as current:
        on_gpu = build_network_on_cpu("x.ckpt")
    assert current.key.device == "cuda" and moved == [("cuda",)], "placed on the fit device once"
    with sw.scope(model, enabled=True, device="cpu") as current:
        on_cpu = build_network_on_cpu("x.ckpt")
    assert current.key.device == "cpu" and on_cpu is not on_gpu and len(BUILDS) == 2
    with sw.scope(model, enabled=True, device="cuda"):
        assert build_network_on_cpu("x.ckpt") is on_gpu, "a hit for the fit device"
    with sw.scope(model, enabled=True):
        assert build_network_on_cpu("x.ckpt") is on_cpu, "no fit device known: the CPU entry"


def test_memory_size_counts_the_shared_tensors(tmp_path, data):
    model = _fit(MethodModel, tmp_path, data)
    assert model._get_memory_size() == model._get_pickled_size() + registry.tensor_bytes(model.model.model_)


def test_info_block(tmp_path, data):
    model = _fit(MethodModel, tmp_path, data, hyperparameters={"checkpoint": "one.ckpt"})
    block = model.get_info()["shared_weights"]
    assert set(block) == set(sw.INFO_KEYS)
    assert block["inputs"] == {"checkpoint": "one.ckpt"}
    assert block["device"] == "cpu" and block["loaded_by"] == "fit" and block["present_before_fit"] is False
    assert block["variant"] == "_Estimator._load_model" and block["library"] == __name__.split(".")[0]


def test_changed_checkpoint_file_invalidates_the_entry(tmp_path, data):
    ckpt = tmp_path / "w.ckpt"
    ckpt.write_bytes(b"one")
    _fit(MethodModel, tmp_path, data, "a", {"checkpoint": str(ckpt)})
    entry = registry.report()["entries"][0]
    assert entry["source"]["file_stamp"]["path"] == str(ckpt)
    _fit(MethodModel, tmp_path, data, "b", {"checkpoint": str(ckpt)})
    assert len(BUILDS) == 1
    ckpt.write_bytes(b"two changed")
    os.utime(ckpt, ns=(0, 0))
    _fit(MethodModel, tmp_path, data, "c", {"checkpoint": str(ckpt)})
    assert len(BUILDS) == 2, "a changed file on disk is rebuilt"


def test_set_device_swaps_the_entry_instead_of_moving_the_module(tmp_path, data, monkeypatch):
    model = _fit(MethodModel, tmp_path, data)
    net = model.model.model_
    moved = []
    monkeypatch.setattr(_Net, "to", lambda self, *a, **k: moved.append(self) or self)
    model.set_device("cpu")  # same device type: nothing happens
    assert model.model.model_ is net and not moved
    monkeypatch.setattr(graph, "_is_tensor", lambda obj: False)  # keep the fake device swap off real CUDA
    monkeypatch.setattr(sw, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model.set_device("cuda")
    assert model.model.device == "cuda"
    assert model.model.model_ is not net, "the CUDA entry replaced the CPU one"
    assert net not in moved, "the shared CPU module was never moved with .to(); the library built a new one"
    assert model._shared_state.device == "cuda" and model.get_device() == "cuda"
    assert [b[0] for b in BUILDS] == ["method", "method"]


def test_cuda_pickle_loads_on_a_machine_without_cuda(tmp_path, data):
    """A predictor fitted on a GPU is loaded where CUDA is unavailable (the CI's check of every model artifact).

    The load rewrites the estimator's device fields from cuda to cpu, which re-keys the per-device cache
    the network sits in; the paths recorded by the pickle must follow, or the attach misses the entry.
    """
    model = _fit(MethodModel, tmp_path, data)
    net = model.model.model_
    loaded = pickle.loads(pickle.dumps(model))
    # Turn the weightless CPU pickle into what a CUDA fit writes: device fields and the cache key name cuda:0.
    state = loaded._shared_state
    state.device = "cuda"
    state.key = state.key.replace(device="cuda")
    loaded.model.device = "cuda:0"
    loaded.model.caches_ = {torch.device("cuda", 0): None}
    cuda_step = ("key", torch.device("cuda", 0))
    state.paths = [
        (tuple(cuda_step if step == ("key", torch.device("cpu")) else step for step in path), index)
        for path, index in state.paths
    ]
    assert any(cuda_step in path for path, _ in state.paths), "the pickle records the path through the cache"

    loaded.set_device("cpu")  # what ``AbstractTorchModel.load`` does when the fit device is unavailable

    assert loaded.model.device == "cpu" and loaded._shared_state.device == "cpu"
    assert loaded.model.model_ is net, "the CPU entry of the same checkpoint"
    assert loaded.model.caches_ == {torch.device("cpu"): net}, "the re-keyed cache holds the network again"
    assert loaded._shared_state.paths == []
    assert len(BUILDS) == 1, "nothing was rebuilt: the CPU entry was already in the registry"
