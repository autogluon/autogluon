"""Tests for the process-wide shared-weights registry and its helpers.

Loaders are counting fakes returning plain objects, so the registry logic (hit, miss, LRU
eviction, release, report) needs no torch. The torch-backed helpers (tensor byte counting, the RNG
guard) run only when torch is installed.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import pickle
import random
import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import autogluon.core.models.abstract._shared_weights_registry as w
from autogluon.core.models.abstract._shared_weights_registry import WeightsKey, make_key

try:
    import torch
except ImportError:  # the core test environment installs no torch; the torch-backed tests below skip
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts and ends with an empty registry, zeroed stats and the default capacity."""
    w.release()
    w.reset_stats()
    w.set_capacity(w.DEFAULT_CAPACITY)
    yield
    w.release()
    w.reset_stats()
    w.set_capacity(w.DEFAULT_CAPACITY)


class _Loader:
    """A counting fake loader returning a fresh sentinel object per call."""

    def __init__(self, value_factory=object):
        self.calls = 0
        self.value_factory = value_factory

    def __call__(self):
        self.calls += 1
        return self.value_factory()


def _key(name: str = "ckpt", device: str = "cpu", **changes) -> WeightsKey:
    return WeightsKey(library="lib", checkpoint=name, variant="classifier", device=device, **changes)


# --- keys -------------------------------------------------------------------------------------


def test_normalize_device_keeps_type_only():
    assert w.normalize_device("cuda:0") == "cuda"
    assert w.normalize_device("cuda") == "cuda"
    assert w.normalize_device(" CPU ") == "cpu"
    assert w.normalize_device(None) == "cpu"


def test_make_key_normalizes_device_path_and_flags(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    a = make_key("lib", "model.safetensors", "classifier", "cuda", b_flag=1, a_flag=True)
    b = make_key("lib", ckpt, "classifier", "cuda:0", a_flag=True, b_flag=1)
    assert a == b
    assert a.checkpoint == str(ckpt.resolve())
    assert a.device == "cuda"
    assert a.flags == (("a_flag", "True"), ("b_flag", "1"))
    assert a.dtype == "float32"


def test_make_key_resolves_symlink_to_blob(tmp_path):
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    link = tmp_path / "snapshots" / "sha" / "model.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    assert make_key("lib", link, "classifier", "cpu").checkpoint == str(blob.resolve())


def test_make_key_keeps_opaque_id():
    key = make_key("lib", "lib:org/model@sha/classification", "classification", None)
    assert key.checkpoint == "lib:org/model@sha/classification"
    assert key.device == "cpu"


def test_weights_key_replace_to_dict_short():
    key = _key(device="cpu", flags=(("use_flash", "True"),))
    moved = key.replace(device="cuda:1")
    assert moved.device == "cuda"
    assert moved != key
    assert key.replace(flags={"z": 1, "a": 2}).flags == (("a", "2"), ("z", "1"))
    assert key.to_dict() == {
        "library": "lib",
        "checkpoint": "ckpt",
        "variant": "classifier",
        "device": "cpu",
        "dtype": "float32",
        "flags": {"use_flash": "True"},
    }
    assert key.short() == "lib/classifier@cpu:float32[ckpt]{use_flash=True}"
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.library = "other"


# --- registry -----------------------------------------------------------------------------------


def test_get_or_load_miss_then_hit_calls_loader_once():
    key = _key()
    loader = _Loader()
    first = w.get_or_load(key, loader)
    second = w.get_or_load(key, loader)
    assert first is second
    assert loader.calls == 1
    assert w.contains(key)
    assert w.loaded_by(key) == "fit"
    stats = w.report()["stats"]
    assert (stats["hits"], stats["misses"]) == (1, 1)
    assert w.peek(key).hits == 1


def test_warmup_stage_then_fit_stage_is_a_hit_with_warmup_provenance():
    key = _key()
    loader = _Loader()
    value = w.get_or_load(key, loader, stage="warmup", source={"repo_id": "org/repo", "filename": "m.bin"})
    entry = w.peek(key)
    assert entry.value is value and entry.loaded_by == "warmup"
    assert entry.source == {"repo_id": "org/repo", "filename": "m.bin"}
    assert w.loaded_by(key) == "warmup"
    assert w.get_or_load(key, _Loader(), stage="fit") is value
    assert loader.calls == 1
    assert w.peek(key).hits == 1
    assert w.peek(key).loaded_by == "warmup"  # provenance is the first load, not the hit
    stats = w.report()["stats"]
    assert (stats["hits"], stats["misses"]) == (1, 1)


def test_loader_exception_registers_nothing():
    key = _key()

    def boom():
        raise RuntimeError("no weights")

    with pytest.raises(RuntimeError, match="no weights"):
        w.get_or_load(key, boom)
    with pytest.raises(RuntimeError, match="no weights"):
        w.get_or_load(key, boom, stage="warmup")
    assert not w.contains(key)
    assert w.peek(key) is None
    assert w.report()["stats"] == {"hits": 0, "misses": 0, "evictions": 0}


def test_lru_eviction_at_capacity_warns_for_warmup_entries(caplog):
    w.set_capacity(1)
    a, b = _key("a"), _key("b")
    loader = _Loader()
    w.get_or_load(a, loader, stage="warmup")
    with caplog.at_level(logging.WARNING, logger=w.__name__):
        w.get_or_load(b, loader)
    assert not w.contains(a)
    assert w.contains(b)
    assert any("(loaded by warmup) evicted" in record.message for record in caplog.records)
    assert w.report()["stats"]["evictions"] == 1
    # Reloading A is a miss again and evicts B (loaded by fit: an info line, counted as a plain eviction).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=w.__name__):
        w.get_or_load(a, loader)
    assert loader.calls == 3
    assert w.report()["stats"] == {"hits": 0, "misses": 3, "evictions": 2}
    assert not any(record.levelno >= logging.WARNING and "evicted" in record.message for record in caplog.records)


def test_lru_order_moves_hit_entries_to_the_end():
    w.set_capacity(2)
    a, b, c = _key("a"), _key("b"), _key("c")
    loader = _Loader()
    w.get_or_load(a, loader)
    w.get_or_load(b, loader)
    w.get_or_load(a, loader)  # hit: A becomes most recently used
    w.get_or_load(c, loader)
    assert w.contains(a)
    assert not w.contains(b)
    assert w.contains(c)


def test_capacity_zero_does_not_store():
    w.set_capacity(0)
    key = _key()
    loader = _Loader()
    first = w.get_or_load(key, loader)
    second = w.get_or_load(key, loader, stage="warmup", source={"repo_id": "org/repo", "filename": "ckpt"})
    assert loader.calls == 2 and first is not second, "every call builds anew"
    assert not w.contains(key) and w.peek(key) is None
    assert w.report()["capacity"] == 0
    assert w.report()["stats"] == {"hits": 0, "misses": 2, "evictions": 0}


def test_set_capacity_evicts_down_immediately():
    loader = _Loader()
    w.get_or_load(_key("a"), loader)
    w.get_or_load(_key("b"), loader)
    w.set_capacity(1)
    assert not w.contains(_key("a"))
    assert w.contains(_key("b"))
    assert w.capacity() == 1


def test_capacity_default_and_env_override(monkeypatch):
    monkeypatch.setattr(w, "_CAPACITY", None)
    monkeypatch.delenv(w.CAPACITY_ENV_VAR, raising=False)
    monkeypatch.delenv(w.LEGACY_CAPACITY_ENV_VAR, raising=False)
    assert w.capacity() == w.DEFAULT_CAPACITY == 2
    monkeypatch.setattr(w, "_CAPACITY", None)
    monkeypatch.setenv(w.CAPACITY_ENV_VAR, "5")
    assert w.capacity() == 5
    monkeypatch.setattr(w, "_CAPACITY", None)
    monkeypatch.setenv(w.CAPACITY_ENV_VAR, "not-a-number")
    assert w.capacity() == w.DEFAULT_CAPACITY
    # The earlier variable name is read when the current one is unset, and loses to it otherwise.
    monkeypatch.setattr(w, "_CAPACITY", None)
    monkeypatch.delenv(w.CAPACITY_ENV_VAR)
    monkeypatch.setenv(w.LEGACY_CAPACITY_ENV_VAR, "3")
    assert w.capacity() == 3
    monkeypatch.setattr(w, "_CAPACITY", None)
    monkeypatch.setenv(w.CAPACITY_ENV_VAR, "4")
    assert w.capacity() == 4


def test_release_drops_references_but_live_holders_keep_them():
    class _Weights:
        pass

    key = _key()
    holder = w.get_or_load(key, _Loader(_Weights))
    ref = weakref.ref(holder)
    assert w.release() == 1
    assert not w.contains(key)
    assert ref() is holder
    del holder
    gc.collect()
    assert ref() is None
    assert w.release() == 0


def test_release_subset():
    loader = _Loader()
    a = _key("a")
    b = WeightsKey(library="other", checkpoint="b", variant="regressor", device="cpu")
    w.get_or_load(a, loader)
    w.get_or_load(b, loader)
    assert w.release(keys=[a, _key("missing")]) == 1
    assert not w.contains(a)
    assert w.contains(b)
    assert w.release(keys=[a]) == 0


def test_report_and_reset_stats_keep_entries(tmp_path):
    blob = tmp_path / "blobs" / "0123456789abcdef"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    key = make_key("lib", blob, "classifier", "cpu")
    assert key.checkpoint == str(blob.resolve())  # the key itself keeps the absolute path
    w.get_or_load(key, _Loader(), stage="warmup", source={"repo_id": "org/repo"})
    w.get_or_load(key, _Loader())
    report = w.report()
    assert report["capacity"] == 2
    assert len(report["entries"]) == 1
    entry = report["entries"][0]
    assert entry["library"] == "lib"
    assert entry["checkpoint"] == "0123456789abcdef"  # the metadata records the blob basename only
    assert "/" not in entry["checkpoint"]
    assert str(tmp_path) not in str(entry)
    assert w.peek(key).key.to_dict()["checkpoint"] == str(blob.resolve())
    assert entry["loaded_by"] == "warmup"
    assert entry["hits"] == 1
    assert entry["source"] == {"repo_id": "org/repo"}
    assert entry["n_bytes"] is None
    assert entry["load_time_s"] >= 0
    assert "value" not in entry
    w.reset_stats()
    after = w.report()
    assert after["stats"] == {"hits": 0, "misses": 0, "evictions": 0}
    assert len(after["entries"]) == 1
    assert after["entries"][0]["hits"] == 0
    assert w.contains(key)


def test_rng_state_unchanged_around_a_loader_without_torch_state():
    """The Python and NumPy generators are restored even when the loader only touches those."""
    random.seed(123)
    np.random.seed(123)
    py_before = random.getstate()
    np_before = np.random.get_state()

    def draws():
        random.random()  # noqa: S311
        np.random.rand(3)
        return object()

    w.get_or_load(_key(), draws)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]


# --- helpers ------------------------------------------------------------------------------------


def test_shallow_copy_shares_every_attribute_without_running_init():
    net = object()
    estimator = SimpleNamespace(net=net, scale=2.0)
    copy = w.shallow_copy(estimator)
    assert copy is not estimator
    assert copy.net is net
    assert copy.__dict__ == estimator.__dict__
    copy.net = None
    assert estimator.net is net  # the live object keeps its network


def test_shared_weights_class_settings_defaults_and_replace():
    settings = w.SharedWeightsClassSettings()
    assert settings.share_weights is True
    assert settings.replace(share_weights=False).share_weights is False
    assert settings.to_dict() == {"share_weights": True}
    with pytest.raises(ValueError, match="Unknown"):
        settings.replace(shared=False)


def test_module_import_is_torch_free():
    code = (
        "import sys, autogluon.core.models.abstract._shared_weights_registry; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)  # noqa: S603


# --- torch-backed helpers -----------------------------------------------------------------------


@requires_torch
def test_tensor_bytes_counts_parameters_buffers_and_shared_storage_once():
    linear = torch.nn.Linear(3, 1)  # 3 weights + 1 bias, float32
    assert w.tensor_bytes(linear) == 16
    linear.register_buffer("scale", torch.ones(2))
    assert w.tensor_bytes(linear) == 24
    tied = torch.nn.Module()
    tied.a = torch.nn.Linear(4, 4, bias=False)
    tied.b = torch.nn.Linear(4, 4, bias=False)
    tied.b.weight = tied.a.weight  # tied weights share one storage
    assert w.tensor_bytes(tied) == 64
    assert w.tensor_bytes(linear.state_dict()) == 24
    assert w.tensor_bytes([linear, tied]) == 88


@requires_torch
def test_tensor_bytes_counts_every_meta_tensor():
    with torch.device("meta"):
        module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert w.tensor_bytes(module) == 160  # 2 x (16 weights + 4 bias) x 4 bytes, not one storage at data_ptr 0


@requires_torch
def test_loader_result_gets_n_bytes_for_modules_and_state_dicts():
    w.get_or_load(_key("module"), lambda: torch.nn.Linear(3, 1))
    w.get_or_load(_key("sd"), lambda: torch.nn.Linear(3, 1).state_dict())
    assert w.peek(_key("module")).n_bytes == 16
    assert w.peek(_key("sd")).n_bytes == 16


@requires_torch
def test_rng_state_unchanged_around_a_loader_that_draws_from_torch_numpy_and_random():
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch_before = torch.get_rng_state().clone()
    np_before = np.random.get_state()
    py_before = random.getstate()

    def build():
        torch.rand(10)
        np.random.rand(10)
        random.random()  # noqa: S311
        return torch.nn.Linear(2, 2)

    w.get_or_load(_key(), build, stage="warmup")
    assert torch.equal(torch.get_rng_state(), torch_before)
    np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
    assert random.getstate() == py_before
    # The next draw is exactly what it would have been without the loader.
    torch.manual_seed(7)
    expected = torch.rand(1)
    torch.manual_seed(7)
    w.get_or_load(_key("other"), build)
    assert torch.equal(torch.rand(1), expected)


@requires_torch
def test_rng_guard_is_the_public_form_of_the_loader_guard():
    torch.manual_seed(3)
    before = torch.get_rng_state().clone()
    with w.rng_guard(cuda=False):
        torch.rand(5)
    assert torch.equal(torch.get_rng_state(), before)


class _Estimator:
    """A fitted-estimator stand-in that pickles without its network, as the wrapper convention does."""

    def __init__(self, net):
        self.net = net
        self.mean = np.zeros(3)

    def __getstate__(self):
        return {**self.__dict__, "net": None}


@requires_torch
def test_weightless_pickle_of_tiny_module_holder_reattaches_the_same_object():
    key = _key("tiny")
    net = w.get_or_load(key, lambda: torch.nn.Linear(3, 1).eval())
    estimator = _Estimator(net)
    weightless = pickle.dumps(estimator)
    with_weights = pickle.dumps(w.shallow_copy(estimator).__dict__)
    assert len(weightless) < len(with_weights)
    loaded = pickle.loads(weightless)
    assert loaded.net is None
    assert estimator.net is net
    loaded.net = w.get_or_load(key, lambda: pytest.fail("reattach must be a registry hit"))
    assert loaded.net is net
    x = torch.ones(2, 3)
    with torch.no_grad():
        assert torch.equal(loaded.net(x), estimator.net(x))
