"""A model's torch tensors are saved beside ``model.pkl``, not inside it, and sized without a copy."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract import _tensor_pickle

torch = pytest.importorskip("torch")


class _Model(AbstractModel):
    pass


class _PinnedModel(AbstractModel):
    pin_tensors_on_load = True


def _model(tmp_path, cls=_Model, state=None):
    model = cls(path=str(tmp_path), name="m")
    model.model = state
    return model


def _files(path: str) -> set[str]:
    return set(os.listdir(path))


def test_tensors_are_written_beside_the_pickle_and_restored(tmp_path):
    cache = torch.randn(4096, 64)
    state = {"cache": cache, "same_cache": cache, "head": cache[:16], "weights": np.arange(10), "name": "x"}
    model = _model(tmp_path, state=state)

    path = model.save()

    assert {_Model.model_file_name, _Model.model_tensors_file_name} <= _files(path)
    assert os.path.getsize(os.path.join(path, _Model.model_file_name)) < cache.nbytes / 10, "the cache left the pickle"
    loaded = _Model.load(path).model
    assert torch.equal(loaded["cache"], cache)
    assert loaded["same_cache"] is loaded["cache"], "one tensor object stays one object"
    assert loaded["head"].untyped_storage().data_ptr() == loaded["cache"].untyped_storage().data_ptr(), (
        "a view keeps sharing its base's storage"
    )
    np.testing.assert_array_equal(loaded["weights"], np.arange(10))
    assert loaded["name"] == "x"


def test_a_model_without_tensors_writes_no_tensor_file_and_removes_a_stale_one(tmp_path):
    model = _model(tmp_path, state={"cache": torch.ones(8)})
    path = model.save()
    assert _Model.model_tensors_file_name in _files(path)

    model.model = {"cache": None}
    model.save()

    assert _Model.model_tensors_file_name not in _files(path)
    assert _Model.load(path).model == {"cache": None}


def test_pin_tensors_on_load_pins_each_storage_once_when_cuda_is_available(tmp_path, monkeypatch):
    cache = torch.randn(1024, 32)
    state = {"cache": cache, "head": cache[:16], "other": torch.ones(3)}
    path = _model(tmp_path, cls=_PinnedModel, state=state).save()
    pinned = []

    def fake_pin_memory(storage, *args, **kwargs):
        pinned.append(storage.nbytes())
        return storage.clone()

    monkeypatch.setattr(torch.UntypedStorage, "pin_memory", fake_pin_memory)
    _Model.load(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _PinnedModel.load(path)
    assert pinned == [], "nothing is pinned without the attribute or without CUDA"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = _PinnedModel.load(path)

    assert sorted(pinned) == sorted([cache.nbytes, 3 * 4]), "one pin per storage"
    assert torch.equal(loaded.model["cache"], cache)
    assert loaded.model["head"].untyped_storage().data_ptr() == loaded.model["cache"].untyped_storage().data_ptr()
    # Saving the loaded model again replaces the file it was loaded from (refit_full re-saves a loaded model).
    loaded.save()
    assert torch.equal(_Model.load(path).model["cache"], cache)


def test_pickled_size_counts_storages_once_without_serializing_tensors(monkeypatch):
    cache = torch.zeros(1000, 100)
    state = {"cache": cache, "views": [cache[i] for i in range(10)], "other": torch.zeros(10, dtype=torch.int64)}
    without_tensors = len(pickle.dumps({"cache": None, "views": [None] * 10, "other": None}, protocol=4))

    def refuse(*args, **kwargs):
        raise AssertionError("a tensor was serialized")

    monkeypatch.setattr(torch.Tensor, "__reduce_ex__", refuse)
    size = _tensor_pickle.pickled_size(state)

    assert size >= cache.nbytes + 80, "the cache's storage and the other tensor are counted once each"
    assert size < cache.nbytes + 80 + without_tensors + 1024, "the ten views add no storage bytes"


def test_get_memory_size_matches_the_saved_files(tmp_path):
    model = _model(tmp_path, state={"cache": torch.randn(2048, 128)})
    size = model.get_memory_size()
    path = model.save()

    on_disk = sum(os.path.getsize(os.path.join(path, name)) for name in _files(path))
    assert abs(size - on_disk) < 0.01 * on_disk


def test_an_object_without_tensors_pickles_as_before(tmp_path):
    obj = {"a": np.arange(100), "b": [1, 2, 3]}
    path = str(tmp_path / "obj.pkl")
    tensors_path = str(tmp_path / "obj_tensors.pt")

    _tensor_pickle.save(obj, path=path, tensors_path=tensors_path)

    assert not os.path.exists(tensors_path)
    with open(path, "rb") as f:
        assert f.read() == pickle.dumps(obj, protocol=_tensor_pickle.PROTOCOL)
    assert _tensor_pickle.pickled_size(obj) == len(pickle.dumps(obj, protocol=_tensor_pickle.PROTOCOL))
