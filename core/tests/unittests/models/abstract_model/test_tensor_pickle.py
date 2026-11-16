"""A model's torch tensors are saved beside ``model.pkl``, not inside it, and sized without a copy."""

from __future__ import annotations

import io
import os
import pickle
import shutil
import subprocess
import sys
import types

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
    without_tensors = len(
        pickle.dumps({"cache": None, "views": [None] * 10, "other": None}, protocol=_tensor_pickle.PROTOCOL)
    )

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


def test_numpy_arrays_are_saved_and_sized_without_a_copy(tmp_path):
    """Protocol 5 hands an array's buffer to the file or the byte counter as it is; protocol 4 copies it."""
    import tracemalloc

    weights = np.ones((1_000_000, 8))  # 64 MB
    model = _model(tmp_path, state={"weights": weights, "cache": torch.ones(4)})
    tracemalloc.start()
    try:
        model.get_memory_size()
        path = model.save()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < weights.nbytes / 4, "no copy of the array"
    np.testing.assert_array_equal(_Model.load(path).model["weights"], weights)


def test_an_object_without_tensors_is_a_plain_pickle(tmp_path):
    obj = {"a": np.arange(100), "b": [1, 2, 3]}
    path = str(tmp_path / "obj.pkl")
    tensors_path = str(tmp_path / "obj_tensors.pt")

    _tensor_pickle.save(obj, path=path, tensors_path=tensors_path)

    assert not os.path.exists(tensors_path)
    with open(path, "rb") as f:
        assert f.read() == pickle.dumps(obj, protocol=_tensor_pickle.PROTOCOL)
    assert _tensor_pickle.pickled_size(obj) == len(pickle.dumps(obj, protocol=_tensor_pickle.PROTOCOL))


@pytest.mark.parametrize(
    "error",
    [OSError("No space left on device"), RuntimeError("PytorchStreamWriter failed writing file data/0")],
    ids=["os-error", "torch-write-error"],
)
def test_a_failed_resave_keeps_the_previous_pair_loadable(tmp_path, monkeypatch, error):
    """A ``torch.save`` that fails to write raises, instead of keeping the tensors in the pickle as for a
    state it refuses."""
    model = _model(tmp_path, state={"weights": torch.zeros(3), "bias": torch.ones(2)})
    path = model.save()
    model.model = {"bias": torch.full((2,), 5.0), "weights": torch.full((3,), 7.0)}

    real_save = torch.save

    def failing_save(obj, f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):  # the tensor file; torch pickles a tensor into a buffer
            raise error
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(type(error)):
        model.save()
    monkeypatch.undo()

    loaded = _Model.load(path).model
    assert torch.equal(loaded["weights"], torch.zeros(3)) and torch.equal(loaded["bias"], torch.ones(2))
    assert _files(path) == {_Model.model_file_name, _Model.model_tensors_file_name}, "no temporary file is left"


def test_a_tensor_file_from_another_save_is_refused(tmp_path):
    first = _model(tmp_path / "a", state={"weights": torch.zeros(3)}).save()
    second = _model(tmp_path / "b", state={"weights": torch.ones(3)}).save()
    os.replace(
        os.path.join(second, _Model.model_tensors_file_name), os.path.join(first, _Model.model_tensors_file_name)
    )

    with pytest.raises(pickle.UnpicklingError, match="different save"):
        _Model.load(first)


def test_a_missing_tensor_file_and_a_plain_unpickle_fail_clearly(tmp_path):
    path = _model(tmp_path, state={"weights": torch.zeros(3)}).save()
    with open(os.path.join(path, _Model.model_file_name), "rb") as f:
        with pytest.raises(pickle.UnpicklingError, match="model_tensors.pt"):
            pickle.load(f)

    os.remove(os.path.join(path, _Model.model_tensors_file_name))
    with pytest.raises(FileNotFoundError, match="tensor file is missing"):
        _Model.load(path)


def test_a_pickle_without_tensor_references_never_reads_the_tensor_file(tmp_path):
    path = _model(tmp_path, state={"weights": np.zeros(3)}).save()
    with open(os.path.join(path, _Model.model_tensors_file_name), "wb") as f:
        f.write(b"not a tensor file")

    np.testing.assert_array_equal(_Model.load(path).model["weights"], np.zeros(3))


@pytest.mark.parametrize("cls", [_Model, _PinnedModel])
def test_cuda_tensors_load_on_the_cpu_without_cuda(tmp_path, monkeypatch, cls):
    cache = torch.arange(10.0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "location_tag", lambda storage: "cuda:0")
        path = _model(tmp_path, cls=cls, state={"cache": cache, "head": cache[2:5]}).save()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    loaded = cls.load(path).model

    assert loaded["cache"].device.type == "cpu" and torch.equal(loaded["cache"], cache)
    assert loaded["head"].untyped_storage().data_ptr() == loaded["cache"].untyped_storage().data_ptr()


@pytest.mark.parametrize("location", ["cuda:3", "mps"])
def test_tensors_of_a_device_this_machine_lacks_load_on_the_cpu(tmp_path, monkeypatch, location):
    cache = torch.arange(10.0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "location_tag", lambda storage: location)
        path = _model(tmp_path, state={"cache": cache}).save()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    loaded = _Model.load(path).model

    assert loaded["cache"].device.type == "cpu" and torch.equal(loaded["cache"], cache)


class _Probe:
    """Records, each time it is pickled, whether the pickle writes its tensors apart."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.seen = []

    def __getstate__(self):
        self.seen.append(_tensor_pickle.tensors_saved_apart())
        return {"tensor": self.tensor, "seen": []}


def test_a_getstate_learns_whether_the_tensors_are_saved_apart(tmp_path):
    probe = _Probe(torch.ones(2))
    model = _model(tmp_path, state={"probe": probe})

    model.save()
    model._get_pickled_size()
    pickle.dumps(model)
    values = torch.arange(4.0)
    probe.tensor = {"float": values, "int": values.view(torch.int32)}  # torch.save refuses: a plain pickle follows
    model.save()

    assert probe.seen == [True, True, False, True, False]
    assert not _tensor_pickle.tensors_saved_apart()


def test_a_load_without_pinning_or_cuda_tensors_never_asks_for_cuda(tmp_path, monkeypatch):
    path = _model(tmp_path, state={"cache": torch.ones(8)}).save()

    def refuse():
        raise AssertionError("CUDA was initialized, which keeps a forked child from using it")

    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    assert torch.equal(_Model.load(path).model["cache"], torch.ones(8))


def test_a_storage_that_cannot_be_pinned_loads_unpinned(tmp_path, monkeypatch):
    path = _model(tmp_path, cls=_PinnedModel, state={"cache": torch.ones(8)}).save()

    def fail_to_pin(storage, *args, **kwargs):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.UntypedStorage, "pin_memory", fail_to_pin)
    assert torch.equal(_PinnedModel.load(path).model["cache"], torch.ones(8))


def test_tensors_torch_save_refuses_stay_in_the_pickle(tmp_path):
    model = _model(tmp_path, state={"cache": torch.ones(8)})
    path = model.save()
    values = torch.arange(4.0)
    model.model = {"float": values, "int": values.view(torch.int32)}

    model.save()

    assert _files(path) == {_Model.model_file_name}, "the stale tensor file is removed"
    loaded = _Model.load(path).model
    assert torch.equal(loaded["float"], values) and torch.equal(loaded["int"], values.view(torch.int32))


def test_cuda_tensors_kept_in_the_pickle_load_on_the_cpu_without_cuda(tmp_path, monkeypatch):
    """A state ``torch.save`` refuses keeps its tensors in the pickle, as a save to S3 does; they restore as
    the tensor file's do, where torch's own unpickling restores them on the device they were saved from."""
    values = torch.arange(4.0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "location_tag", lambda storage: "cuda:0")
        path = _model(tmp_path, state={"float": values, "int": values.view(torch.int32)}).save()
    assert _files(path) == {_Model.model_file_name}, "torch.save refused the state: one pickle"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with open(os.path.join(path, _Model.model_file_name), "rb") as f, pytest.raises(RuntimeError, match="CUDA"):
        pickle.load(f)
    loaded = _Model.load(path).model

    assert loaded["float"].device.type == "cpu" and torch.equal(loaded["float"], values)
    assert loaded["int"].device.type == "cpu" and torch.equal(loaded["int"], values.view(torch.int32))


def test_saves_into_one_folder_at_once_never_mix_their_files(tmp_path, monkeypatch):
    first = _model(tmp_path, state={"weights": torch.zeros(3)})
    second = _model(tmp_path, state={"weights": torch.ones(3)})
    replace = os.replace
    calls = []

    def replace_then_save_again(src, dst):
        replace(src, dst)
        calls.append(dst)
        if len(calls) == 1:
            second.save()  # runs between the first save's two moves

    monkeypatch.setattr(_tensor_pickle.os, "replace", replace_then_save_again)
    first.save()
    monkeypatch.undo()

    path = first.path
    assert _files(path) == {_Model.model_file_name, _Model.model_tensors_file_name}, "no temporary file is left"
    # The first save's pickle beside the second save's tensors: refused, never mixed.
    with pytest.raises(pickle.UnpicklingError, match="different save"):
        _Model.load(path)
    second.save()
    assert torch.equal(_Model.load(path).model["weights"], torch.ones(3))


def test_an_object_without_tensors_never_imports_torch(tmp_path):
    code = (
        "import os\n"
        "import sys\n"
        "from autogluon.core.models import AbstractModel\n"
        f"model = AbstractModel(path={str(tmp_path)!r}, name='m')\n"
        "model.model = {'weights': [1.0, 2.0]}\n"
        "model.get_memory_size()\n"
        "path = model.save()\n"
        "assert AbstractModel.load(path).model == model.model\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "with open(os.path.join(path, AbstractModel.model_file_name), 'rb') as f:\n"
        "    assert f.read(2) == b'\\x80\\x05', 'protocol 5, as with torch imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)  # noqa: S603


def test_pickled_size_counts_sparse_and_meta_tensors_as_written():
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (100_000, 100_000), check_invariants=True
    )
    meta = torch.empty(10_000_000, device="meta")

    assert _tensor_pickle.pickled_size({"sparse": sparse, "meta": meta}) < 10_000


class _FakeS3:
    """S3 as boto3's resource reaches it, backed by a local folder: ``s3://bucket/<key>`` is ``<root>/<key>``."""

    def __init__(self, root):
        self.root = root
        #: ``("get", key)`` reads an object into memory, ``("download", key, local_path)`` copies it to a file.
        self.requests: list[tuple] = []
        #: Keys answered with 403, as S3 answers a missing one when the caller may not list the bucket.
        self.denied: set[str] = set()

    def Bucket(self, name):
        assert name == "bucket"
        return self

    def Object(self, key):
        def get():
            self.requests.append(("get", key))
            return {"Body": io.BytesIO(self._path(key, "GetObject").read_bytes())}

        return types.SimpleNamespace(get=get)

    def download_file(self, key, local_path):
        shutil.copyfile(self._path(key, "HeadObject"), local_path)  # boto3 asks for the object's head first
        self.requests.append(("download", key, local_path))

    def _path(self, key, operation):
        from botocore.exceptions import ClientError

        if key in self.denied:
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, operation)
        if not (self.root / key).exists():
            raise ClientError({"Error": {"Code": "404" if operation == "HeadObject" else "NoSuchKey"}}, operation)
        return self.root / key


@pytest.fixture
def fake_s3(tmp_path, monkeypatch):
    import boto3

    s3 = _FakeS3(tmp_path / "bucket")
    monkeypatch.setattr(boto3, "resource", lambda service: s3)
    return s3


def test_a_model_saved_locally_and_copied_to_s3_loads_from_there(tmp_path, fake_s3):
    cache = torch.randn(64, 8)
    shutil.copytree(_model(tmp_path / "local", state={"cache": cache, "head": cache[:4]}).save(), fake_s3.root / "m")

    loaded = _Model.load("s3://bucket/m/").model

    assert torch.equal(loaded["cache"], cache)
    assert loaded["head"].untyped_storage().data_ptr() == loaded["cache"].untyped_storage().data_ptr()
    get, download = fake_s3.requests
    assert get == ("get", "m/model.pkl") and download[:2] == ("download", "m/model_tensors.pt")
    assert not os.path.exists(download[2]), "the downloaded tensor file goes after the load"


@pytest.mark.parametrize("tensors", [False, True], ids=["saved-locally-without-tensors", "saved-to-s3-with-tensors"])
def test_a_pickle_on_s3_without_tensor_references_is_read_alone(tmp_path, fake_s3, tensors):
    """A model saved to S3 keeps its tensors in the pickle, which is read into memory as before."""
    state = {"cache": torch.randn(64, 8)} if tensors else {"weights": np.arange(3)}
    model = _model(tmp_path / "local", state=state)
    if tensors:  # what a save to an S3 path writes
        (fake_s3.root / "m").mkdir(parents=True)
        (fake_s3.root / "m" / model.model_file_name).write_bytes(pickle.dumps(model, protocol=_tensor_pickle.PROTOCOL))
    else:
        shutil.copytree(model.save(), fake_s3.root / "m")

    loaded = _Model.load("s3://bucket/m/").model

    for name, value in state.items():
        np.testing.assert_array_equal(np.asarray(loaded[name]), np.asarray(value))
    assert fake_s3.requests == [("get", "m/model.pkl")]


def test_a_missing_tensor_file_on_s3_fails_clearly(tmp_path, fake_s3):
    shutil.copytree(_model(tmp_path / "local", state={"cache": torch.ones(8)}).save(), fake_s3.root / "m")
    os.remove(fake_s3.root / "m" / _Model.model_tensors_file_name)

    with pytest.raises(FileNotFoundError, match="s3://bucket/m/model_tensors.pt"):
        _Model.load("s3://bucket/m/")


def test_a_tensor_file_on_s3_that_cannot_be_read_fails_naming_it(tmp_path, fake_s3):
    shutil.copytree(_model(tmp_path / "local", state={"cache": torch.ones(8)}).save(), fake_s3.root / "m")
    fake_s3.denied.add("m/model_tensors.pt")

    with pytest.raises(OSError, match="s3://bucket/m/model_tensors.pt.*403") as raised:
        _Model.load("s3://bucket/m/")
    assert not isinstance(raised.value, FileNotFoundError), "a 403 may also mean the file exists but is not readable"
