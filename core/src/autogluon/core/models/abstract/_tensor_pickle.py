"""Pickle a model with its torch tensors in a file of their own.

AutoGluon pickles a fitted model to save it (``model.pkl``) and, in memory, to measure its size
(``AbstractModel._get_pickled_size``), both during the fit. torch pickles a tensor by writing its
storage with ``torch.save`` into an in-memory buffer and passing the buffer's bytes on to the pickle,
so each tensor is copied twice, and ``pickle.dumps`` holds one more copy of the whole model. For a
network that is a fixed cost. For an in-context model that keeps an encoded context or a key/value
cache of its training rows (tens of KB per row and ensemble member), the copies grow with the data:
on a 100k-row table they take tens of GB of RAM and minutes of the fit.

:func:`save` pickles the object with each tensor replaced by a reference and writes the tensors with
``torch.save`` to a second file, which streams every storage to disk once (tensors that view one
storage keep sharing it). Both files carry a token of the save that wrote them, and the pickle is
moved into place last, so a pair from two different saves fails to load instead of mixing tensors.
A state ``torch.save`` refuses (two views of one storage with different dtypes) keeps its tensors in
the pickle. :func:`load` reads the tensor file only when the pickle refers to it, and restores each
tensor on the device it was saved from, or on the CPU when that CUDA device is not available, so a
``__getstate__`` that asks :func:`tensors_saved_apart` can leave its tensors where they are. A tensor kept
in the pickle (a state ``torch.save`` refuses, a save to S3) is restored the same way.
:func:`pickled_size` streams the pickle into a byte counter and adds every storage's bytes once,
without serializing any tensor.

A model saved to S3 keeps its tensors in the pickle. One saved to a local folder and copied to S3
loads from there too: :func:`load` downloads its tensor file when the pickle refers to it.

An object without tensors is pickled byte for byte as before and gets no tensor file. In a process
that never imported torch no object can hold a tensor, so torch is never imported here for it.
"""

from __future__ import annotations

import contextvars
import functools
import io
import logging
import os
import pickle
import sys
import tempfile
import uuid
from typing import Any, Callable

from autogluon.common.utils import s3_utils

from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl

logger = logging.getLogger(__name__)

#: The pickle protocol of ``save_pkl``, so a model's size estimate matches what ``save`` writes.
PROTOCOL = 4


_SAVING_APART = contextvars.ContextVar("autogluon_tensors_saved_apart", default=False)


def tensors_saved_apart() -> bool:
    """Whether the pickle being written keeps its tensors out of it (:func:`save`, :func:`pickled_size`).

    Such a pickle need not move tensors to the CPU so that it loads without CUDA: :func:`load` restores
    them on the CPU when their CUDA device is not available. A plain pickle written meanwhile, from
    inside a ``__getstate__``, sees the same answer.
    """
    return _SAVING_APART.get()


def _imported_torch():
    """torch when this process has imported it, else ``None``."""
    return sys.modules.get("torch")


def _tensor_ref(token: str, index: int):
    """Stands for a tensor in a model pickle; :func:`load` resolves it from the tensor file instead."""
    raise pickle.UnpicklingError(
        "This pickle refers to torch tensors saved in a file beside it (model_tensors.pt). "
        "Load the model with its class's `load`, which reads both files."
    )


class _TensorPickler(pickle.Pickler):
    """Pickles every tensor as a reference to its index in :attr:`tensors`.

    ``reducer_override`` is called only for objects of non-builtin types, so ints, strings, lists
    and dicts are pickled at the speed of the plain pickler. The pickle memoizes each reduced object,
    so a tensor referenced twice is stored once and loads as one object.
    """

    def __init__(self, file, tensor_type: type, token: str = "", protocol: int = PROTOCOL):
        super().__init__(file, protocol=protocol)
        self._tensor_type = tensor_type
        self.token = token
        self.tensors: list = []

    def dump(self, obj: Any) -> None:
        reset = _SAVING_APART.set(True)
        try:
            super().dump(obj)
        finally:
            _SAVING_APART.reset(reset)

    def reducer_override(self, obj: Any):
        if not isinstance(obj, self._tensor_type):
            return NotImplemented
        self.tensors.append(obj)
        return _tensor_ref, (self.token, len(self.tensors) - 1)


class _TensorUnpickler(pickle.Unpickler):
    """Resolves tensor references from the tensor file, read on the first reference, and restores the
    storages the pickle holds itself as the tensor file's are restored."""

    def __init__(self, file, read_tensors: Callable[[], dict], tensors_path: str, pin: bool = False):
        super().__init__(file)
        self._read_tensors = read_tensors  # raises FileNotFoundError for a missing tensor file
        self._tensors_path = tensors_path
        self._pin = pin
        self._restore: Callable | None = None
        self._saved: dict | None = None

    def find_class(self, module: str, name: str):
        if module == __name__ and name == _tensor_ref.__name__:
            return self._resolve
        if module == "torch.storage" and name == "_load_from_bytes":
            return self._load_storage
        return super().find_class(module, name)

    def _load_storage(self, data: bytes):
        """A storage pickled the way torch pickles one, through ``torch.save`` into bytes.

        torch's own ``_load_from_bytes`` restores it where it was saved, which fails without that CUDA
        device; this restores it through :func:`_map_location`, as the tensor file does.
        """
        import torch

        if self._restore is None:
            self._restore = _map_location(self._pin)
        return torch.load(io.BytesIO(data), map_location=self._restore, weights_only=False)

    def _resolve(self, token: str, index: int):
        if self._saved is None:
            self._saved = self._read_tensors()
        if self._saved.get("token") != token:
            raise pickle.UnpicklingError(
                f"{self._tensors_path} was written by a different save than the pickle beside it "
                "(an interrupted or concurrent save); save the model again."
            )
        return self._saved["tensors"][index]


class _ByteCounter:
    """A write-only file that keeps only the number of bytes written."""

    def __init__(self):
        self.nbytes = 0

    def write(self, data) -> int:
        size = memoryview(data).nbytes
        self.nbytes += size
        return size


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def save(obj: Any, path: str, tensors_path: str, verbose: bool = True) -> None:
    """Pickle ``obj`` to ``path`` and write its tensors to ``tensors_path``.

    ``tensors_path`` is written only when ``obj`` holds tensors; a file left there by an earlier save
    is removed otherwise. Both files are written under temporary names of this save and moved into
    place, the pickle last. An S3 ``path``, or tensors ``torch.save`` refuses, stay inside the pickle.
    """
    torch = _imported_torch()
    if s3_utils.is_s3_url(path):
        save_pkl.save(path=path, object=obj, verbose=verbose)
        return
    if torch is None:
        save_pkl.save(path=path, object=obj, verbose=verbose)
        _remove(tensors_path)
        return
    if verbose:
        logger.log(15, f"Saving {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    token = uuid.uuid4().hex
    # Temporary names of this save's own, so saves into one folder at the same time never write into or
    # move each other's files; the token check catches the pair they leave.
    tmp_path, tmp_tensors_path = f"{path}.{token}.tmp", f"{tensors_path}.{token}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickler = _TensorPickler(f, tensor_type=torch.Tensor, token=token)
            pickler.dump(obj)
        if pickler.tensors and _save_tensors(torch, {"token": token, "tensors": pickler.tensors}, tmp_tensors_path):
            # A model whose pinning failed on load still maps the old tensor file; replacing the file keeps
            # that mapping valid, where rewriting it in place would not (SIGBUS).
            os.replace(tmp_tensors_path, tensors_path)
        else:
            if pickler.tensors:
                with open(tmp_path, "wb") as f:
                    pickle.Pickler(f, protocol=PROTOCOL).dump(obj)
            _remove(tensors_path)
        os.replace(tmp_path, path)
    finally:
        _remove(tmp_path)
        _remove(tmp_tensors_path)


def _save_tensors(torch, saved: dict, path: str) -> bool:
    """Write ``saved`` with ``torch.save``; False when torch refuses the tensors, which a pickle accepts.

    The CRC32 of each record, which ``torch.load`` never checks, is skipped; it is about 40% of the save.
    """
    from torch.utils.serialization import config as serialization_config

    try:
        with serialization_config.patch({"save.compute_crc32": False}):
            torch.save(saved, path)
    except RuntimeError as err:
        # Such as "Cannot save multiple tensors or storages that view the same data as different types".
        logger.log(20, f"Keeping the tensors in the pickle, since torch.save refused them: {err}")
        return False
    return True


def _map_location(pin: bool) -> Callable:
    """``torch.load``'s ``map_location``: each storage where it was saved, on the CPU when its CUDA
    device is not available (no CUDA, or fewer devices than at the save), and copied into pinned memory
    when ``pin``, CUDA is available and it is on the CPU. A storage that cannot be pinned stays as loaded."""
    import torch
    from torch.serialization import default_restore_location

    # Asking whether CUDA is available initializes the driver, after which a forked child cannot use CUDA,
    # so only a load that pins or holds CUDA storages asks.
    cuda = functools.cache(torch.cuda.is_available)
    device_count = functools.cache(torch.cuda.device_count)
    unpinned = []

    def has_device(location: str) -> bool:
        index = location.partition(":")[2]
        return cuda() and (not index or int(index) < device_count())

    def restore(storage, location: str):
        if location.startswith("cuda") and not has_device(location):
            restored = storage  # torch hands the storage over on the CPU
        else:
            restored = default_restore_location(storage, location)
        if pin and restored.device.type == "cpu" and cuda():
            try:
                restored = restored.pin_memory()
            except RuntimeError as err:
                if not unpinned:
                    logger.warning(f"Could not pin the model's tensors, keeping them in pageable memory: {err}")
                unpinned.append(restored.nbytes())
        return restored

    return restore


def load(path: str, tensors_path: str, pin: bool = False, verbose: bool = True) -> Any:
    """Load what :func:`save` wrote, from a local folder or from S3.

    The pickle is read through ``load_pkl``, which reads one on S3 into memory. A tensor file on S3 is
    downloaded to a temporary folder and read from there without memory-mapping, so the folder goes
    once the tensors are loaded.

    The tensor file is read only when the pickle refers to it. With ``pin`` and CUDA available, every
    CPU storage is copied from the memory-mapped file into pinned memory, once per storage, so views
    keep sharing it; a storage that cannot be pinned stays mapped. Tensors saved on a CUDA device load
    on the CPU when that device is not available.
    """
    if s3_utils.is_s3_url(path):
        if verbose:
            load_pkl._warn_s3_pickle_load(path)
        read_tensors = functools.partial(_read_tensors_from_s3, tensors_path, pin)
    else:
        read_tensors = functools.partial(_read_tensors, tensors_path, pin)

    def unpickle(file) -> Any:
        return _TensorUnpickler(file, read_tensors=read_tensors, tensors_path=tensors_path, pin=pin).load()

    return load_pkl.load_with_fn(path, unpickle, verbose=verbose)


def _read_tensors(tensors_path: str, pin: bool, may_map: bool = True) -> dict:
    """A local tensor file, memory-mapped when ``may_map`` and its CPU storages are copied into pinned memory."""
    import torch

    if not os.path.exists(tensors_path):
        raise FileNotFoundError(f"The model's tensor file is missing: {tensors_path}")
    mmap = may_map and pin and torch.cuda.is_available() and os.name != "nt"
    # The tensor file is as trusted as the pickle beside it, which is unpickled without restrictions.
    return torch.load(tensors_path, map_location=_map_location(pin), mmap=mmap, weights_only=False)


def _read_tensors_from_s3(tensors_path: str, pin: bool) -> dict:
    """A tensor file on S3, through a temporary folder; an error names the file."""
    from botocore.exceptions import ClientError

    with tempfile.TemporaryDirectory() as local_dir:
        local_path = os.path.join(local_dir, os.path.basename(tensors_path))
        try:
            s3_utils.download_s3_file(local_path=local_path, s3_path=tensors_path)
        except ClientError as err:
            # Without the permission to list the bucket, S3 answers 403 for a missing file as well.
            if err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(f"The model's tensor file is missing: {tensors_path}") from err
            raise OSError(f"Could not download the model's tensor file {tensors_path}: {err}") from err
        return _read_tensors(local_path, pin, may_map=False)


def _tensors_nbytes(torch, tensors: list, protocol: int) -> int:
    """Bytes ``torch.save`` writes for ``tensors``: each dense storage once; other layouts as pickled."""
    seen: set = set()
    total = 0
    for tensor in tensors:
        if tensor.layout == torch.strided and tensor.device.type != "meta":
            try:
                storage = tensor.untyped_storage()
                key, nbytes = (storage.device, storage.data_ptr()), storage.nbytes()
            except (RuntimeError, NotImplementedError):
                key, nbytes = id(tensor), None
        else:
            # Sparse and meta tensors write their components or nothing, not numel * itemsize.
            key, nbytes = id(tensor), None
        if key in seen:
            continue
        seen.add(key)
        total += nbytes if nbytes is not None else len(pickle.dumps(tensor, protocol=protocol))
    return total


def pickled_size(obj: Any, protocol: int = PROTOCOL) -> int:
    """Bytes of the pickle of ``obj`` plus the bytes of its tensors' storages, each storage counted once.

    The pickle is streamed into a byte counter and dense tensors are counted, not serialized.
    """
    counter = _ByteCounter()
    torch = _imported_torch()
    if torch is None:
        pickle.Pickler(counter, protocol=protocol).dump(obj)
        return counter.nbytes
    pickler = _TensorPickler(counter, tensor_type=torch.Tensor, protocol=protocol)
    pickler.dump(obj)
    return counter.nbytes + _tensors_nbytes(torch, pickler.tensors, protocol)
