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
storage keep sharing it). :func:`load` resolves the references from that file, memory-mapped on
request. :func:`pickled_size` streams the pickle into a byte counter and adds every storage's bytes
once, without serializing any tensor.

An object without tensors is pickled exactly as before and gets no tensor file. In a process that
never imported torch no object can hold a tensor, so torch is never imported here for it.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from typing import Any

from autogluon.common.utils import s3_utils

from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ._shared_weights_registry import tensor_bytes

logger = logging.getLogger(__name__)

#: The pickle protocol of ``save_pkl``, so a model's size estimate matches what ``save`` writes.
PROTOCOL = 4
_TENSOR_REF = "autogluon.tensor"


def _imported_torch():
    """torch when this process has imported it, else ``None``."""
    return sys.modules.get("torch")


class _TensorPickler(pickle.Pickler):
    """Pickles every tensor as a reference to its index in :attr:`tensors`."""

    def __init__(self, file, tensor_type: type, protocol: int = PROTOCOL):
        super().__init__(file, protocol=protocol)
        self._tensor_type = tensor_type
        self.tensors: list = []
        self._index_by_id: dict[int, int] = {}

    def persistent_id(self, obj: Any):
        if not isinstance(obj, self._tensor_type):
            return None
        # The same tensor object referenced twice gets one index, so the load restores one object.
        index = self._index_by_id.get(id(obj))
        if index is None:
            index = self._index_by_id[id(obj)] = len(self.tensors)
            self.tensors.append(obj)
        return _TENSOR_REF, index


class _TensorUnpickler(pickle.Unpickler):
    def __init__(self, file, tensors: list):
        super().__init__(file)
        self._tensors = tensors

    def persistent_load(self, pid: Any):
        if not (isinstance(pid, tuple) and len(pid) == 2 and pid[0] == _TENSOR_REF):
            raise pickle.UnpicklingError(f"Unknown persistent id in model pickle: {pid!r}")
        return self._tensors[pid[1]]


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
    is removed otherwise. An S3 ``path`` keeps the tensors inside the pickle.
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
    with open(path, "wb") as f:
        pickler = _TensorPickler(f, tensor_type=torch.Tensor)
        pickler.dump(obj)
    if not pickler.tensors:
        _remove(tensors_path)
        return
    # Written next to the target and then swapped in: a model loaded with ``mmap`` reads its tensors
    # from the old file, and rewriting that file in place would invalidate the mapping (SIGBUS).
    tmp_path = tensors_path + ".tmp"
    torch.save(pickler.tensors, tmp_path)
    os.replace(tmp_path, tensors_path)


def load(path: str, tensors_path: str, mmap: bool = False, verbose: bool = True) -> Any:
    """Load what :func:`save` wrote; with ``mmap``, the tensors stay mapped from ``tensors_path``.

    Tensors saved on a CUDA device are loaded on the CPU when CUDA is not available. ``mmap`` is
    ignored on Windows, where a mapped file cannot be replaced by a later save.
    """
    if s3_utils.is_s3_url(path) or not os.path.exists(tensors_path):
        return load_pkl.load(path=path, verbose=verbose)
    import torch

    if verbose:
        logger.log(15, f"Loading: {path}")
    # The tensor file is as trusted as the pickle beside it, which is unpickled without restrictions.
    tensors = torch.load(
        tensors_path,
        map_location=None if torch.cuda.is_available() else "cpu",
        mmap=mmap and os.name != "nt",
        weights_only=False,
    )
    with open(path, "rb") as f:
        return _TensorUnpickler(f, tensors).load()


def pickled_size(obj: Any, protocol: int = PROTOCOL) -> int:
    """Bytes of the pickle of ``obj`` plus the bytes of its tensors' storages, each storage counted once.

    No copy of ``obj`` is made: the pickle is streamed into a byte counter and the tensors are
    counted, not serialized.
    """
    counter = _ByteCounter()
    torch = _imported_torch()
    if torch is None:
        pickle.Pickler(counter, protocol=protocol).dump(obj)
        return counter.nbytes
    pickler = _TensorPickler(counter, tensor_type=torch.Tensor, protocol=protocol)
    pickler.dump(obj)
    return counter.nbytes + (tensor_bytes(pickler.tensors) if pickler.tensors else 0)
