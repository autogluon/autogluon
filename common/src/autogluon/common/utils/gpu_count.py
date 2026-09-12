"""The CUDA device count torch would report, computed without importing torch.

`torch.cuda.device_count()` reads the device list from NVML and applies `CUDA_VISIBLE_DEVICES`
itself; this module does the same through `nvutil`, plus the one input NVML cannot see: whether
the installed torch was built with CUDA at all. Importing torch costs hundreds of milliseconds
to seconds, which a fit without GPU models should not pay for a count.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys

from . import nvutil

#: `CUDA_VISIBLE_DEVICES` unset: torch treats every device as visible.
_MAX_DEVICES = 64


@functools.lru_cache(maxsize=1)
def torch_version_info() -> tuple[str, str | None] | None:
    """The installed torch's `(__version__, cuda)`, or None when torch is not installed.

    Read from torch's `version.py`, which is a few assignments, rather than by importing torch;
    `cuda` is the CUDA version the build targets, None for a CPU-only build.
    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    version_file = os.path.join(list(spec.submodule_search_locations)[0], "version.py")
    if not os.path.exists(version_file):
        return None
    namespace: dict = {}
    with open(version_file) as f:
        exec(f.read(), namespace)
    return str(namespace.get("__version__")), namespace.get("cuda")


def torch_build_has_cuda() -> bool | None:
    """Whether the installed torch was built with CUDA; None when torch is not installed."""
    info = torch_version_info()
    return None if info is None else info[1] is not None


def _strtoul(s: str) -> int:
    """The non-negative integer `s` starts with, else -1, as the CUDA driver parses ordinals."""
    if not s:
        return -1
    idx = 0
    for idx, c in enumerate(s):
        if not (c.isdigit() or (idx == 0 and c in "+-")):
            break
        if idx + 1 == len(s):
            idx += 1
    return int(s[:idx]) if idx > 0 else -1


def _parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
    ids: list[str] = []
    for elem in lst.split(","):
        # A repeated id results in an empty set.
        if elem in ids:
            return []
        # Anything but the prefix ends the list.
        if not elem.startswith(prefix):
            break
        ids.append(elem)
    return ids


def parse_visible_devices() -> list[int] | list[str]:
    """`CUDA_VISIBLE_DEVICES` as ordinals, or as `GPU-`/`MIG-` prefixed ids, as the CUDA driver reads it."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")
    if var is None:
        return list(range(_MAX_DEVICES))
    if var.startswith("GPU-"):
        return _parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return _parse_list_with_prefix(var, "MIG-")
    ordinals: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # A repeated ordinal results in an empty set.
        if x in ordinals:
            return []
        # A negative value ends the list.
        if x < 0:
            break
        ordinals.append(x)
    return ordinals


def _uuid_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    """Ordinals of the devices whose UUID starts with each candidate; an ambiguous or unknown one ends the list."""
    ordinals: list[int] = []
    for candidate in candidates:
        matches = [idx for idx, uuid in enumerate(uuids) if uuid.startswith(candidate)]
        if len(matches) != 1:
            break
        if matches[0] in ordinals:
            return []
        ordinals.append(matches[0])
    return ordinals


#: Count per `CUDA_VISIBLE_DEVICES` value: the only thing that changes a process's visible devices,
#: and an NVML query costs a 15 ms init/shutdown cycle.
_count_cache: dict[str | None, int | None] = {}


def cuda_visible_device_count() -> int | None:
    """Number of CUDA devices visible to a process, from NVML and `CUDA_VISIBLE_DEVICES`.

    None when NVML is unavailable or the ids are MIG partitions, which NVML does not enumerate
    the way the driver does; callers then fall back to torch.
    """
    key = os.getenv("CUDA_VISIBLE_DEVICES")
    if key not in _count_cache:
        _count_cache[key] = _cuda_visible_device_count()
    return _count_cache[key]


def _cuda_visible_device_count() -> int | None:
    visible = parse_visible_devices()
    if not visible:
        return 0
    if not nvutil.cudaInit():
        return None
    try:
        if isinstance(visible[0], str):
            if visible[0].startswith("MIG-"):
                return None
            return len(_uuid_ordinals(visible, nvutil.cudaDeviceGetUUIDs()))
        raw_count = nvutil.cudaDeviceGetCount()
        for idx, ordinal in enumerate(visible):
            if ordinal >= raw_count:
                return idx
        return len(visible)
    except nvutil.NVMLError:
        return None
    finally:
        nvutil.cudaShutdown()


def visible_device_memory() -> list[tuple[int, int, int]] | None:
    """`(total, free, used)` bytes of each visible CUDA device, in visible order, or None when only torch can tell.

    Visible ordinals name NVML devices directly; UUID and MIG ids are left to torch.
    """
    visible = parse_visible_devices()
    if not visible:
        return []
    if isinstance(visible[0], str) or not nvutil.cudaInit():
        return None
    try:
        raw_count = nvutil.cudaDeviceGetCount()
        indices = []
        for ordinal in visible:
            if ordinal >= raw_count:
                break
            indices.append(ordinal)
        return [nvutil.cudaDeviceGetMemoryInfo(index) for index in indices]
    except nvutil.NVMLError:
        return None
    finally:
        nvutil.cudaShutdown()


def gpu_count_without_torch(cuda_only: bool = False) -> int | None:
    """The count `ResourceManager.get_gpu_count_torch` would return, or None when only torch can tell.

    None is returned when torch is not installed, when NVML gives no answer, and on macOS when
    the CUDA count is zero and other accelerators (MPS) are allowed, since only torch reports those.
    """
    has_cuda = torch_build_has_cuda()
    if has_cuda is None:
        return None
    if not has_cuda:
        count = 0
    else:
        count = cuda_visible_device_count()
        if count is None:
            return None
    if count == 0 and not cuda_only and sys.platform == "darwin":
        return None
    return count
