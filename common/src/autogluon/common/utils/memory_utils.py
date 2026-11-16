"""Memory utilities for accurate resource detection in constrained environments.

This module provides functions for determining the correct amount of available memory
in containerized environments (e.g. Docker, Kubernetes) where a cgroup memory limit may
be set that is lower than the total memory of the host machine.
"""

import logging
import os

logger = logging.getLogger(__name__)

# cgroup v2 (unified hierarchy) exposes the memory limit here.
_CGROUP_V2_MAX_PATH = "/sys/fs/cgroup/memory.max"
# cgroup v1 exposes the memory limit here.
_CGROUP_V1_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

# When a cgroup has no limit, the kernel reports either the literal string "max" (v2) or a
# very large sentinel close to the unsigned 64-bit maximum (v1). Treat anything at or above
# this threshold as "no limit set".
_UNLIMITED_THRESHOLD = 1 << 62


def _read_cgroup_limit_file(path: str) -> int | None:
    """Read a single cgroup memory-limit file and return the limit in bytes, or ``None``.

    Returns ``None`` when the file does not exist, is unreadable, reports no limit, or does
    not contain a usable positive integer.
    """
    try:
        with open(path) as f:
            content = f.read().strip()
    except (OSError, ValueError):
        return None

    if content == "max":
        return None

    try:
        limit = int(content)
    except ValueError:
        return None

    if limit <= 0 or limit >= _UNLIMITED_THRESHOLD:
        return None

    return limit


def get_cgroup_memory_limit() -> int | None:
    """Return the cgroup memory limit in bytes, or ``None`` if no limit applies.

    Checks the cgroup v2 limit first, then falls back to cgroup v1. Returns ``None`` on any
    non-Linux platform, when no cgroup limit file is present, or when no limit is set.
    """
    for path in (_CGROUP_V2_MAX_PATH, _CGROUP_V1_LIMIT_PATH):
        limit = _read_cgroup_limit_file(path)
        if limit is not None:
            logger.debug(f"Detected cgroup memory limit from {path}: {limit} bytes")
            return limit
    return None
