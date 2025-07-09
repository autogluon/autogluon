"""CPU utilities for accurate resource detection in constrained environments.

This module provides functions for determining the correct number of available CPU cores
in containerized environments, SLURM clusters, and other resource-constrained systems.
"""

import logging
import math
import os

import loky
import psutil

logger = logging.getLogger(__name__)


def get_cpu_count_cgroup_fallback(os_cpu_count):
    """Fallback cgroup detection with robust error handling.

    This provides a fallback when loky's cgroup detection fails due to
    IO errors, permission issues, or corrupted cgroup files.

    Parameters
    ----------
    os_cpu_count : int
        Default CPU count to return if no cgroup limits are found

    Returns
    -------
    int
        CPU count based on cgroup limits or the provided os_cpu_count if no limits found
    """
    cpu_max_fname = "/sys/fs/cgroup/cpu.max"
    cfs_quota_fname = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_fname = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

    if os.path.exists(cpu_max_fname):
        # cgroup v2
        try:
            with open(cpu_max_fname) as fh:
                cpu_quota_us, cpu_period_us = fh.read().strip().split()
                if cpu_quota_us == "max":
                    return os_cpu_count
                else:
                    cpu_quota_us = int(cpu_quota_us)
                    cpu_period_us = int(cpu_period_us)
                    if cpu_quota_us > 0 and cpu_period_us > 0:
                        cgroup_count = math.ceil(cpu_quota_us / cpu_period_us)
                        logger.debug(f"Fallback detected cgroup v2 CPU limit: {cgroup_count}")
                        return cgroup_count
        except (IOError, ValueError, OSError) as e:
            logger.debug(f"Fallback error reading cgroup v2 CPU limit: {e}")

    elif os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        # cgroup v1
        try:
            with open(cfs_quota_fname) as fh:
                cpu_quota_us = fh.read().strip()
            with open(cfs_period_fname) as fh:
                cpu_period_us = fh.read().strip()

            if cpu_quota_us != "-1":  # No limit is set to -1
                cpu_quota_us = int(cpu_quota_us)
                cpu_period_us = int(cpu_period_us)
                if cpu_quota_us > 0 and cpu_period_us > 0:
                    cgroup_count = math.ceil(cpu_quota_us / cpu_period_us)
                    logger.debug(f"Fallback detected cgroup v1 CPU limit: {cgroup_count}")
                    return cgroup_count
        except (IOError, ValueError, OSError) as e:
            logger.debug(f"Fallback error reading cgroup v1 CPU limit: {e}")

    return os_cpu_count


def get_available_cpu_count(only_physical_cores=False):
    """
    Get the number of available CPU cores, respecting container limits,
    CPU affinity, and environment variables.

    This function uses a hybrid approach:
    1. Environment variables (AG_CPU_COUNT, SLURM_CPUS_PER_TASK) - highest priority
    2. Loky's CPU detection (primary method, handles most cases)
    3. Fallback cgroup detection (for edge cases where loky fails)

    Parameters
    ----------
    only_physical_cores : bool, default=False
        If True, detects only physical CPU cores (not including hyperthreading/SMT).
        This can be beneficial for CPU-intensive tasks like time series forecasting
        where physical cores often provide better performance than logical cores.

    Returns
    -------
    int
        The number of available CPU cores.
    """
    # 1. Check environment variables first (highest priority)
    env_var_names = ["AG_CPU_COUNT", "SLURM_CPUS_PER_TASK"]
    for var_name in env_var_names:
        if var_name in os.environ:
            try:
                count = int(os.environ[var_name])
                if count > 0:
                    logger.debug(f"{var_name} environment variable detected: {count}")
                    # Environment variable overrides everything
                    return count
            except ValueError:
                pass

    # 2. Try loky's CPU count (primary method)
    try:
        loky_count = loky.cpu_count(only_physical_cores=only_physical_cores)
        logger.debug(f"loky.cpu_count(only_physical_cores={only_physical_cores}): {loky_count}")

        # Ensure we never return less than 1 to avoid issues
        result = max(1, loky_count)
        logger.debug(f"Final CPU count (loky): {result}")
        return result

    except Exception as e:
        logger.debug(f"loky.cpu_count() failed: {e}, falling back to custom detection")

        # 3. Fallback to custom cgroup detection for edge cases
        default_count = psutil.cpu_count(logical=not only_physical_cores) or psutil.cpu_count()
        result = get_cpu_count_cgroup_fallback(default_count)
        logger.debug(f"Final CPU count (fallback): {result}")
        return max(1, result)
