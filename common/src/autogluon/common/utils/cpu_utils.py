"""CPU utilities for accurate resource detection in constrained environments.

This module provides functions for determining the correct number of available CPU cores
in containerized environments, SLURM clusters, and other resource-constrained systems.
"""

import os
import math
import multiprocessing

import logging
import loky
import psutil

logger = logging.getLogger(__name__)


def get_cpu_count_cgroup(os_cpu_count):
    """Get CPU count from cgroup limits (commonly used in Docker)

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
                        logger.debug(f"Detected cgroup v2 CPU limit: {cgroup_count}")
                        return cgroup_count
        except (IOError, ValueError) as e:
            logger.debug(f"Error reading cgroup v2 CPU limit: {e}")

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
                    logger.debug(f"Detected cgroup v1 CPU limit: {cgroup_count}")
                    return cgroup_count
        except (IOError, ValueError) as e:
            logger.debug(f"Error reading cgroup v1 CPU limit: {e}")

    return os_cpu_count


def get_available_cpu_count():
    """
    Get the number of available CPU cores, respecting container limits,
    CPU affinity, and environment variables.

    This function checks multiple sources to determine the true number of CPUs
    available to the current process:

    1. Environment variables (AG_CPU_COUNT, SLURM_CPUS_PER_TASK)
    2. Loky's CPU detection
    3. CPU affinity via os.sched_getaffinity
    4. CPU affinity via psutil
    5. cgroup limits (for Docker containers)
    6. Default multiprocessing.cpu_count()

    Returns
    -------
    int
        The number of available CPU cores. Returns the minimum count from
        all successful detection methods to avoid CPU oversubscription.
    """
    # Start with system reported CPU count as the default
    default_cpu_count = multiprocessing.cpu_count()
    available_counts = [default_cpu_count]

    # Log all detected values for debugging
    logger.debug(f"System default CPU count: {default_cpu_count}")

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

    # 2. Try loky's CPU count which handles various container scenarios
    loky_count = loky.cpu_count()
    logger.debug(f"loky.cpu_count(): {loky_count}")
    available_counts.append(loky_count)

    # 3. Check CPU affinity using os.sched_getaffinity
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
            logger.debug(f"CPU affinity count (os.sched_getaffinity): {affinity_count}")
            available_counts.append(affinity_count)
        except Exception as e:
            logger.debug(f"Error getting CPU affinity via os.sched_getaffinity: {e}")

    # 4. Try CPU affinity using psutil as a fallback
    try:
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            psutil_affinity_count = len(p.cpu_affinity())
            logger.debug(f"CPU affinity count (psutil): {psutil_affinity_count}")
            available_counts.append(psutil_affinity_count)
    except Exception as e:
        logger.debug(f"Error getting CPU affinity via psutil: {e}")

    # 5. Check cgroup limits (Docker containers with --cpus=N parameter)
    cgroup_count = get_cpu_count_cgroup(default_cpu_count)
    if cgroup_count != default_cpu_count:
        logger.debug(f"cgroup CPU count: {cgroup_count}")
        available_counts.append(cgroup_count)

    # Return the minimum count to avoid oversubscription
    # but never return less than 1 to avoid issues
    result = max(1, min(available_counts))
    logger.debug(f"Final CPU count after checking all methods: {result}")

    return result
