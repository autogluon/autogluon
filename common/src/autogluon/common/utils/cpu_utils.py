"""CPU utilities for accurate resource detection in constrained environments.

This module provides functions for determining the correct number of available CPU cores
in containerized environments, SLURM clusters, and other resource-constrained systems.
"""

import logging
import os

import joblib

logger = logging.getLogger(__name__)


def get_available_cpu_count(only_physical_cores: bool = False) -> int:
    """
    Get the number of available CPU cores, respecting container limits,
    CPU affinity, and environment variables.

    Uses loky for robust CPU detection in containerized environments.
    If loky fails, the error will be clear and users can set AG_CPU_COUNT
    environment variable as an override.

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
                    return count
            except ValueError:
                pass

    # 2. Use joblib's robust CPU detection
    result = joblib.cpu_count(only_physical_cores=only_physical_cores)
    logger.debug(f"loky.cpu_count(only_physical_cores={only_physical_cores}): {result}")

    # Ensure we never return less than 1
    return max(1, result)
