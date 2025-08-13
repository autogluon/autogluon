import os
from unittest.mock import patch

from autogluon.common.utils.cpu_utils import get_available_cpu_count
from autogluon.common.utils.resource_utils import ResourceManager


def test_get_cpu_count_matches_available_count():
    """Verify ResourceManager.get_cpu_count() uses our simplified logic"""
    assert ResourceManager.get_cpu_count() == get_available_cpu_count()


def test_get_cpu_count_matches_available_count_physical_cores():
    """Verify ResourceManager.get_cpu_count() uses our simplified logic for physical cores"""
    assert ResourceManager.get_cpu_count(only_physical_cores=True) == get_available_cpu_count(only_physical_cores=True)


@patch.dict(os.environ, {"AG_CPU_COUNT": "4"}, clear=True)
def test_ag_cpu_count_environment_variable():
    """Test that AG_CPU_COUNT environment variable is respected"""
    assert get_available_cpu_count() == 4


@patch.dict(os.environ, {"AG_CPU_COUNT": "4"}, clear=True)
def test_ag_cpu_count_environment_variable_overrides_physical_cores():
    """Test that AG_CPU_COUNT environment variable overrides only_physical_cores"""
    assert get_available_cpu_count(only_physical_cores=True) == 4


@patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "6"}, clear=True)
def test_slurm_cpus_per_task_environment_variable():
    """Test that SLURM_CPUS_PER_TASK environment variable is respected"""
    assert get_available_cpu_count() == 6


@patch.dict(os.environ, {"AG_CPU_COUNT": "4", "SLURM_CPUS_PER_TASK": "8"}, clear=True)
def test_ag_cpu_count_takes_precedence():
    """Test that AG_CPU_COUNT takes precedence over SLURM_CPUS_PER_TASK"""
    assert get_available_cpu_count() == 4


@patch("joblib.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_loky_logical_cores_detection(mock_loky_cpu_count):
    """Test that joblib.cpu_count() is used for logical cores"""
    mock_loky_cpu_count.return_value = 8

    result = get_available_cpu_count(only_physical_cores=False)

    mock_loky_cpu_count.assert_called_with(only_physical_cores=False)
    assert result == 8


@patch("joblib.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_loky_physical_cores_detection(mock_loky_cpu_count):
    """Test that joblib.cpu_count() is used for physical cores"""
    mock_loky_cpu_count.return_value = 4

    result = get_available_cpu_count(only_physical_cores=True)

    mock_loky_cpu_count.assert_called_with(only_physical_cores=True)
    assert result == 4


@patch("joblib.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_minimum_cpu_count_is_one(mock_loky_cpu_count):
    """Test that we never return less than 1 CPU"""
    mock_loky_cpu_count.return_value = 0  # Edge case

    result = get_available_cpu_count()

    # Should return at least 1
    assert result == 1


def test_normal_operation():
    """Test that the function works under normal conditions"""
    # This should work without any mocking
    logical = get_available_cpu_count(only_physical_cores=False)
    physical = get_available_cpu_count(only_physical_cores=True)

    assert logical > 0
    assert physical > 0
    assert logical >= physical  # Logical should be >= physical cores
