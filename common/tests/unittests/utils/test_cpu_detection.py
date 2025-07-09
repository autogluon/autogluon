import os
from unittest.mock import MagicMock, patch

from autogluon.common.utils.cpu_utils import get_available_cpu_count, get_cpu_count_cgroup_fallback
from autogluon.common.utils.resource_utils import ResourceManager


def test_get_cpu_count_matches_available_count():
    """Verify ResourceManager.get_cpu_count() uses our new logic"""
    assert ResourceManager.get_cpu_count() == get_available_cpu_count()


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


@patch("loky.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_loky_primary_method_physical_cores(mock_loky_cpu_count):
    """Test that loky.cpu_count() is used as primary method for physical cores"""
    mock_loky_cpu_count.return_value = 4

    result = get_available_cpu_count(only_physical_cores=True)

    # Should call loky.cpu_count(only_physical_cores=True)
    mock_loky_cpu_count.assert_called_with(only_physical_cores=True)
    assert result == 4


@patch("loky.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_loky_primary_method_logical_cores(mock_loky_cpu_count):
    """Test that loky.cpu_count() is used as primary method for logical cores"""
    mock_loky_cpu_count.return_value = 8

    result = get_available_cpu_count(only_physical_cores=False)

    # Should call loky.cpu_count(only_physical_cores=False)
    mock_loky_cpu_count.assert_called_with(only_physical_cores=False)
    assert result == 8


@patch("loky.cpu_count")
@patch("autogluon.common.utils.cpu_utils.get_cpu_count_cgroup_fallback")
@patch.dict(os.environ, {}, clear=True)
def test_fallback_when_loky_fails(mock_fallback, mock_loky_cpu_count):
    """Test that fallback cgroup detection is used when loky fails"""
    mock_loky_cpu_count.side_effect = Exception("Loky failed")
    mock_fallback.return_value = 2

    result = get_available_cpu_count()

    # Should call fallback method
    mock_fallback.assert_called_once()
    assert result == 2


@patch("os.path.exists")
@patch("builtins.open")
def test_fallback_cgroup_v2_detection(mock_open, mock_exists):
    """Test that fallback cgroup v2 detection works with robust error handling"""
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/cpu.max"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "200000 100000"  # 2 CPUs
    mock_open.return_value = mock_file

    assert get_cpu_count_cgroup_fallback(8) == 2


@patch("os.path.exists")
@patch("builtins.open")
def test_fallback_cgroup_v2_io_error_handling(mock_open, mock_exists):
    """Test that fallback cgroup detection handles IO errors gracefully"""
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/cpu.max"
    mock_open.side_effect = IOError("Permission denied")

    # Should return original count on error
    assert get_cpu_count_cgroup_fallback(8) == 8


@patch("os.path.exists")
@patch("builtins.open")
def test_fallback_cgroup_v1_detection(mock_open, mock_exists):
    """Test that fallback cgroup v1 detection works with robust error handling"""
    mock_exists.side_effect = lambda path: path in [
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    ]
    mock_files = {"quota": MagicMock(), "period": MagicMock()}
    mock_files["quota"].__enter__.return_value.read.return_value = "300000"  # 3 CPUs
    mock_files["period"].__enter__.return_value.read.return_value = "100000"

    def mock_open_side_effect(filename, *args, **kwargs):
        if filename == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
            return mock_files["quota"]
        elif filename == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
            return mock_files["period"]

    mock_open.side_effect = mock_open_side_effect

    assert get_cpu_count_cgroup_fallback(8) == 3


@patch("loky.cpu_count")
@patch.dict(os.environ, {}, clear=True)
def test_minimum_cpu_count_is_one(mock_loky_cpu_count):
    """Test that we never return less than 1 CPU"""
    mock_loky_cpu_count.return_value = 0  # Edge case

    result = get_available_cpu_count()

    # Should return at least 1
    assert result == 1
