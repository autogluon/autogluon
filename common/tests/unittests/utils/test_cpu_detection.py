import os
from unittest.mock import MagicMock, patch

from autogluon.common.utils.cpu_utils import get_available_cpu_count, get_cpu_count_cgroup
from autogluon.common.utils.resource_utils import ResourceManager


def test_get_cpu_count_matches_available_count():
    """Verify ResourceManager.get_cpu_count() uses our new logic"""
    assert ResourceManager.get_cpu_count() == get_available_cpu_count()


@patch.dict(os.environ, {"AG_CPU_COUNT": "4"}, clear=True)
def test_ag_cpu_count_environment_variable():
    """Test that AG_CPU_COUNT environment variable is respected"""
    assert get_available_cpu_count() == 4


@patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "6"}, clear=True)
def test_slurm_cpus_per_task_environment_variable():
    """Test that SLURM_CPUS_PER_TASK environment variable is respected"""
    assert get_available_cpu_count() == 6


@patch.dict(os.environ, {"AG_CPU_COUNT": "4", "SLURM_CPUS_PER_TASK": "8"}, clear=True)
def test_ag_cpu_count_takes_precedence():
    """Test that AG_CPU_COUNT takes precedence over SLURM_CPUS_PER_TASK"""
    assert get_available_cpu_count() == 4


@patch("os.path.exists")
@patch("builtins.open")
def test_cgroup_v2_detection(mock_open, mock_exists):
    """Test that cgroup v2 CPU limits are detected properly"""
    # Mock cgroup v2 file
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/cpu.max"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "200000 100000"  # 2 CPUs
    mock_open.return_value = mock_file

    assert get_cpu_count_cgroup(8) == 2


@patch("os.path.exists")
@patch("builtins.open")
def test_cgroup_v2_no_limit(mock_open, mock_exists):
    """Test that cgroup v2 'max' quota means no limit"""
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/cpu.max"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "max 100000"
    mock_open.return_value = mock_file

    assert get_cpu_count_cgroup(8) == 8  # Should return original count


@patch("os.path.exists")
@patch("builtins.open")
def test_cgroup_v1_detection(mock_open, mock_exists):
    """Test that cgroup v1 CPU limits are detected properly"""
    # Mock cgroup v1 files
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

    assert get_cpu_count_cgroup(8) == 3


@patch("os.path.exists")
@patch("builtins.open")
def test_cgroup_v1_no_limit(mock_open, mock_exists):
    """Test that cgroup v1 quota -1 means no limit"""
    mock_exists.side_effect = lambda path: path in [
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    ]
    mock_files = {"quota": MagicMock(), "period": MagicMock()}
    mock_files["quota"].__enter__.return_value.read.return_value = "-1"
    mock_files["period"].__enter__.return_value.read.return_value = "100000"

    def mock_open_side_effect(filename, *args, **kwargs):
        if filename == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
            return mock_files["quota"]
        elif filename == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
            return mock_files["period"]

    mock_open.side_effect = mock_open_side_effect

    assert get_cpu_count_cgroup(8) == 8  # Should return original count


@patch.dict(os.environ, {}, clear=True)
@patch("multiprocessing.cpu_count")
@patch("os.sched_getaffinity")
@patch("autogluon.common.utils.cpu_utils.get_cpu_count_cgroup")
def test_returns_minimum_from_all_methods(mock_cgroup, mock_affinity, mock_mp_count):
    """Test that we return the minimum CPU count from all detection methods"""
    # Mock different values from different methods
    mock_mp_count.return_value = 16
    mock_affinity.return_value = set(range(8))  # 8 CPUs in affinity
    mock_cgroup.return_value = 4  # This should be the minimum

    # The final result should be the minimum of all methods
    assert get_available_cpu_count() == 4
