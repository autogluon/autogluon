import os
from unittest import mock
from unittest.mock import mock_open, patch

from autogluon.common.utils import memory_utils
from autogluon.common.utils.memory_utils import get_cgroup_memory_limit


def test_cgroup_v2_limit_is_read():
    with patch("builtins.open", mock_open(read_data="16000000000\n")):
        with patch("os.path", wraps=os.path):
            assert get_cgroup_memory_limit() == 16000000000


def test_cgroup_v2_max_means_no_limit():
    # cgroup v2 reports the literal string "max" when no limit is set.
    with patch("builtins.open", mock_open(read_data="max\n")):
        assert get_cgroup_memory_limit() is None


def test_cgroup_v1_huge_sentinel_means_no_limit():
    # cgroup v1 reports a near-uint64-max sentinel when no limit is set.
    with patch("builtins.open", mock_open(read_data="9223372036854771712\n")):
        assert get_cgroup_memory_limit() is None


def test_cgroup_missing_files_returns_none():
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert get_cgroup_memory_limit() is None


def test_cgroup_non_integer_returns_none():
    with patch("builtins.open", mock_open(read_data="not-a-number\n")):
        assert get_cgroup_memory_limit() is None


def test_memory_size_capped_by_cgroup_limit():
    import psutil

    _mock_vem = psutil.virtual_memory()
    with patch.object(psutil, "virtual_memory", lambda: _mock_vem):
        from autogluon.common.utils.resource_utils import ResourceManager

        cgroup_limit = int(_mock_vem.total // 4)  # a limit lower than the host total
        with patch.object(ResourceManager, "_get_memory_size", wraps=ResourceManager._get_memory_size):
            with patch("autogluon.common.utils.resource_utils.get_cgroup_memory_limit", return_value=cgroup_limit):
                with patch.dict(os.environ, {}, clear=True):
                    assert ResourceManager._get_memory_size() == cgroup_limit


def test_memory_size_uses_host_when_no_cgroup_limit():
    import psutil

    _mock_vem = psutil.virtual_memory()
    with patch.object(psutil, "virtual_memory", lambda: _mock_vem):
        from autogluon.common.utils.resource_utils import ResourceManager

        with patch("autogluon.common.utils.resource_utils.get_cgroup_memory_limit", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                assert ResourceManager._get_memory_size() == _mock_vem.total


def test_env_var_takes_precedence_over_cgroup():
    from autogluon.common.utils.resource_utils import ResourceManager

    with patch("autogluon.common.utils.resource_utils.get_cgroup_memory_limit", return_value=1):
        with patch.dict(os.environ, {"AG_MEMORY_LIMIT_IN_GB": "48"}, clear=True):
            # 48 GB in bytes, not the 1-byte cgroup value
            assert ResourceManager._get_memory_size() == max(int(48 * (1024.0**3)), 1)
