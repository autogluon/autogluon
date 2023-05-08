import pytest

from autogluon.common.utils.version_utils import VersionManager
from contextlib import contextmanager


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    plugin = config.pluginmanager.getplugin("mypy")
    plugin.mypy_argv.append("--ignore-missing-imports")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@contextmanager
def mock_ag_version(mocked_version: str):
    original_get_ag_version = VersionManager.get_ag_version
    VersionManager.get_ag_version = lambda *args, **kwargs: mocked_version
    yield
    VersionManager.get_ag_version = original_get_ag_version


@pytest.fixture
def mock_ag_version_mgr():
    return mock_ag_version
