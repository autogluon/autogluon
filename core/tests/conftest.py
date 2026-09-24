import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--runplatform", action="store_true", default=False, help="run all skipped platform tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "platform: mark test as Ubuntu/Linux/Mac platform test")
    plugin = config.pluginmanager.getplugin("mypy")
    plugin.mypy_argv.append("--ignore-missing-imports")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_platform = pytest.mark.skip(reason="need --runplatform option to run")
    custom_markers = dict(slow=skip_slow, platform=skip_platform)
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        custom_markers.pop("slow", None)
    if config.getoption("--runplatform"):
        # --runplatform given in cli: do not skip platform tests
        custom_markers.pop("platform", None)
    for item in items:
        for marker in custom_markers:
            if marker in item.keywords:
                item.add_marker(custom_markers[marker])


@pytest.fixture(autouse=True, scope="session")
def _ag_default_base_path(tmp_path_factory):
    """Redirect auto-generated model paths into pytest's temporary directory.

    Without this, every model or predictor created without an explicit `path` leaves an
    `AutogluonModels/ag-<timestamp>` directory behind in the working directory. pytest reclaims
    its own tmp dirs (keeping only the last few sessions), so nothing accumulates in the repo.
    """
    base_path = tmp_path_factory.mktemp("ag_models")
    prev = os.environ.get("AG_DEFAULT_BASE_PATH")
    os.environ["AG_DEFAULT_BASE_PATH"] = str(base_path)
    yield str(base_path)
    if prev is None:
        os.environ.pop("AG_DEFAULT_BASE_PATH", None)
    else:
        os.environ["AG_DEFAULT_BASE_PATH"] = prev
