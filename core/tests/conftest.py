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
