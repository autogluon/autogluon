import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run_single_gpu", action="store_true", default=False, help="run single-gpu tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "single_gpu: mark test as single-gpu to run")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_sgpu = pytest.mark.skip(reason="need --run_single_gpu option to run")
    custom_markers = dict(slow=skip_slow, single_gpu=skip_sgpu)
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        custom_markers.pop("slow", None)
    if config.getoption("--run_single_gpu", None):
        # --runslow given in cli: do not skip single-gpu tests
        custom_markers.pop("single_gpu", None)
    for item in items:
        for marker in custom_markers:
            if marker in item.keywords:
                item.add_marker(custom_markers[marker])
