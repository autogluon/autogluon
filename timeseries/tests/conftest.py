import shutil
import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

    # known gluonts warnings
    config.addinivalue_line("filterwarnings", "ignore:Using `json`-module:UserWarning")

    # pandas future warnings on timestamp freq being deprecated
    config.addinivalue_line("filterwarnings", "ignore:.+freq:FutureWarning")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="function")
def temp_model_path():
    """Pytest fixture to save as model paths that clean up after themselves"""
    td = tempfile.mkdtemp()
    yield td
    shutil.rmtree(td)
