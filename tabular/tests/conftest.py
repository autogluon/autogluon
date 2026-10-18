from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from autogluon.common.utils.resource_utils import ResourceManager


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--runregression", action="store_true", default=False, help="run regression tests")
    parser.addoption("--runpyodide", action="store_true", default=False, help="run Pyodide tests")
    parser.addoption("--run-multi-gpu", action="store_true", default=False, help="run multi-GPU tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "regression: mark test as regression test")
    config.addinivalue_line("markers", "pyodide: mark test as pyodide test")
    config.addinivalue_line("markers", "multi_gpu: mark test to run on multi-GPU CI only")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_regression = pytest.mark.skip(reason="need --runregression option to run")
    skip_pyodide = pytest.mark.skip(reason="need --runpyodide option to run")
    skip_multi_gpu = pytest.mark.skip(reason="need --run-multi-gpu option to run")
    custom_markers = dict(
        slow=skip_slow,
        regression=skip_regression,
        pyodide=skip_pyodide,
        multi_gpu=skip_multi_gpu,
    )
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        custom_markers.pop("slow", None)
    if config.getoption("--runregression"):
        # --runregression given in cli: do not skip slow tests
        custom_markers.pop("regression", None)
    if config.getoption("--runpyodide"):
        # --runpyodide given in cli: do not skip pyodide tests
        custom_markers.pop("pyodide", None)
    if config.getoption("--run-multi-gpu"):
        # --run-multi-gpu given in cli: do not skip multi-GPU tests
        custom_markers.pop("multi_gpu", None)

    for item in items:
        for marker in custom_markers:
            if marker in item.keywords:
                item.add_marker(custom_markers[marker])

    # Normalize the file paths and use a consistent comparison method
    normalized_path = lambda p: os.path.normpath(str(p))
    resource_allocation_path = normalized_path("tests/unittests/resource_allocation")

    # Reordering logic to ensure tests under ./unittests/resource_allocation run last
    # TODO: Fix this once resource_allocation tests are robost enough to run with other tests without ordering issues
    resource_allocation_tests = [item for item in items if resource_allocation_path in normalized_path(item.fspath)]
    other_tests = [item for item in items if resource_allocation_path not in normalized_path(item.fspath)]

    items.clear()
    items.extend(other_tests)
    items.extend(resource_allocation_tests)


@contextmanager
def mock_system_resourcses(num_cpus=None, num_gpus=None):
    original_get_cpu_count = ResourceManager.get_cpu_count
    original_get_gpu_count = ResourceManager.get_gpu_count
    if num_cpus is not None:
        ResourceManager.get_cpu_count = lambda: num_cpus
    if num_gpus is not None:
        ResourceManager.get_gpu_count = lambda: num_gpus
    yield
    ResourceManager.get_cpu_count = original_get_cpu_count
    ResourceManager.get_gpu_count = original_get_gpu_count


@pytest.fixture
def mock_system_resources_ctx_mgr():
    return mock_system_resourcses


@pytest.fixture
def mock_num_cpus():
    return 16


@pytest.fixture
def mock_num_gpus():
    return 2


@pytest.fixture
def k_fold():
    return 2


@pytest.fixture(autouse=True, scope="session")
def _ag_default_base_path(tmp_path_factory):
    """Redirect auto-generated predictor paths into pytest's temporary directory.

    Without this, every `TabularPredictor(...)` created without an explicit `path` leaves an
    `AutogluonModels/ag-<timestamp>` directory behind in the working directory, and `FitHelper`
    writes its predictors under `./datasets/`. pytest reclaims its own tmp dirs (keeping only the
    last few sessions), so nothing accumulates in the repo.
    """
    base_path = tmp_path_factory.mktemp("ag_models")
    prev = os.environ.get("AG_DEFAULT_BASE_PATH")
    os.environ["AG_DEFAULT_BASE_PATH"] = str(base_path)
    yield str(base_path)
    if prev is None:
        os.environ.pop("AG_DEFAULT_BASE_PATH", None)
    else:
        os.environ["AG_DEFAULT_BASE_PATH"] = prev
