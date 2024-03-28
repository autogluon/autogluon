from uuid import uuid4

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


@pytest.fixture()
def temp_model_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp(str(uuid4())[:6])
    return str(fn)


@pytest.fixture(scope="session")
def hf_model_path(tmp_path_factory):
    """Force HuggingFace to cache the model config once and reuse it from a temporary cache directory.
    This prevents inflating HuggingFace download numbers as an HTTP request is sent every time
    ``ChronosPipeline.from_pretrained`` is called.
    """
    model_hub_id = "amazon/chronos-t5-tiny"
    # get cache path for huggingface model
    cache_path = tmp_path_factory.mktemp("hf_hub_cache")

    try:
        from autogluon.timeseries.models.chronos.pipeline import ChronosPipeline

        # download and cache model from hf hub
        _ = ChronosPipeline.from_pretrained(model_hub_id, cache_dir=str(cache_path))

        # get model snapshot path
        snapshots_path = cache_path / "models--amazon--chronos-t5-tiny" / "snapshots"
        assert snapshots_path.exists()
        snapshot_dir = next(snapshots_path.iterdir())
        assert snapshot_dir.is_dir()

        yield str(snapshot_dir)
    except:
        yield model_hub_id  # fallback to hub id if no snapshots found
