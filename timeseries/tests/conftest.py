import multiprocessing
import os
from uuid import uuid4

import pytest

_HF_HUB_DEPENDENCIES = [
    "autogluon/chronos-t5-tiny",
    "autogluon/chronos-bolt-tiny",
    "autogluon/chronos-2",
]


def download_and_cache_hf_hub_dependencies():
    from transformers import AutoModel

    for dependency in _HF_HUB_DEPENDENCIES:
        _ = AutoModel.from_pretrained(dependency)


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


def pytest_sessionstart():
    """The following is a workaround to cache the dependencies from Hugging Face Hub once and
    run the test session with HF_HUB_OFFLINE, i.e., preventing any HTTP calls from Hugging Face.

    The code first calls `from_pretrained` in order to download and cache the two models (if they aren't
    cached already) in a subprocess, and then sets the HF_HUB_OFFLINE environment variable to True, in
    order to prevent any calls from the main process. The caching has to be done in a subprocess due to
    the way Hugging Face Hub works: if HF_HUB_OFFLINE=1 is set in the main process before importing
    transformers, then the models cannot be downloaded and cached. If it is set after importing transformers,
    HF will have cached HF_HUB_OFFLINE=0 already and the updated environment variable will not take effect.
    """
    process = multiprocessing.Process(target=download_and_cache_hf_hub_dependencies)
    process.start()
    process.join()

    os.environ["HF_HUB_OFFLINE"] = "1"


@pytest.fixture()
def temp_model_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp(str(uuid4())[:6])
    return str(fn)


@pytest.fixture(scope="module")
def dummy_hyperparameters():
    """Hyperparameters passed to the models during tests to minimize training time."""
    return {
        "max_epochs": 1,
        "num_batches_per_epoch": 1,
        "n_jobs": 1,
        "use_fallback_model": False,
        "model_path": "autogluon/chronos-bolt-tiny",
    }


@pytest.fixture(scope="module")
def df_with_static():
    from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
    from .unittests.common import DATAFRAME_WITH_STATIC

    feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=[])
    df = DATAFRAME_WITH_STATIC.copy(deep=False)
    df = feature_generator.fit_transform(df)
    return df, feature_generator.covariate_metadata


@pytest.fixture(scope="module")
def df_with_covariates():
    from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
    from .unittests.common import DATAFRAME_WITH_COVARIATES

    known_covariates_names = [col for col in DATAFRAME_WITH_COVARIATES.columns if col != "target"]
    feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    df = DATAFRAME_WITH_COVARIATES.copy(deep=False)
    df = feature_generator.fit_transform(df)
    return df, feature_generator.covariate_metadata


@pytest.fixture(scope="module")
def df_with_covariates_and_metadata():
    """Create a TimeSeriesDataFrame with covariates & static features.

    Returns the preprocessed TimeSeriesDataFrame and the respective CovariateMetadata.
    """
    from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

    from .unittests.common import DATAFRAME_WITH_STATIC_AND_COVARIATES

    data = DATAFRAME_WITH_STATIC_AND_COVARIATES.copy()
    feat_gen = TimeSeriesFeatureGenerator("target", known_covariates_names=["cov1", "cov2"])
    data = feat_gen.fit_transform(data)
    yield data, feat_gen.covariate_metadata
